"""Script 12 — Évaluation du modèle SFT sur val et test sets + rapport Markdown."""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

# Workaround : Unsloth patche les modèles Qwen3 au niveau de transformers à l'installation.
# Même en chargeant via AutoModelForCausalLM, les forward patchés produisent des tenseurs
# non-contigus qui crashent cuBLAS standard. cublasLt utilise un code path compatible.
torch.backends.cuda.preferred_blas_library("cublaslt")

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import mlflow
import pandas as pd
import torch
from datasets import DatasetDict, load_from_disk
from dotenv import load_dotenv
from peft import PeftModel
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, fbeta_score, recall_score
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)
from utils import SYSTEM_PROMPT, extract_urgency_from_response, get_logger

PROJECT_ROOT = _SCRIPTS_DIR.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

# ── Constantes ────────────────────────────────────────────────────────────────

MODEL_NAME = os.getenv("MODEL_NAME", "unsloth/Qwen3-1.7B")
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "sft"
SFT_FINAL_DIR = PROJECT_ROOT / "data" / "final" / "sft"

MAX_SEQ_LENGTH = 1024
MAX_NEW_TOKENS = 512
DO_SAMPLE = False
SEED = 42
N_EXAMPLES = 10
BATCH_SIZE_EVAL = 8  # examples per GPU batch — reduce to 4 or 2 if CUDA OOM

MLFLOW_EXPERIMENT = "sft-qwen3-1.7b-triage"
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"

URGENCY_LABELS = ["max", "moderate", "deferred"]


# ── Fonctions ─────────────────────────────────────────────────────────────────


def load_training_config(checkpoint_dir: Path) -> dict:
    """Charge les hyperparamètres d'entraînement depuis le checkpoint.

    Lit ``training_config.json`` (sauvegardé par script 11) et complète
    avec les paramètres LoRA depuis ``adapter_config.json`` si disponible.
    Retourne un dict vide avec des valeurs "—" si les fichiers sont absents.

    Args:
        checkpoint_dir: Répertoire du checkpoint SFT.

    Returns:
        Dict des hyperparamètres, valeurs "—" pour les champs manquants.
    """
    config: dict = {}

    training_cfg = checkpoint_dir / "training_config.json"
    if training_cfg.exists():
        config.update(json.loads(training_cfg.read_text(encoding="utf-8")))

    # Complément depuis adapter_config.json si training_config.json absent ou incomplet
    adapter_cfg = checkpoint_dir / "adapter_config.json"
    if adapter_cfg.exists():
        adapter = json.loads(adapter_cfg.read_text(encoding="utf-8"))
        config.setdefault("lora_r", adapter.get("r", "—"))
        config.setdefault("lora_alpha", adapter.get("lora_alpha", "—"))
        config.setdefault("lora_dropout", adapter.get("lora_dropout", "—"))
        config.setdefault("lora_target_modules", adapter.get("target_modules", "—"))
        config.setdefault("model_name", adapter.get("base_model_name_or_path", "—"))

    defaults = ["learning_rate", "epochs", "batch_size", "grad_accum", "max_seq_length", "seed"]
    for key in defaults:
        config.setdefault(key, "—")

    return config


@mlflow.trace(span_type="RETRIEVER", name="load_finetuned_model")
def load_finetuned_model(
    model_name: str,
    checkpoint_dir: Path,
    max_seq_length: int,
) -> tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    """Charge le modèle de base puis applique les poids LoRA depuis le checkpoint.

    Utilise HF standard (pas Unsloth) car les patches Qwen3 d'Unsloth produisent
    des tenseurs non-contigus incompatibles avec model.generate() + PEFT LoRA.

    Args:
        model_name: Identifiant du modèle de base sur HuggingFace Hub.
        checkpoint_dir: Répertoire contenant adapter_model.safetensors.
        max_seq_length: Longueur maximale de séquence.

    Returns:
        Tuple (modèle prêt pour l'inférence, tokenizer).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Qwen3-Base has no chat_template — inject the standard Qwen3 ChatML template
    # so apply_chat_template works in generate_responses_batch.
    if not tokenizer.chat_template:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, str(checkpoint_dir))
    model.eval()
    # Supprime le max_length de la generation_config (Qwen3 le fixe à 32768),
    # ce qui crée un warning quand max_new_tokens est aussi passé à generate().
    model.generation_config.max_length = None  # type: ignore[reportAttributeAccessIssue]
    return model, tokenizer  # type: ignore[reportReturnType]


def generate_responses_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    instructions: list[str],
    max_new_tokens: int = MAX_NEW_TOKENS,
    batch_size: int = BATCH_SIZE_EVAL,
) -> tuple[list[str], int]:
    """Generate responses for a list of instructions using batched GPU inference.

    Sends ``batch_size`` prompts at once to ``model.generate()``, saturating the
    GPU instead of issuing one call per example. Left-padding (set in
    ``load_finetuned_model``) ensures all sequences in a batch are aligned to
    the same length — a requirement for correct batch generation with causal LMs.

    On CUDA OOM for a whole batch, falls back gracefully: empties the cache,
    marks the batch outputs as empty strings, and increments the OOM counter.
    Use a smaller ``--batch-size`` to avoid OOM on low-VRAM GPUs.

    Args:
        model: Fine-tuned model in eval mode.
        tokenizer: Tokenizer with ``padding_side="left"``.
        instructions: List of user instruction strings to evaluate.
        max_new_tokens: Maximum tokens to generate per example.
        batch_size: Number of examples per GPU batch.

    Returns:
        Tuple of (responses, n_oom) where ``responses[i]`` is the decoded
        text for ``instructions[i]`` and ``n_oom`` counts OOM-skipped examples.
    """
    im_end_id: int = tokenizer.convert_tokens_to_ids("<|im_end|>")  # type: ignore[assignment]
    responses: list[str] = [""] * len(instructions)
    n_oom = 0

    for batch_start in tqdm(range(0, len(instructions), batch_size), desc="Generating"):
        batch = instructions[batch_start : batch_start + batch_size]

        # apply_chat_template is the source of truth for special tokens.
        # Pre-fill the <think> block as empty to suppress Qwen3 chain-of-thought:
        # the model would otherwise generate a variable-length <think>...</think>
        # preamble that pushes the urgency label past the 150-char parsing window.
        prompts = [
            str(
                tokenizer.apply_chat_template(  # type: ignore[union-attr]
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": instr},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            + "<think>\n\n</think>\n"
            for instr in batch
        ]

        # Left-padded batch — all sequences share the same input_length
        inputs = tokenizer(  # type: ignore[call-overload]
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        ).to(model.device)  # type: ignore[union-attr]
        input_length: int = inputs["input_ids"].shape[1]  # type: ignore[index]

        try:
            with torch.no_grad():
                output_ids = model.generate(  # type: ignore[reportCallIssue]
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=DO_SAMPLE,
                    temperature=1.0,
                    eos_token_id=im_end_id,
                )
        except RuntimeError as exc:
            if "CUDA out of memory" in str(exc):
                n_oom += len(batch)
                torch.cuda.empty_cache()
                continue  # responses stay "" for this batch
            raise

        for i in range(len(batch)):
            gen_ids = output_ids[i][input_length:]
            eos_pos = (gen_ids == im_end_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                gen_ids = gen_ids[: eos_pos[0]]
            responses[batch_start + i] = str(tokenizer.decode(gen_ids, skip_special_tokens=True))

    return responses, n_oom


def check_format_compliance(text: str) -> bool:
    """Vérifie qu'une réponse respecte le format attendu.

    Le format attendu inclut :
    1. Un indicateur de niveau d'urgence (MAXIMALE/MODÉRÉE/DIFFÉRÉE)
    2. Au moins 3 phrases (évaluation + recommandations minimales)

    Args:
        text: Texte de la réponse générée.

    Returns:
        True si le format est respecté.
    """
    has_urgency = extract_urgency_from_response(text) is not None
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 10]
    has_structure = len(sentences) >= 3
    return has_urgency and has_structure


def evaluate_split(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    df: pd.DataFrame,
    split_name: str,
    n_eval: int | None = None,
    batch_size: int = BATCH_SIZE_EVAL,
    logger=None,
) -> dict:
    """Evaluate the model on a split using batched GPU generation.

    Collects all instructions, sends them to ``generate_responses_batch`` in
    GPU batches, then builds the predictions list from the returned responses.

    Args:
        model: Fine-tuned model.
        tokenizer: Tokenizer with ``padding_side="left"``.
        df: DataFrame with columns instruction, response, urgency_level.
        split_name: Split name used for logging and MLflow spans.
        n_eval: Number of examples to evaluate (None = all).
        batch_size: Examples per GPU batch passed to ``generate_responses_batch``.
        logger: Optional logger instance.

    Returns:
        Dict with metrics and the full predictions list.
    """
    if n_eval is not None and n_eval < len(df):
        df = df.sample(n=n_eval, random_state=SEED).reset_index(drop=True)
        if logger:
            logger.info("[{}] Sous-échantillon de {} exemples.", split_name, n_eval)

    n_total = len(df)
    predictions: list[dict] = []

    with mlflow.start_span(name=f"evaluate_split_{split_name}", span_type="CHAIN") as span:
        span.set_inputs({"split": split_name, "n_total": n_total, "n_requested": n_eval or n_total})
        t0 = time.monotonic()

        instructions: list[str] = list(df["instruction"])
        generated_responses, n_oom = generate_responses_batch(
            model, tokenizer, instructions, batch_size=batch_size
        )
        if n_oom > 0 and logger:
            logger.warning("[{}] {} examples skipped due to CUDA OOM.", split_name, n_oom)

        for i, (_, row) in enumerate(df.iterrows()):
            generated = generated_responses[i]
            predicted_urgency = extract_urgency_from_response(generated)
            predictions.append(
                {
                    "instruction": row["instruction"],
                    "reference_urgency": row["urgency_level"],
                    "predicted_urgency": predicted_urgency,
                    "generated_response": generated,
                    "format_ok": check_format_compliance(generated),
                }
            )

        duration_s = round(time.monotonic() - t0, 1)

        # Calcul des métriques
        y_true_all = [p["reference_urgency"] for p in predictions]
        y_pred_all = [p["predicted_urgency"] for p in predictions]

        n_unparseable = sum(1 for p in y_pred_all if p is None)
        valid_pairs = [
            (t, p) for t, p in zip(y_true_all, y_pred_all, strict=False) if p is not None
        ]

        if valid_pairs:
            y_true = [t for t, _ in valid_pairs]
            y_pred = [p for _, p in valid_pairs]
            accuracy = accuracy_score(y_true, y_pred)
            f1_macro = f1_score(
                y_true,
                y_pred,
                average="macro",
                labels=URGENCY_LABELS,
                zero_division=0,  # type: ignore[reportArgumentType]
            )
            recall_macro = recall_score(
                y_true,
                y_pred,
                average="macro",
                labels=URGENCY_LABELS,
                zero_division=0,  # type: ignore[reportArgumentType]
            )
            f2_macro = fbeta_score(
                y_true,
                y_pred,
                beta=2,
                average="macro",
                labels=URGENCY_LABELS,
                zero_division=0,  # type: ignore[reportArgumentType]
            )
            cm = confusion_matrix(y_true, y_pred, labels=URGENCY_LABELS)
        else:
            accuracy = 0.0
            f1_macro = 0.0
            recall_macro = 0.0
            f2_macro = 0.0
            cm = None

        format_compliance = sum(1 for p in predictions if p["format_ok"]) / len(predictions)
        response_lengths = [len(p["generated_response"].split()) for p in predictions]
        response_length_mean = (
            sum(response_lengths) / len(response_lengths) if response_lengths else 0
        )

        metrics = {
            "accuracy": round(accuracy, 4),
            "f1_macro": round(f1_macro, 4),
            "recall_macro": round(recall_macro, 4),
            "f2_macro": round(f2_macro, 4),
            "format_compliance": round(format_compliance, 4),
            "response_length_mean": round(response_length_mean, 1),
            "n_unparseable": n_unparseable,
            "n_evaluated": len(predictions),
            "confusion_matrix": cm,
            "predictions": predictions,
        }

        if logger:
            logger.info(
                "[{}] accuracy={:.2f}% | f1_macro={:.4f} | recall_macro={:.4f} | f2_macro={:.4f} | format={:.1f}% | unparseable={} | oom={} | {:.0f}s",
                split_name,
                accuracy * 100,
                f1_macro,
                recall_macro,
                f2_macro,
                format_compliance * 100,
                n_unparseable,
                n_oom,
                duration_s,
            )

        span.set_outputs(
            {
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "recall_macro": metrics["recall_macro"],
                "f2_macro": metrics["f2_macro"],
                "format_compliance": metrics["format_compliance"],
                "n_evaluated": metrics["n_evaluated"],
                "n_unparseable": n_unparseable,
                "n_oom": n_oom,
                "duration_s": duration_s,
                "throughput_examples_per_min": round(len(predictions) / duration_s * 60, 1)
                if duration_s > 0
                else 0,
            }
        )

    return metrics


def sample_good_bad_examples(
    predictions: list[dict],
    n: int = N_EXAMPLES,
    seed: int = SEED,
) -> tuple[list[dict], list[dict]]:
    """Échantillonne des exemples de bonnes et mauvaises prédictions.

    Args:
        predictions: Liste des prédictions avec reference et predicted urgency.
        n: Nombre total d'exemples à retourner (n/2 bons + n/2 mauvais).
        seed: Graine pour la reproductibilité.

    Returns:
        Tuple (bons_exemples, mauvais_exemples).
    """
    rng = random.Random(seed)
    good = [p for p in predictions if p["predicted_urgency"] == p["reference_urgency"]]
    bad = [p for p in predictions if p["predicted_urgency"] != p["reference_urgency"]]

    n_half = n // 2
    good_sample = rng.sample(good, min(n_half, len(good)))
    bad_sample = rng.sample(bad, min(n_half, len(bad)))

    return good_sample, bad_sample


def generate_eval_report(
    val_metrics: dict | None,
    test_metrics: dict,
    good_examples: list[dict],
    bad_examples: list[dict],
    run_timestamp: str,
    training_config: dict,
) -> str:
    """Génère un rapport d'évaluation en Markdown.

    Inclut les hyperparamètres d'entraînement (lus depuis le checkpoint),
    les métriques, la matrice de confusion, des exemples de bonnes/mauvaises
    prédictions, et une recommandation.

    Args:
        val_metrics: Métriques sur le val set, ou None si --eval-val non activé.
        test_metrics: Métriques sur le test set (toujours disponible).
        good_examples: Exemples de prédictions correctes.
        bad_examples: Exemples de prédictions incorrectes.
        run_timestamp: Horodatage de l'exécution (format YYYYMMDD_HHMMSS).
        training_config: Hyperparamètres lus depuis le checkpoint.

    Returns:
        Rapport complet en Markdown.
    """

    def _fmt(m: dict | None) -> dict:
        """Formate un dict de métriques en strings. Retourne 'N/A' si m est None."""
        if m is None:
            return {
                k: "N/A"
                for k in [
                    "accuracy",
                    "f1_macro",
                    "recall_macro",
                    "f2_macro",
                    "format_compliance",
                    "response_length_mean",
                    "n_unparseable",
                ]
            }
        return {
            "accuracy": f"{m['accuracy']:.2%}",
            "f1_macro": f"{m['f1_macro']:.4f}",
            "recall_macro": f"{m['recall_macro']:.4f}",
            "f2_macro": f"{m['f2_macro']:.4f}",
            "format_compliance": f"{m['format_compliance']:.1%}",
            "response_length_mean": f"{m['response_length_mean']:.0f} mots",
            "n_unparseable": f"{m['n_unparseable']}/{m['n_evaluated']}",
        }

    v = _fmt(val_metrics)
    t = _fmt(test_metrics)

    date_str = datetime.strptime(run_timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    target_modules = training_config.get("lora_target_modules", "—")
    if isinstance(target_modules, list):
        target_modules = ", ".join(target_modules)

    # Build the metrics table separately to avoid f-string interpolation issues
    # inside a large triple-quoted block (e.g. emoji in SYSTEM_PROMPT scope).
    metrics_table = (
        "| Métrique | Val Set | Test Set |\n"
        "|---|---|---|\n"
        "| Accuracy | " + str(v["accuracy"]) + " | " + str(t["accuracy"]) + " |\n"
        "| F1 Macro | " + str(v["f1_macro"]) + " | " + str(t["f1_macro"]) + " |\n"
        "| Recall Macro | " + str(v["recall_macro"]) + " | " + str(t["recall_macro"]) + " |\n"
        "| F2 Macro (\u03b2=2) | " + str(v["f2_macro"]) + " | " + str(t["f2_macro"]) + " |\n"
        "| Format Compliance | "
        + str(v["format_compliance"])
        + " | "
        + str(t["format_compliance"])
        + " |\n"
        "| Longueur moyenne réponse | "
        + str(v["response_length_mean"])
        + " | "
        + str(t["response_length_mean"])
        + " |\n"
        "| Non-parseables | " + str(v["n_unparseable"]) + " | " + str(t["n_unparseable"]) + " |"
    )

    report = (
        "# Rapport d'Évaluation SFT\n"
        "## project14 — Agent de Triage Médical (Qwen3-1.7B + LoRA)\n\n"
        f"**Date** : {date_str}\n"
        f"**Modèle** : {training_config.get('model_name', MODEL_NAME)}\n"
        f"**Checkpoint** : {CHECKPOINT_DIR}\n\n"
        "### Hyperparamètres d'entraînement\n\n"
        "| Paramètre | Valeur |\n"
        "|---|---|\n"
        f"| LoRA r | {training_config['lora_r']} |\n"
        f"| LoRA alpha | {training_config['lora_alpha']} |\n"
        f"| LoRA dropout | {training_config['lora_dropout']} |\n"
        f"| LoRA target modules | {target_modules} |\n"
        f"| Learning rate | {training_config['learning_rate']} |\n"
        f"| Epochs | {training_config['epochs']} |\n"
        f"| Batch size | {training_config['batch_size']} |\n"
        f"| Gradient accumulation | {training_config['grad_accum']} |\n"
        f"| Max seq length | {training_config['max_seq_length']} |\n"
        f"| Seed | {training_config['seed']} |\n\n"
        "---\n\n"
        "## 1. Métriques\n\n" + metrics_table + "\n\n---\n\n"
        "## 2. Matrice de confusion (Val Set)\n\n"
    )
    if val_metrics is None:
        report += "*Non évalué (utiliser --eval-val pour générer).*\n"
    else:
        cm_val = val_metrics.get("confusion_matrix")
        if cm_val is not None:
            report += "| | Prédit max | Prédit moderate | Prédit deferred |\n"
            report += "|---|---|---|---|\n"
            for i, label in enumerate(URGENCY_LABELS):
                row_vals = " | ".join(str(cm_val[i][j]) for j in range(len(URGENCY_LABELS)))
                report += f"| **Réel {label}** | {row_vals} |\n"
        else:
            report += "Matrice non disponible (aucune prédiction valide).\n"

    report += "\n---\n\n## 3. Matrice de confusion (Test Set)\n\n"
    cm_test = test_metrics.get("confusion_matrix")
    if cm_test is not None:
        report += "| | Prédit max | Prédit moderate | Prédit deferred |\n"
        report += "|---|---|---|---|\n"
        for i, label in enumerate(URGENCY_LABELS):
            row_vals = " | ".join(str(cm_test[i][j]) for j in range(len(URGENCY_LABELS)))
            report += f"| **Réel {label}** | {row_vals} |\n"
    else:
        report += "Matrice non disponible (aucune prédiction valide).\n"

    report += "\n---\n\n## 4. Exemples de bonnes prédictions\n\n"
    for i, ex in enumerate(good_examples, 1):
        report += f"### Exemple {i} (✓ {ex['reference_urgency']})\n\n"
        report += f"**Instruction** : {ex['instruction'][:300]}\n\n"
        report += f"**Réponse générée** : {ex['generated_response'][:500]}\n\n"

    report += "---\n\n## 5. Exemples de mauvaises prédictions\n\n"
    for i, ex in enumerate(bad_examples, 1):
        report += f"### Exemple {i} (attendu: {ex['reference_urgency']}, prédit: {ex['predicted_urgency']})\n\n"
        report += f"**Instruction** : {ex['instruction'][:300]}\n\n"
        report += f"**Réponse générée** : {ex['generated_response'][:500]}\n\n"

    # Recommandation basée sur le test set (référence honnête, toujours disponible)
    acc = test_metrics["accuracy"]
    report += "---\n\n## 6. Recommandation\n\n"
    if val_metrics is None:
        report += "*Note : --eval-val non activé, recommandation basée sur le Test Set.*\n\n"
    if acc >= 0.70:
        report += f"**Accuracy ({acc:.1%}) >= 70%** : Modèle prêt pour la phase DPO (semaine 3).\n"
    elif acc >= 0.60:
        report += (
            f"**Accuracy ({acc:.1%}) >= 60%** : Acceptable pour un POC. "
            "Ajuster les hyperparamètres (augmenter epochs, lr, ou r) pour améliorer.\n"
        )
    else:
        report += (
            f"**Accuracy ({acc:.1%}) < 60%** : Insuffisant. "
            "Revoir le dataset (distribution, qualité des réponses) ou l'architecture (r, target_modules).\n"
        )

    return report


def main() -> None:
    """Pipeline d'évaluation SFT : génération + métriques + rapport."""
    parser = argparse.ArgumentParser(
        description="Évaluation du modèle SFT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Note sur --eval-val :\n"
            "  Le val set est utilisé pendant l'entraînement pour sélectionner le\n"
            "  meilleur checkpoint (load_best_model_at_end=True). Les métriques val\n"
            "  sont donc biaisées (le modèle a été optimisé sur ce split).\n"
            "  Seul le test set fournit une estimation honnête des performances.\n"
            "  --eval-val est utile uniquement pour debugger ou vérifier la cohérence\n"
            "  avec les métriques du Trainer (~3h30 de GPU supplémentaires)."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Logging DEBUG")
    parser.add_argument(
        "--n-eval",
        type=int,
        default=None,
        help="Nombre d'exemples à évaluer par split (None = tous). Utile pour le debug.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE_EVAL,
        help=f"Exemples par batch GPU (défaut: {BATCH_SIZE_EVAL}). Réduire à 4 ou 2 si CUDA OOM.",
    )
    parser.add_argument(
        "--eval-val",
        action="store_true",
        default=False,
        help="Évalue aussi sur le val set (biaisé — voir note ci-dessous). Désactivé par défaut.",
    )
    args = parser.parse_args()

    logger = get_logger("12_evaluate_sft", verbose=args.verbose)

    # Vérification du checkpoint
    adapter_path = CHECKPOINT_DIR / "adapter_model.safetensors"
    if not adapter_path.exists():
        logger.error(
            "Checkpoint LoRA non trouvé : %s. Lancer 11_train_sft.py d'abord.", adapter_path
        )
        sys.exit(1)

    # Vérification du dataset
    if not SFT_FINAL_DIR.exists():
        logger.error("Dataset manquant : {}", SFT_FINAL_DIR)
        sys.exit(1)

    # Le start_run en début de main() rattache tous les spans @mlflow.trace
    # et mlflow.start_span() au même run (évite les traces orphelines).
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    # System metrics : CPU, RAM, GPU utilization + VRAM (nécessite pynvml)
    mlflow.enable_system_metrics_logging()
    with mlflow.start_run(run_name="eval-sft"):
        # Chargement du modèle
        logger.info("Chargement du modèle fine-tuné depuis {}...", CHECKPOINT_DIR)
        model, tokenizer = load_finetuned_model(MODEL_NAME, CHECKPOINT_DIR, MAX_SEQ_LENGTH)
        logger.info("Modèle chargé en mode inférence.")

        # Chargement des données
        sft = DatasetDict(load_from_disk(str(SFT_FINAL_DIR)))  # type: ignore[arg-type]
        # urgency_level est encodé en ClassLabel (entiers) depuis le script 04.
        # On décode en strings pour que y_true soit homogène avec les prédictions
        # ("max", "moderate", "deferred") retournées par extract_urgency_from_response.
        urgency_feature = sft["test"].features["urgency_level"]
        df_test = pd.DataFrame(sft["test"].to_pandas())
        df_test["urgency_level"] = df_test["urgency_level"].map(urgency_feature.int2str)
        logger.info("Test: {} exemples", len(df_test))

        # Évaluation val (optionnelle — biaisée, désactivée par défaut)
        val_metrics = None
        if args.eval_val:
            logger.warning(
                "--eval-val activé : le val set a servi à sélectionner le checkpoint "
                "(load_best_model_at_end=True) — métriques biaisées, ~3h30 supplémentaires."
            )
            df_val = pd.DataFrame(sft["val"].to_pandas())
            df_val["urgency_level"] = df_val["urgency_level"].map(urgency_feature.int2str)
            val_metrics = evaluate_split(
                model,
                tokenizer,
                df_val,
                "val",
                n_eval=args.n_eval,
                batch_size=args.batch_size,
                logger=logger,
            )

        # Évaluation test (référence honnête)
        logger.info("Évaluation sur le test set...")
        test_metrics = evaluate_split(
            model,
            tokenizer,
            df_test,
            "test",
            n_eval=args.n_eval,
            batch_size=args.batch_size,
            logger=logger,
        )

        # Exemples (depuis test si val non disponible)
        source_predictions = (
            val_metrics["predictions"] if val_metrics else test_metrics["predictions"]
        )
        good_ex, bad_ex = sample_good_bad_examples(source_predictions)

        # Rapport — nom horodaté pour ne pas écraser les évaluations précédentes
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = PROJECT_ROOT / "reports" / "sft" / f"eval_report_{run_timestamp}.md"
        training_config = load_training_config(CHECKPOINT_DIR)
        report = generate_eval_report(
            val_metrics,
            test_metrics,
            good_ex,
            bad_ex,
            run_timestamp,
            training_config,
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        logger.info("Rapport d'évaluation sauvegardé dans {}", report_path)

        # Métriques + artefacts
        metrics_to_log = {
            "test_accuracy": test_metrics["accuracy"],
            "test_f1_macro": test_metrics["f1_macro"],
            "test_recall_macro": test_metrics["recall_macro"],
            "test_f2_macro": test_metrics["f2_macro"],
            "test_format_compliance": test_metrics["format_compliance"],
        }
        if val_metrics:
            metrics_to_log.update(
                {
                    "val_accuracy": val_metrics["accuracy"],
                    "val_f1_macro": val_metrics["f1_macro"],
                    "val_recall_macro": val_metrics["recall_macro"],
                    "val_f2_macro": val_metrics["f2_macro"],
                    "val_format_compliance": val_metrics["format_compliance"],
                    "val_n_unparseable": val_metrics["n_unparseable"],
                }
            )
        mlflow.log_metrics(metrics_to_log)
        mlflow.log_artifact(str(report_path))

    logger.info("=== Évaluation terminée. ===")


if __name__ == "__main__":
    main()
