"""Script 12 — Évaluation du modèle SFT sur val et test sets + rapport Markdown."""

import argparse
import random
import sys
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
from peft import PeftModel
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerFast

from utils import extract_urgency_from_response, format_chat_prompt, get_logger

PROJECT_ROOT = _SCRIPTS_DIR.parent

# ── Constantes ────────────────────────────────────────────────────────────────

MODEL_NAME = "unsloth/Qwen3-1.7B-Base"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "sft"
SFT_VAL_PATH = PROJECT_ROOT / "data" / "final" / "sft_val.parquet"
SFT_TEST_PATH = PROJECT_ROOT / "data" / "final" / "sft_test.parquet"
REPORT_PATH = CHECKPOINT_DIR / "eval_report.md"

MAX_SEQ_LENGTH = 1024
MAX_NEW_TOKENS = 512
DO_SAMPLE = False
SEED = 42
N_EXAMPLES = 10

MLFLOW_EXPERIMENT = "sft-qwen3-1.7b-triage"
MLFLOW_TRACKING_URI = str(PROJECT_ROOT / "mlruns")

URGENCY_LABELS = ["max", "moderate", "deferred"]


# ── Fonctions ─────────────────────────────────────────────────────────────────


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

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, str(checkpoint_dir))
    model.eval()
    # Supprime le max_length de la generation_config (Qwen3 le fixe à 32768),
    # ce qui crée un warning quand max_new_tokens est aussi passé à generate().
    model.generation_config.max_length = None
    return model, tokenizer


def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    instruction: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Génère une réponse à partir d'une instruction en mode greedy.

    Formate l'instruction en ChatML (sans le tour réponse), tokenise,
    génère, puis décode uniquement les tokens produits jusqu'au premier
    token <|im_end|> (EOS du tour assistant en ChatML Qwen3).

    Args:
        model: Modèle fine-tuné prêt pour l'inférence.
        tokenizer: Tokenizer associé.
        instruction: Texte de l'instruction utilisateur.
        max_new_tokens: Nombre max de tokens à générer.

    Returns:
        Texte de la réponse générée, tronqué au premier <|im_end|>.
    """
    with mlflow.start_span(name="generate_response", span_type="LLM") as span:
        span.set_inputs({"instruction": instruction[:300]})

        prompt = format_chat_prompt(instruction, response="")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_length = inputs["input_ids"].shape[1]

        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=DO_SAMPLE,
                temperature=1.0,
                eos_token_id=im_end_id,
            )

        generated_ids = output_ids[0][input_length:]
        eos_positions = (generated_ids == im_end_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            generated_ids = generated_ids[:eos_positions[0]]

        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        span.set_outputs({"response": response[:300]})
        return response


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
    logger=None,
) -> dict:
    """Évalue le modèle sur un split en générant des réponses.

    Pour chaque exemple, génère une réponse, extrait le niveau d'urgence
    prédit, et compare avec la référence.

    Args:
        model: Modèle fine-tuné.
        tokenizer: Tokenizer associé.
        df: DataFrame avec colonnes instruction, response, urgency_level.
        split_name: Nom du split pour le logging.
        n_eval: Nombre d'exemples à évaluer (None = tous).
        logger: Logger optionnel.

    Returns:
        Dictionnaire avec métriques et liste des prédictions.
    """
    if n_eval is not None and n_eval < len(df):
        df = df.sample(n=n_eval, random_state=SEED).reset_index(drop=True)
        if logger:
            logger.info("[%s] Sous-échantillon de %d exemples.", split_name, n_eval)

    predictions: list[dict] = []
    for idx in tqdm(range(len(df)), desc=f"Évaluation {split_name}"):
        row = df.iloc[idx]
        try:
            generated = generate_response(model, tokenizer, row["instruction"])
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                if logger:
                    logger.warning("OOM à l'exemple %d, skip.", idx)
                torch.cuda.empty_cache()
                generated = ""
            else:
                raise

        predicted_urgency = extract_urgency_from_response(generated)
        predictions.append({
            "instruction": row["instruction"],
            "reference_urgency": row["urgency_level"],
            "predicted_urgency": predicted_urgency,
            "generated_response": generated,
            "format_ok": check_format_compliance(generated),
        })

    # Calcul des métriques
    y_true_all = [p["reference_urgency"] for p in predictions]
    y_pred_all = [p["predicted_urgency"] for p in predictions]

    n_unparseable = sum(1 for p in y_pred_all if p is None)
    valid_pairs = [(t, p) for t, p in zip(y_true_all, y_pred_all) if p is not None]

    if valid_pairs:
        y_true = [t for t, _ in valid_pairs]
        y_pred = [p for _, p in valid_pairs]
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro", labels=URGENCY_LABELS, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=URGENCY_LABELS)
    else:
        accuracy = 0.0
        f1_macro = 0.0
        cm = None

    format_compliance = sum(1 for p in predictions if p["format_ok"]) / len(predictions)
    response_lengths = [len(p["generated_response"].split()) for p in predictions]
    response_length_mean = sum(response_lengths) / len(response_lengths) if response_lengths else 0

    metrics = {
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "format_compliance": round(format_compliance, 4),
        "response_length_mean": round(response_length_mean, 1),
        "n_unparseable": n_unparseable,
        "n_evaluated": len(predictions),
        "confusion_matrix": cm,
        "predictions": predictions,
    }

    if logger:
        logger.info(
            "[%s] accuracy=%.2f%% | f1_macro=%.4f | format=%.1f%% | unparseable=%d/%d",
            split_name, accuracy * 100, f1_macro, format_compliance * 100,
            n_unparseable, len(predictions),
        )

    with mlflow.start_span(name=f"evaluate_split_{split_name}", span_type="CHAIN") as span:
        span.set_inputs({"split": split_name, "n_evaluated": metrics["n_evaluated"]})
        span.set_outputs({
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "format_compliance": metrics["format_compliance"],
            "n_unparseable": metrics["n_unparseable"],
        })

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
    val_metrics: dict,
    test_metrics: dict,
    good_examples: list[dict],
    bad_examples: list[dict],
) -> str:
    """Génère un rapport d'évaluation en Markdown.

    Inclut les métriques, la matrice de confusion, des exemples
    de bonnes/mauvaises prédictions, et une recommandation.

    Args:
        val_metrics: Métriques sur le val set.
        test_metrics: Métriques sur le test set.
        good_examples: Exemples de prédictions correctes.
        bad_examples: Exemples de prédictions incorrectes.

    Returns:
        Rapport complet en Markdown.
    """
    report = f"""# Rapport d'Évaluation SFT
## project14 — Agent de Triage Médical (Qwen3-1.7B + LoRA)

**Date** : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Modèle** : {MODEL_NAME}
**Checkpoint** : {CHECKPOINT_DIR}

---

## 1. Métriques

| Métrique | Val Set | Test Set |
|---|---|---|
| Accuracy | {val_metrics['accuracy']:.2%} | {test_metrics['accuracy']:.2%} |
| F1 Macro | {val_metrics['f1_macro']:.4f} | {test_metrics['f1_macro']:.4f} |
| Format Compliance | {val_metrics['format_compliance']:.1%} | {test_metrics['format_compliance']:.1%} |
| Longueur moyenne réponse | {val_metrics['response_length_mean']:.0f} mots | {test_metrics['response_length_mean']:.0f} mots |
| Non-parseables | {val_metrics['n_unparseable']}/{val_metrics['n_evaluated']} | {test_metrics['n_unparseable']}/{test_metrics['n_evaluated']} |

---

## 2. Matrice de confusion (Val Set)

"""
    cm = val_metrics.get("confusion_matrix")
    if cm is not None:
        report += "| | Prédit max | Prédit moderate | Prédit deferred |\n"
        report += "|---|---|---|---|\n"
        for i, label in enumerate(URGENCY_LABELS):
            row_vals = " | ".join(str(cm[i][j]) for j in range(len(URGENCY_LABELS)))
            report += f"| **Réel {label}** | {row_vals} |\n"
    else:
        report += "Matrice non disponible (aucune prédiction valide).\n"

    report += "\n---\n\n## 3. Exemples de bonnes prédictions\n\n"
    for i, ex in enumerate(good_examples, 1):
        report += f"### Exemple {i} (✓ {ex['reference_urgency']})\n\n"
        report += f"**Instruction** : {ex['instruction'][:300]}\n\n"
        report += f"**Réponse générée** : {ex['generated_response'][:500]}\n\n"

    report += "---\n\n## 4. Exemples de mauvaises prédictions\n\n"
    for i, ex in enumerate(bad_examples, 1):
        report += f"### Exemple {i} (attendu: {ex['reference_urgency']}, prédit: {ex['predicted_urgency']})\n\n"
        report += f"**Instruction** : {ex['instruction'][:300]}\n\n"
        report += f"**Réponse générée** : {ex['generated_response'][:500]}\n\n"

    # Recommandation
    acc = val_metrics["accuracy"]
    report += "---\n\n## 5. Recommandation\n\n"
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
    parser = argparse.ArgumentParser(description="Évaluation du modèle SFT")
    parser.add_argument("--verbose", action="store_true", help="Logging DEBUG")
    parser.add_argument(
        "--n-eval", type=int, default=None,
        help="Nombre d'exemples à évaluer par split (None = tous). Utile pour le debug.",
    )
    args = parser.parse_args()

    logger = get_logger("12_evaluate_sft", verbose=args.verbose)

    # Vérification du checkpoint
    adapter_path = CHECKPOINT_DIR / "adapter_model.safetensors"
    if not adapter_path.exists():
        logger.error("Checkpoint LoRA non trouvé : %s. Lancer 11_train_sft.py d'abord.", adapter_path)
        sys.exit(1)

    # Vérification des fichiers de données
    for path in [SFT_VAL_PATH, SFT_TEST_PATH]:
        if not path.exists():
            logger.error("Fichier manquant : %s", path)
            sys.exit(1)

    # Le start_run en début de main() rattache tous les spans @mlflow.trace
    # et mlflow.start_span() au même run (évite les traces orphelines).
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    # System metrics : CPU, RAM, GPU utilization + VRAM (nécessite pynvml)
    mlflow.enable_system_metrics_logging()
    with mlflow.start_run(run_name="eval-sft"):
        # Chargement du modèle
        logger.info("Chargement du modèle fine-tuné depuis %s...", CHECKPOINT_DIR)
        model, tokenizer = load_finetuned_model(MODEL_NAME, CHECKPOINT_DIR, MAX_SEQ_LENGTH)
        logger.info("Modèle chargé en mode inférence.")

        # Chargement des données
        df_val = pd.read_parquet(SFT_VAL_PATH)
        df_test = pd.read_parquet(SFT_TEST_PATH)
        logger.info("Val: %d exemples | Test: %d exemples", len(df_val), len(df_test))

        # Évaluation
        logger.info("Évaluation sur le val set...")
        val_metrics = evaluate_split(model, tokenizer, df_val, "val", n_eval=args.n_eval, logger=logger)

        logger.info("Évaluation sur le test set...")
        test_metrics = evaluate_split(model, tokenizer, df_test, "test", n_eval=args.n_eval, logger=logger)

        # Exemples
        good_ex, bad_ex = sample_good_bad_examples(val_metrics["predictions"])

        # Rapport
        report = generate_eval_report(val_metrics, test_metrics, good_ex, bad_ex)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(report, encoding="utf-8")
        logger.info("Rapport d'évaluation sauvegardé dans %s", REPORT_PATH)

        # Métriques + artefacts
        mlflow.log_metrics({
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_format_compliance": val_metrics["format_compliance"],
            "val_n_unparseable": val_metrics["n_unparseable"],
            "test_accuracy": test_metrics["accuracy"],
            "test_f1_macro": test_metrics["f1_macro"],
            "test_format_compliance": test_metrics["format_compliance"],
        })
        mlflow.log_artifact(str(REPORT_PATH))

    logger.info("=== Évaluation terminée. ===")


if __name__ == "__main__":
    main()
