"""Script 21 — Comparaison SFT vs DPO : accuracy, F1 et qualité clinique.

Évalue les deux checkpoints sur sft_val et sft_test, puis génère un rapport
de comparaison avant/après alignement avec exemples qualitatifs.
"""

import argparse
import re
import random
import sys
from datetime import datetime
from pathlib import Path

import torch

torch.backends.cuda.preferred_blas_library("cublaslt")

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import mlflow
import pandas as pd
from peft import PeftModel
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerFast

from utils import (
    extract_urgency_from_response,
    format_chat_prompt,
    format_dpo_prompt,
    get_logger,
)

PROJECT_ROOT = _SCRIPTS_DIR.parent

# ── Constantes ────────────────────────────────────────────────────────────────

MODEL_NAME = "unsloth/Qwen3-1.7B-Base"
SFT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "sft"
DPO_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "dpo"
REPORT_PATH = DPO_CHECKPOINT / "eval_report.md"

SFT_VAL_PATH = PROJECT_ROOT / "data" / "final" / "sft_val.parquet"
SFT_TEST_PATH = PROJECT_ROOT / "data" / "final" / "sft_test.parquet"
DPO_VAL_PATH = PROJECT_ROOT / "data" / "final" / "dpo_val.parquet"

MAX_SEQ_LENGTH = 1024
MAX_NEW_TOKENS = 512
DO_SAMPLE = False
N_EXAMPLES = 5        # exemples de comparaison qualitative SFT vs DPO
SEED = 42

URGENCY_LABELS = ["max", "moderate", "deferred"]

MLFLOW_EXPERIMENT = "dpo-qwen3-1.7b-triage"
MLFLOW_TRACKING_URI = str(PROJECT_ROOT / "mlruns")

# Regex pour supprimer les artifacts de génération Qwen3 :
# - ForCanBeConverted, 𫟦, caractères de remplacement Unicode (U+FFFD)
# - Reprise hallucinée de conversation : \nuser\n... ou \nassistant\n...
_ARTIFACT_RE = re.compile(
    r"(ForCanBeConverted|𫟦|\uFFFD+|\n\s*(?:user|assistant)\s*\n.*)",
    re.DOTALL | re.IGNORECASE,
)


def _strip_artifacts(text: str) -> str:
    """Supprime les tokens parasites connus produits par Qwen3.

    Args:
        text: Texte décodé brut depuis le tokenizer.

    Returns:
        Texte nettoyé, tronqué avant toute reprise hallucinée de conversation.
    """
    return _ARTIFACT_RE.sub("", text).strip()


# ── Fonctions ─────────────────────────────────────────────────────────────────


@mlflow.trace(span_type="RETRIEVER", name="load_model")
def load_model(
    model_name: str,
    checkpoint_dir: Path,
) -> tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    """Charge le modèle de base et applique les poids LoRA depuis checkpoint_dir.

    Utilise AutoModelForCausalLM (pas Unsloth) pour éviter les patchs Qwen3
    qui produisent des tenseurs non-contigus lors de model.generate().

    Args:
        model_name: Identifiant HuggingFace du modèle de base.
        checkpoint_dir: Répertoire contenant adapter_model.safetensors.

    Returns:
        Tuple (modèle en mode inférence, tokenizer).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, str(checkpoint_dir))
    model.eval()
    return model, tokenizer


@mlflow.trace(span_type="RETRIEVER", name="load_sft_merged_for_dpo_eval")
def load_sft_merged_for_dpo_eval(
    model_name: str,
    sft_checkpoint: Path,
    dpo_checkpoint: Path,
) -> tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    """Charge le modèle DPO : base + SFT merged + DPO LoRA.

    Le checkpoint DPO ne contient que le LoRA DPO. Il a été entraîné sur
    un modèle qui avait déjà les poids SFT fusionnés. On doit reconstruire
    la même base (base + SFT merged) avant d'appliquer le LoRA DPO.

    Args:
        model_name: Identifiant HuggingFace du modèle de base.
        sft_checkpoint: Répertoire du checkpoint LoRA SFT.
        dpo_checkpoint: Répertoire du checkpoint LoRA DPO.

    Returns:
        Tuple (modèle DPO en mode inférence, tokenizer).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Charger le modèle de base
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # 2. Appliquer les poids SFT et les fusionner
    model = PeftModel.from_pretrained(base, str(sft_checkpoint))
    model = model.merge_and_unload()

    # 3. Appliquer le LoRA DPO
    model = PeftModel.from_pretrained(model, str(dpo_checkpoint))
    model.eval()
    return model, tokenizer


def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    instruction: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Génère une réponse en mode greedy depuis une instruction.

    Args:
        model: Modèle (SFT ou DPO) prêt pour l'inférence.
        tokenizer: Tokenizer associé.
        instruction: Texte de l'instruction médicale.
        max_new_tokens: Nombre max de tokens à générer.

    Returns:
        Texte de la réponse générée (sans le prompt, tronqué au premier <|im_end|>).
    """
    with mlflow.start_span(name="generate_response", span_type="LLM") as span:
        span.set_inputs({"instruction": instruction[:300]})

        prompt = format_chat_prompt(instruction, response="")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_length = inputs["input_ids"].shape[1]

        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        stop_ids: list[int] = [im_end_id]
        if tokenizer.eos_token_id is not None and tokenizer.eos_token_id != im_end_id:
            stop_ids.append(tokenizer.eos_token_id)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=DO_SAMPLE,
                temperature=1.0,
                eos_token_id=stop_ids,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        generated_ids = output_ids[0][input_length:]
        for stop_id in stop_ids:
            positions = (generated_ids == stop_id).nonzero(as_tuple=True)[0]
            if len(positions) > 0:
                generated_ids = generated_ids[: positions[0]]
                break

        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        response = _strip_artifacts(text)
        span.set_outputs({"response": response[:300]})
        return response


def check_format_compliance(text: str) -> bool:
    """Vérifie que la réponse contient un label d'urgence et une structure minimale.

    Args:
        text: Texte de la réponse générée.

    Returns:
        True si le format est respecté.
    """
    has_urgency = extract_urgency_from_response(text) is not None
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 10]
    return has_urgency and len(sentences) >= 3


def evaluate_split(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    df: pd.DataFrame,
    split_name: str,
    n_eval: int | None = None,
    logger=None,
) -> dict:
    """Évalue l'accuracy de classification d'urgence sur un split SFT.

    Args:
        model: Modèle à évaluer (SFT ou DPO).
        tokenizer: Tokenizer associé.
        df: DataFrame SFT avec colonnes [instruction, urgency_level].
        split_name: Nom du split pour les logs.
        n_eval: Nombre d'exemples à évaluer (None = tous).
        logger: Logger optionnel.

    Returns:
        Dict {accuracy, f1_macro, format_compliance, n_unparseable, n_evaluated, predictions}.
    """
    if n_eval is not None and n_eval < len(df):
        df = df.sample(n=n_eval, random_state=SEED).reset_index(drop=True)

    predictions: list[dict] = []
    for idx in tqdm(range(len(df)), desc=f"Éval {split_name}"):
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
        predicted = extract_urgency_from_response(generated)
        predictions.append({
            "instruction": row["instruction"],
            "reference_urgency": row["urgency_level"],
            "predicted_urgency": predicted,
            "generated_response": generated,
            "format_ok": check_format_compliance(generated),
        })

    valid = [(p["reference_urgency"], p["predicted_urgency"]) for p in predictions if p["predicted_urgency"]]
    n_unparseable = len(predictions) - len(valid)

    if valid:
        y_true = [t for t, _ in valid]
        y_pred = [p for _, p in valid]
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", labels=URGENCY_LABELS, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=URGENCY_LABELS)
    else:
        accuracy, f1, cm = 0.0, 0.0, None

    format_compliance = sum(p["format_ok"] for p in predictions) / len(predictions)
    avg_len = sum(len(p["generated_response"].split()) for p in predictions) / len(predictions)

    metrics = {
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1, 4),
        "format_compliance": round(format_compliance, 4),
        "response_length_mean": round(avg_len, 1),
        "n_unparseable": n_unparseable,
        "n_evaluated": len(predictions),
        "confusion_matrix": cm,
        "predictions": predictions,
    }

    if logger:
        logger.info(
            "[%s] accuracy=%.2f%% | f1=%.4f | format=%.1f%% | unparseable=%d/%d",
            split_name,
            accuracy * 100,
            f1,
            format_compliance * 100,
            n_unparseable,
            len(predictions),
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


def compare_responses_on_dpo_val(
    sft_model: PreTrainedModel,
    dpo_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    dpo_val_path: Path,
    n: int = N_EXAMPLES,
    seed: int = SEED,
) -> list[dict]:
    """Génère des réponses avec SFT et DPO sur les prompts du val DPO.

    Ces prompts sont issus d'UltraMedical-Preference et permettent d'observer
    l'impact de l'alignement sur des questions médicales réelles.

    Args:
        sft_model: Modèle SFT.
        dpo_model: Modèle DPO.
        tokenizer: Tokenizer partagé.
        dpo_val_path: Chemin vers dpo_val.parquet.
        n: Nombre de comparaisons à générer.
        seed: Graine pour le sous-échantillonnage.

    Returns:
        Liste de dicts {prompt, sft_response, dpo_response}.
    """
    df = pd.read_parquet(dpo_val_path).sample(n=min(n, 100), random_state=seed).head(n)
    comparisons = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Comparaisons SFT vs DPO"):
        prompt = row["prompt"]
        sft_resp = generate_response(sft_model, tokenizer, prompt)
        dpo_resp = generate_response(dpo_model, tokenizer, prompt)
        comparisons.append({
            "prompt": prompt,
            "sft_response": sft_resp,
            "dpo_response": dpo_resp,
            "chosen_reference": row.get("chosen", ""),
        })
    return comparisons


def generate_dpo_eval_report(
    sft_val_metrics: dict,
    dpo_val_metrics: dict,
    sft_test_metrics: dict,
    dpo_test_metrics: dict,
    comparisons: list[dict],
) -> str:
    """Génère un rapport Markdown de comparaison SFT vs DPO.

    Args:
        sft_val_metrics: Métriques SFT sur val set.
        dpo_val_metrics: Métriques DPO sur val set.
        sft_test_metrics: Métriques SFT sur test set.
        dpo_test_metrics: Métriques DPO sur test set.
        comparisons: Comparaisons qualitatives SFT vs DPO.

    Returns:
        Rapport complet en Markdown.
    """

    def delta(a: float, b: float, pct: bool = False) -> str:
        diff = b - a
        sign = "+" if diff >= 0 else ""
        if pct:
            return f"{sign}{diff * 100:.1f}%"
        return f"{sign}{diff:.4f}"

    report = f"""# Rapport d'Évaluation DPO
## project14 — Agent de Triage Médical (Qwen3-1.7B + SFT + DPO)

**Date** : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Modèle de base** : {MODEL_NAME}
**Checkpoint SFT** : {SFT_CHECKPOINT}
**Checkpoint DPO** : {DPO_CHECKPOINT}

---

## 1. Métriques comparées — Val Set

| Métrique | SFT | DPO | Delta |
|---|---|---|---|
| Accuracy | {sft_val_metrics['accuracy']:.2%} | {dpo_val_metrics['accuracy']:.2%} | {delta(sft_val_metrics['accuracy'], dpo_val_metrics['accuracy'], pct=True)} |
| F1 Macro | {sft_val_metrics['f1_macro']:.4f} | {dpo_val_metrics['f1_macro']:.4f} | {delta(sft_val_metrics['f1_macro'], dpo_val_metrics['f1_macro'])} |
| Format Compliance | {sft_val_metrics['format_compliance']:.1%} | {dpo_val_metrics['format_compliance']:.1%} | {delta(sft_val_metrics['format_compliance'], dpo_val_metrics['format_compliance'], pct=True)} |
| Non-parseables | {sft_val_metrics['n_unparseable']}/{sft_val_metrics['n_evaluated']} | {dpo_val_metrics['n_unparseable']}/{dpo_val_metrics['n_evaluated']} | — |
| Longueur réponse (mots) | {sft_val_metrics['response_length_mean']:.0f} | {dpo_val_metrics['response_length_mean']:.0f} | {delta(sft_val_metrics['response_length_mean'], dpo_val_metrics['response_length_mean'])} |

## 2. Métriques comparées — Test Set

| Métrique | SFT | DPO | Delta |
|---|---|---|---|
| Accuracy | {sft_test_metrics['accuracy']:.2%} | {dpo_test_metrics['accuracy']:.2%} | {delta(sft_test_metrics['accuracy'], dpo_test_metrics['accuracy'], pct=True)} |
| F1 Macro | {sft_test_metrics['f1_macro']:.4f} | {dpo_test_metrics['f1_macro']:.4f} | {delta(sft_test_metrics['f1_macro'], dpo_test_metrics['f1_macro'])} |
| Format Compliance | {sft_test_metrics['format_compliance']:.1%} | {dpo_test_metrics['format_compliance']:.1%} | {delta(sft_test_metrics['format_compliance'], dpo_test_metrics['format_compliance'], pct=True)} |

---

## 3. Matrice de confusion DPO (Val Set)

"""
    cm = dpo_val_metrics.get("confusion_matrix")
    if cm is not None:
        report += "| | Prédit max | Prédit moderate | Prédit deferred |\n"
        report += "|---|---|---|---|\n"
        for i, label in enumerate(URGENCY_LABELS):
            row_vals = " | ".join(str(cm[i][j]) for j in range(len(URGENCY_LABELS)))
            report += f"| **Réel {label}** | {row_vals} |\n"
    else:
        report += "Matrice non disponible.\n"

    report += "\n---\n\n## 4. Comparaisons qualitatives SFT vs DPO\n\n"
    for i, comp in enumerate(comparisons, 1):
        report += f"### Exemple {i}\n\n"
        report += f"**Prompt** : {comp['prompt'][:300]}\n\n"
        report += f"**Réponse SFT** :\n{comp['sft_response'][:600]}\n\n"
        report += f"**Réponse DPO** :\n{comp['dpo_response'][:600]}\n\n"
        if comp["chosen_reference"]:
            report += f"**Référence chosen** :\n{comp['chosen_reference'][:400]}\n\n"
        report += "---\n\n"

    # Recommandation
    acc_delta = dpo_val_metrics["accuracy"] - sft_val_metrics["accuracy"]
    report += "## 5. Recommandation\n\n"
    if acc_delta >= -0.03 and dpo_val_metrics["accuracy"] >= 0.65:
        report += (
            f"**DPO validé** : accuracy DPO ({dpo_val_metrics['accuracy']:.1%}) "
            f"sans régression significative (delta={acc_delta:+.1%}). "
            "Prêt pour l'export (22_export_model.py).\n"
        )
    elif acc_delta < -0.05:
        report += (
            f"**Régression détectée** : accuracy DPO ({dpo_val_metrics['accuracy']:.1%}) "
            f"chute de {abs(acc_delta):.1%} vs SFT. "
            "Augmenter beta (0.1 → 0.3) pour renforcer la contrainte KL, "
            "ou réduire à 1 epoch.\n"
        )
    else:
        report += (
            f"**Résultat mitigé** : delta accuracy = {acc_delta:+.1%}. "
            "Analyse qualitative nécessaire avant export.\n"
        )

    return report


def main() -> None:
    """Pipeline d'évaluation DPO : compare SFT vs DPO + génère le rapport."""
    parser = argparse.ArgumentParser(description="Évaluation comparative SFT vs DPO")
    parser.add_argument("--verbose", action="store_true", help="Logging DEBUG")
    parser.add_argument(
        "--n-eval", type=int, default=None,
        help="Nombre d'exemples par split (None = tous). Utile pour le debug.",
    )
    parser.add_argument(
        "--n-comparisons", type=int, default=N_EXAMPLES,
        help="Nombre de comparaisons qualitatives SFT vs DPO.",
    )
    args = parser.parse_args()

    logger = get_logger("21_evaluate_dpo", verbose=args.verbose)

    # Vérifications
    for path in [SFT_CHECKPOINT / "adapter_model.safetensors",
                 DPO_CHECKPOINT / "adapter_model.safetensors"]:
        if not path.exists():
            logger.error("Checkpoint manquant : %s", path)
            sys.exit(1)

    for path in [SFT_VAL_PATH, SFT_TEST_PATH, DPO_VAL_PATH]:
        if not path.exists():
            logger.error("Fichier manquant : %s", path)
            sys.exit(1)

    # Le start_run en début de main() rattache tous les spans @mlflow.trace
    # et mlflow.start_span() au même run (évite les traces orphelines).
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="eval-dpo"):
        # Chargement des données
        df_val = pd.read_parquet(SFT_VAL_PATH)
        df_test = pd.read_parquet(SFT_TEST_PATH)
        logger.info("Val: %d | Test: %d exemples SFT", len(df_val), len(df_test))

        # ── Évaluation SFT ────────────────────────────────────────────────────
        logger.info("Chargement du modèle SFT...")
        sft_model, sft_tokenizer = load_model(MODEL_NAME, SFT_CHECKPOINT)

        logger.info("Évaluation SFT sur val...")
        sft_val = evaluate_split(sft_model, sft_tokenizer, df_val, "sft-val", args.n_eval, logger)
        logger.info("Évaluation SFT sur test...")
        sft_test = evaluate_split(sft_model, sft_tokenizer, df_test, "sft-test", args.n_eval, logger)

        # ── Évaluation DPO ────────────────────────────────────────────────────
        logger.info("Chargement du modèle DPO (base + SFT merged + DPO LoRA)...")
        dpo_model, dpo_tokenizer = load_sft_merged_for_dpo_eval(
            MODEL_NAME, SFT_CHECKPOINT, DPO_CHECKPOINT
        )

        logger.info("Évaluation DPO sur val...")
        dpo_val = evaluate_split(dpo_model, dpo_tokenizer, df_val, "dpo-val", args.n_eval, logger)
        logger.info("Évaluation DPO sur test...")
        dpo_test = evaluate_split(dpo_model, dpo_tokenizer, df_test, "dpo-test", args.n_eval, logger)

        # ── Comparaisons qualitatives ─────────────────────────────────────────
        logger.info("Génération de %d comparaisons qualitatives SFT vs DPO...", args.n_comparisons)
        comparisons = compare_responses_on_dpo_val(
            sft_model, dpo_model, sft_tokenizer, DPO_VAL_PATH, n=args.n_comparisons
        )

        # ── Rapport ───────────────────────────────────────────────────────────
        report = generate_dpo_eval_report(sft_val, dpo_val, sft_test, dpo_test, comparisons)
        DPO_CHECKPOINT.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(report, encoding="utf-8")
        logger.info("Rapport sauvegardé dans %s", REPORT_PATH)

        mlflow.log_metrics({
            "sft_val_accuracy": sft_val["accuracy"],
            "sft_val_f1_macro": sft_val["f1_macro"],
            "dpo_val_accuracy": dpo_val["accuracy"],
            "dpo_val_f1_macro": dpo_val["f1_macro"],
            "dpo_val_format_compliance": dpo_val["format_compliance"],
            "sft_test_accuracy": sft_test["accuracy"],
            "dpo_test_accuracy": dpo_test["accuracy"],
            "accuracy_delta": dpo_val["accuracy"] - sft_val["accuracy"],
        })
        mlflow.log_artifact(str(REPORT_PATH))

    logger.info("=== Évaluation DPO terminée. ===")


if __name__ == "__main__":
    main()
