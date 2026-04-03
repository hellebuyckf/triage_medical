"""Script 21 — Comparaison SFT vs DPO : accuracy, F1 et qualité clinique.

Évalue les deux checkpoints sur sft_val et sft_test, puis génère un rapport
de comparaison avant/après alignement avec exemples qualitatifs.
"""

import argparse
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

if torch.cuda.is_available(): torch.backends.cuda.preferred_blas_library("cublaslt")

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import mlflow
import pandas as pd
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
from utils import (
    SYSTEM_PROMPT,
    check_demo_env,
    extract_urgency_from_response,
    get_logger,
)

PROJECT_ROOT = _SCRIPTS_DIR.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

# ── Constantes ────────────────────────────────────────────────────────────────

MODEL_NAME = os.getenv("MODEL_NAME", "unsloth/Qwen3-1.7B")
SFT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "sft"
DPO_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "dpo"
REPORTS_DIR = PROJECT_ROOT / "reports" / "dpo"

SFT_FINAL_DIR = PROJECT_ROOT / "data" / "final" / "sft"
DPO_FINAL_DIR = PROJECT_ROOT / "data" / "final" / "dpo"

MAX_SEQ_LENGTH = 1024
MAX_NEW_TOKENS = 512
DO_SAMPLE = False
N_EXAMPLES = 5  # exemples de comparaison qualitative SFT vs DPO
BATCH_SIZE_EVAL = 8  # exemples par batch GPU — réduire à 4 ou 2 si CUDA OOM
SEED = 42

URGENCY_LABELS = ["max", "moderate", "deferred"]

# Seuils d'acceptation cliniques — calibrés POC (données de qualité moyenne, modèle 1.7B).
# Voir specs/EVAL-Metrics-Cliniques.md pour la justification complète.
CLINICAL_THRESHOLDS: dict[str, float] = {
    "recall_max": 0.75,  # Rappel classe "max" : rater une urgence critique est inacceptable
    "f2_macro": 0.60,  # F2 (β=2) pénalise les faux négatifs
    "format_compliance": 0.70,  # Réponses au format parseable par le workflow clinique
    "accuracy": 0.60,  # Accuracy globale minimale
}

MLFLOW_EXPERIMENT = "dpo-qwen3-1.7b-triage"
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}",
)

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
    # Qwen3-Base n'a pas de chat_template — on injecte le template ChatML standard
    # pour que apply_chat_template fonctionne dans generate_responses_batch.
    if not tokenizer.chat_template:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
        )

    # 1. Charger le modèle de base
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # 2. Appliquer les poids SFT et les fusionner
    model = PeftModel.from_pretrained(base, str(sft_checkpoint))
    model = model.merge_and_unload()  # type: ignore[reportCallIssue]

    # 3. Appliquer le LoRA DPO
    model = PeftModel.from_pretrained(model, str(dpo_checkpoint))
    model.eval()
    model.generation_config.max_length = None  # type: ignore[reportAttributeAccessIssue]
    return model, tokenizer  # type: ignore[reportReturnType]


def generate_responses_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    instructions: list[str],
    max_new_tokens: int = MAX_NEW_TOKENS,
    batch_size: int = BATCH_SIZE_EVAL,
) -> tuple[list[str], int]:
    """Génère des réponses pour une liste d'instructions par batches GPU.

    Envoie ``batch_size`` prompts en parallèle à ``model.generate()``, saturant
    le GPU au lieu d'une passe par exemple. Le left-padding (défini dans
    ``load_sft_merged_for_dpo_eval``) aligne toutes les séquences d'un batch.

    Compatible avec ``model.disable_adapter()`` : le contexte d'appel détermine
    si le LoRA DPO est actif (DPO) ou désactivé (SFT fusionné).

    Args:
        model: Modèle (base + SFT merged + DPO LoRA) en mode éval.
        tokenizer: Tokenizer avec ``padding_side="left"``.
        instructions: Liste des instructions médicales à évaluer.
        max_new_tokens: Tokens maximum à générer par exemple.
        batch_size: Exemples par batch GPU.

    Returns:
        Tuple (réponses, n_oom) où ``réponses[i]`` correspond à ``instructions[i]``
        et ``n_oom`` compte les exemples sautés pour CUDA OOM.
    """
    im_end_id: int = tokenizer.convert_tokens_to_ids("<|im_end|>")  # type: ignore[assignment]
    responses: list[str] = [""] * len(instructions)
    n_oom = 0

    for batch_start in tqdm(range(0, len(instructions), batch_size), desc="Generating"):
        batch = instructions[batch_start : batch_start + batch_size]

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
                    repetition_penalty=1.1,
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
            text = str(tokenizer.decode(gen_ids, skip_special_tokens=True))
            responses[batch_start + i] = _strip_artifacts(text)

    return responses, n_oom


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
    batch_size: int = BATCH_SIZE_EVAL,
    logger=None,
) -> dict:
    """Évalue l'accuracy de classification d'urgence sur un split via batch generation.

    Appelle ``generate_responses_batch`` pour saturer le GPU. Compatible avec
    ``model.disable_adapter()`` : le contexte d'appel détermine le mode SFT ou DPO.

    Args:
        model: Modèle à évaluer (adaptateur actif ou désactivé selon le contexte).
        tokenizer: Tokenizer avec ``padding_side="left"``.
        df: DataFrame SFT avec colonnes [instruction, urgency_level].
        split_name: Nom du split pour les logs et MLflow.
        n_eval: Nombre d'exemples à évaluer (None = tous).
        batch_size: Exemples par batch GPU.
        logger: Logger optionnel.

    Returns:
        Dict {accuracy, f1_macro, format_compliance, n_unparseable, n_evaluated, predictions}.
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
            logger.warning("[{}] {} exemples sautés pour CUDA OOM.", split_name, n_oom)

        for i, (_, row) in enumerate(df.iterrows()):
            generated = generated_responses[i]
            predicted = extract_urgency_from_response(generated)
            predictions.append(
                {
                    "instruction": row["instruction"],
                    "reference_urgency": row["urgency_level"],
                    "predicted_urgency": predicted,
                    "generated_response": generated,
                    "format_ok": check_format_compliance(generated),
                }
            )

        duration_s = round(time.monotonic() - t0, 1)

        valid = [
            (p["reference_urgency"], p["predicted_urgency"])
            for p in predictions
            if p["predicted_urgency"]
        ]
        n_unparseable = len(predictions) - len(valid)

        if valid:
            y_true = [t for t, _ in valid]
            y_pred = [p for _, p in valid]
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro", labels=URGENCY_LABELS, zero_division=0)  # type: ignore[reportArgumentType]
            recall_macro = recall_score(
                y_true,
                y_pred,
                average="macro",
                labels=URGENCY_LABELS,
                zero_division=0,  # type: ignore[reportArgumentType]
            )
            # Recall par classe — index 0 = "max" (urgence critique, métrique clinique clé)
            recall_per_class = recall_score(
                y_true,
                y_pred,
                average=None,
                labels=URGENCY_LABELS,
                zero_division=0,  # type: ignore[reportArgumentType]
            )
            recall_max = float(recall_per_class[0])
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
            accuracy, f1, recall_macro, recall_max, f2_macro, cm = 0.0, 0.0, 0.0, 0.0, 0.0, None

        format_compliance = sum(p["format_ok"] for p in predictions) / len(predictions)
        avg_len = sum(len(p["generated_response"].split()) for p in predictions) / len(predictions)

        metrics = {
            "accuracy": round(accuracy, 4),
            "f1_macro": round(f1, 4),
            "recall_macro": round(recall_macro, 4),
            "recall_max": round(recall_max, 4),
            "f2_macro": round(f2_macro, 4),
            "format_compliance": round(format_compliance, 4),
            "response_length_mean": round(avg_len, 1),
            "n_unparseable": n_unparseable,
            "n_evaluated": len(predictions),
            "confusion_matrix": cm,
            "predictions": predictions,
        }

        if logger:
            logger.info(
                "[{}] accuracy={:.2f}% | f1={:.4f} | recall_macro={:.4f} | f2_macro={:.4f} | format={:.1f}% | unparseable={} | oom={} | {:.0f}s",
                split_name,
                accuracy * 100,
                f1,
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


def compare_responses_on_dpo_val(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    dpo_val_path: Path,
    n: int = N_EXAMPLES,
    batch_size: int = BATCH_SIZE_EVAL,
    seed: int = SEED,
) -> list[dict]:
    """Génère des réponses SFT et DPO sur les prompts du val DPO via batch generation.

    Deux passes batch sur le même modèle :
    1. SFT : ``model.disable_adapter()`` → poids SFT fusionnés, batch entier.
    2. DPO : adaptateur actif → batch entier.

    Args:
        model: Modèle unique (base + SFT merged + DPO LoRA).
        tokenizer: Tokenizer avec ``padding_side="left"``.
        dpo_val_path: Répertoire du DatasetDict DPO (contient le split val).
        n: Nombre de comparaisons à générer.
        batch_size: Exemples par batch GPU.
        seed: Graine pour le sous-échantillonnage.

    Returns:
        Liste de dicts {prompt, sft_response, dpo_response, chosen_reference}.
    """
    dpo = DatasetDict(load_from_disk(str(dpo_val_path)))  # type: ignore[arg-type]
    df = pd.DataFrame(dpo["val"].to_pandas()).sample(n=min(n, 100), random_state=seed).head(n)
    prompts = [str(row["prompt"]) for _, row in df.iterrows()]
    chosen_refs = [str(row.get("chosen", "")) for _, row in df.iterrows()]

    with model.disable_adapter():  # type: ignore[reportAttributeAccessIssue]
        sft_responses, _ = generate_responses_batch(
            model, tokenizer, prompts, batch_size=batch_size
        )
    dpo_responses, _ = generate_responses_batch(model, tokenizer, prompts, batch_size=batch_size)

    return [
        {
            "prompt": prompt,
            "sft_response": sft_resp,
            "dpo_response": dpo_resp,
            "chosen_reference": chosen_ref,
        }
        for prompt, sft_resp, dpo_resp, chosen_ref in zip(
            prompts, sft_responses, dpo_responses, chosen_refs, strict=True
        )
    ]


def check_clinical_thresholds(
    metrics: dict,
    thresholds: dict[str, float] = CLINICAL_THRESHOLDS,
) -> list[dict]:
    """Vérifie les métriques cliniques par rapport aux seuils d'acceptation POC.

    Args:
        metrics: Dict de métriques issu de evaluate_split.
        thresholds: Dict {nom_metrique: seuil_minimal}. Défaut : CLINICAL_THRESHOLDS.

    Returns:
        Liste de dicts {criterion, value, threshold, passed}.
    """
    return [
        {
            "criterion": criterion,
            "value": float(metrics.get(criterion, 0.0)),
            "threshold": threshold,
            "passed": float(metrics.get(criterion, 0.0)) >= threshold,
        }
        for criterion, threshold in thresholds.items()
    ]


def _format_clinical_comparison_table(
    sft_metrics: dict,
    dpo_metrics: dict,
    thresholds: dict[str, float] = CLINICAL_THRESHOLDS,
) -> str:
    """Génère le tableau Markdown de comparaison clinique SFT vs DPO avec PASS/FAIL.

    Args:
        sft_metrics: Métriques SFT (test set).
        dpo_metrics: Métriques DPO (test set).
        thresholds: Dict {nom_metrique: seuil_minimal}.

    Returns:
        Tableau Markdown formaté.
    """
    _labels = {
        "recall_max": "Recall URGENCE MAX",
        "f2_macro": "F2 Macro (β=2)",
        "format_compliance": "Format Compliance",
        "accuracy": "Accuracy globale",
    }
    _pct_keys = {"recall_max", "format_compliance", "accuracy"}
    lines = [
        "| Critère | SFT | DPO | Seuil (POC) | Statut SFT | Statut DPO |",
        "|---|---|---|---|---|---|",
    ]
    for criterion, threshold in thresholds.items():
        name = _labels.get(criterion, criterion)
        sv = float(sft_metrics.get(criterion, 0.0))
        dv = float(dpo_metrics.get(criterion, 0.0))
        if criterion in _pct_keys:
            sv_str, dv_str, thr_str = f"{sv:.1%}", f"{dv:.1%}", f"≥ {threshold:.0%}"
        else:
            sv_str, dv_str, thr_str = f"{sv:.4f}", f"{dv:.4f}", f"≥ {threshold:.2f}"
        sft_status = "✅ PASS" if sv >= threshold else "❌ FAIL"
        dpo_status = "✅ PASS" if dv >= threshold else "❌ FAIL"
        lines.append(f"| {name} | {sv_str} | {dv_str} | {thr_str} | {sft_status} | {dpo_status} |")
    return "\n".join(lines)


def generate_dpo_eval_report(
    sft_val_metrics: dict | None,
    dpo_val_metrics: dict | None,
    sft_test_metrics: dict,
    dpo_test_metrics: dict,
    comparisons: list[dict],
) -> str:
    """Génère un rapport Markdown de comparaison SFT vs DPO.

    Args:
        sft_val_metrics: Métriques SFT sur val set, ou None si --eval-val non activé.
        dpo_val_metrics: Métriques DPO sur val set, ou None si --eval-val non activé.
        sft_test_metrics: Métriques SFT sur test set (toujours disponible).
        dpo_test_metrics: Métriques DPO sur test set (toujours disponible).
        comparisons: Comparaisons qualitatives SFT vs DPO.

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
                    "recall_max",
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
            "recall_max": f"{m['recall_max']:.2%}",
            "f2_macro": f"{m['f2_macro']:.4f}",
            "format_compliance": f"{m['format_compliance']:.1%}",
            "response_length_mean": f"{m['response_length_mean']:.0f}",
            "n_unparseable": f"{m['n_unparseable']}/{m['n_evaluated']}",
        }

    def _delta(a: dict | None, b: dict | None, key: str, pct: bool = False) -> str:
        """Calcule le delta entre deux métriques. Retourne 'N/A' si l'une est None."""
        if a is None or b is None:
            return "N/A"
        diff = b[key] - a[key]
        sign = "+" if diff >= 0 else ""
        return f"{sign}{diff * 100:.1f}%" if pct else f"{sign}{diff:.4f}"

    sv = _fmt(sft_val_metrics)
    dv = _fmt(dpo_val_metrics)
    st = _fmt(sft_test_metrics)
    dt = _fmt(dpo_test_metrics)

    has_val = sft_val_metrics is not None and dpo_val_metrics is not None

    # Header
    report = (
        "# Rapport d'Évaluation DPO\n"
        "## project14 — Agent de Triage Médical (Qwen3-1.7B + SFT + DPO)\n\n"
        f"**Date** : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"**Modèle de base** : {MODEL_NAME}\n"
        f"**Checkpoint SFT** : {SFT_CHECKPOINT}\n"
        f"**Checkpoint DPO** : {DPO_CHECKPOINT}\n\n"
        "---\n\n"
        "## 1. Métriques comparées — Val Set\n\n"
    )

    # Val set section — table only when both sets of metrics are available
    if not has_val:
        report += "*Non évalué (utiliser --eval-val pour générer).*\n"
    else:
        report += (
            "| Métrique | SFT | DPO | Delta |\n"
            "|---|---|---|---|\n"
            "| Accuracy | "
            + sv["accuracy"]
            + " | "
            + dv["accuracy"]
            + " | "
            + _delta(sft_val_metrics, dpo_val_metrics, "accuracy", pct=True)
            + " |\n"
            "| F1 Macro | "
            + sv["f1_macro"]
            + " | "
            + dv["f1_macro"]
            + " | "
            + _delta(sft_val_metrics, dpo_val_metrics, "f1_macro")
            + " |\n"
            "| Recall Macro | "
            + sv["recall_macro"]
            + " | "
            + dv["recall_macro"]
            + " | "
            + _delta(sft_val_metrics, dpo_val_metrics, "recall_macro")
            + " |\n"
            "| **Recall URGENCE MAX** | "
            + sv["recall_max"]
            + " | "
            + dv["recall_max"]
            + " | "
            + _delta(sft_val_metrics, dpo_val_metrics, "recall_max", pct=True)
            + " |\n"
            "| F2 Macro (\u03b2=2) | "
            + sv["f2_macro"]
            + " | "
            + dv["f2_macro"]
            + " | "
            + _delta(sft_val_metrics, dpo_val_metrics, "f2_macro")
            + " |\n"
            "| Format Compliance | "
            + sv["format_compliance"]
            + " | "
            + dv["format_compliance"]
            + " | "
            + _delta(sft_val_metrics, dpo_val_metrics, "format_compliance", pct=True)
            + " |\n"
            "| Non-parseables | " + sv["n_unparseable"] + " | " + dv["n_unparseable"] + " | — |\n"
            "| Longueur réponse (mots) | "
            + sv["response_length_mean"]
            + " | "
            + dv["response_length_mean"]
            + " | "
            + _delta(sft_val_metrics, dpo_val_metrics, "response_length_mean")
            + " |\n"
        )

    # Test set section
    report += (
        "\n## 2. Métriques comparées — Test Set\n\n"
        "| Métrique | SFT | DPO | Delta |\n"
        "|---|---|---|---|\n"
        "| Accuracy | "
        + st["accuracy"]
        + " | "
        + dt["accuracy"]
        + " | "
        + _delta(sft_test_metrics, dpo_test_metrics, "accuracy", pct=True)
        + " |\n"
        "| F1 Macro | "
        + st["f1_macro"]
        + " | "
        + dt["f1_macro"]
        + " | "
        + _delta(sft_test_metrics, dpo_test_metrics, "f1_macro")
        + " |\n"
        "| Recall Macro | "
        + st["recall_macro"]
        + " | "
        + dt["recall_macro"]
        + " | "
        + _delta(sft_test_metrics, dpo_test_metrics, "recall_macro")
        + " |\n"
        "| **Recall URGENCE MAX** | "
        + st["recall_max"]
        + " | "
        + dt["recall_max"]
        + " | "
        + _delta(sft_test_metrics, dpo_test_metrics, "recall_max", pct=True)
        + " |\n"
        "| F2 Macro (\u03b2=2) | "
        + st["f2_macro"]
        + " | "
        + dt["f2_macro"]
        + " | "
        + _delta(sft_test_metrics, dpo_test_metrics, "f2_macro")
        + " |\n"
        "| Format Compliance | "
        + st["format_compliance"]
        + " | "
        + dt["format_compliance"]
        + " | "
        + _delta(sft_test_metrics, dpo_test_metrics, "format_compliance", pct=True)
        + " |\n\n"
        "---\n\n"
        "## 3. Matrice de confusion DPO (Val Set)\n\n"
    )
    if dpo_val_metrics is None:
        report += "*Non évalué (utiliser --eval-val pour générer).*\n"
    else:
        cm_val = dpo_val_metrics.get("confusion_matrix")
        if cm_val is not None:
            report += "| | Prédit max | Prédit moderate | Prédit deferred |\n"
            report += "|---|---|---|---|\n"
            for i, label in enumerate(URGENCY_LABELS):
                row_vals = " | ".join(str(cm_val[i][j]) for j in range(len(URGENCY_LABELS)))
                report += f"| **Réel {label}** | {row_vals} |\n"
        else:
            report += "Matrice non disponible (aucune prédiction valide).\n"

    report += "\n---\n\n## 4. Matrice de confusion DPO (Test Set)\n\n"
    cm_test = dpo_test_metrics.get("confusion_matrix")
    if cm_test is not None:
        report += "| | Prédit max | Prédit moderate | Prédit deferred |\n"
        report += "|---|---|---|---|\n"
        for i, label in enumerate(URGENCY_LABELS):
            row_vals = " | ".join(str(cm_test[i][j]) for j in range(len(URGENCY_LABELS)))
            report += f"| **Réel {label}** | {row_vals} |\n"
    else:
        report += "Matrice non disponible (aucune prédiction valide).\n"

    report += "\n---\n\n## 5. Comparaisons qualitatives SFT vs DPO\n\n"
    for i, comp in enumerate(comparisons, 1):
        report += f"### Exemple {i}\n\n"
        report += f"**Prompt** : {comp['prompt'][:300]}\n\n"
        report += f"**Réponse SFT** :\n{comp['sft_response'][:600]}\n\n"
        report += f"**Réponse DPO** :\n{comp['dpo_response'][:600]}\n\n"
        if comp["chosen_reference"]:
            report += f"**Référence chosen** :\n{comp['chosen_reference'][:400]}\n\n"
        report += "---\n\n"

    # Recommandation basée sur le test set (référence honnête, toujours disponible)
    acc_dpo = dpo_test_metrics["accuracy"]
    acc_sft = sft_test_metrics["accuracy"]
    acc_delta = acc_dpo - acc_sft
    report += "## 6. Recommandation\n\n"
    if sft_val_metrics is None or dpo_val_metrics is None:
        report += "*Note : --eval-val non activé, recommandation basée sur le Test Set.*\n\n"
    if acc_delta >= -0.03 and acc_dpo >= 0.65:
        report += (
            f"**DPO validé** : accuracy DPO ({acc_dpo:.1%}) "
            f"sans régression significative (delta={acc_delta:+.1%}). "
            "Prêt pour l'export (22_export_model.py).\n"
        )
    elif acc_delta < -0.05:
        report += (
            f"**Régression détectée** : accuracy DPO ({acc_dpo:.1%}) "
            f"chute de {abs(acc_delta):.1%} vs SFT. "
            "Augmenter beta (0.1 → 0.3) pour renforcer la contrainte KL, "
            "ou réduire à 1 epoch.\n"
        )
    else:
        report += (
            f"**Résultat mitigé** : delta accuracy = {acc_delta:+.1%}. "
            "Analyse qualitative nécessaire avant export.\n"
        )

    # Critères cliniques — seuils POC (données de qualité moyenne, modèle 1.7B)
    clinical_table = _format_clinical_comparison_table(sft_test_metrics, dpo_test_metrics)
    sft_results = check_clinical_thresholds(sft_test_metrics)
    dpo_results = check_clinical_thresholds(dpo_test_metrics)
    n_pass_sft = sum(1 for r in sft_results if r["passed"])
    n_pass_dpo = sum(1 for r in dpo_results if r["passed"])
    n_total = len(sft_results)
    report += (
        "\n---\n\n## 7. Critères cliniques — Seuils d'acceptation POC\n\n"
        "> Seuils calibrés pour un POC sur données de qualité moyenne (modèle 1.7B, labels inférés).\n"
        "> Voir `specs/EVAL-Metrics-Cliniques.md` pour la justification complète.\n\n"
        + clinical_table
        + f"\n\n**SFT : {n_pass_sft}/{n_total} critères atteints"
        + (" — validé.**\n" if n_pass_sft == n_total else ".**\n")
        + f"**DPO : {n_pass_dpo}/{n_total} critères atteints"
        + (" — validé.**\n" if n_pass_dpo == n_total else ".**\n")
    )

    return report


def main() -> None:
    """Pipeline d'évaluation DPO : compare SFT vs DPO + génère le rapport."""
    parser = argparse.ArgumentParser(
        description="Évaluation comparative SFT vs DPO.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Note sur --eval-val :\n"
            "  Le val set est utilisé pendant l'entraînement pour sélectionner le\n"
            "  meilleur checkpoint (load_best_model_at_end=True). Les métriques val\n"
            "  sont donc biaisées (le modèle a été optimisé sur ce split).\n"
            "  Seul le test set fournit une estimation honnête des performances.\n"
            "  --eval-val est utile uniquement pour debugger ou vérifier la cohérence\n"
            "  avec les métriques du Trainer (~3h30 de GPU supplémentaires par modèle)."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Logging DEBUG")
    parser.add_argument(
        "--n-eval",
        type=int,
        default=None,
        help="Nombre d'exemples par split (None = tous). Utile pour le debug.",
    )
    parser.add_argument(
        "--n-comparisons",
        type=int,
        default=N_EXAMPLES,
        help="Nombre de comparaisons qualitatives SFT vs DPO.",
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
    check_demo_env()

    logger = get_logger("21_evaluate_dpo", verbose=args.verbose)

    # Vérifications
    for path in [
        SFT_CHECKPOINT / "adapter_model.safetensors",
        DPO_CHECKPOINT / "adapter_model.safetensors",
    ]:
        if not path.exists():
            logger.error("Checkpoint manquant : {}", path)
            sys.exit(1)

    for path in [SFT_FINAL_DIR, DPO_FINAL_DIR]:
        if not path.exists():
            logger.error("Dataset manquant : {}", path)
            sys.exit(1)

    # Le start_run en début de main() rattache tous les spans @mlflow.trace
    # et mlflow.start_span() au même run (évite les traces orphelines).
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    # System metrics : CPU, RAM, GPU utilization + VRAM (nécessite pynvml)
    mlflow.enable_system_metrics_logging()
    with mlflow.start_run(run_name="eval-dpo"):
        # Chargement des données
        # urgency_level est encodé en ClassLabel (entiers). On décode en strings
        # pour que y_true soit homogène avec les prédictions de extract_urgency_from_response.
        sft = DatasetDict(load_from_disk(str(SFT_FINAL_DIR)))  # type: ignore[arg-type]
        urgency_feature = sft["test"].features["urgency_level"]
        df_test = pd.DataFrame(sft["test"].to_pandas())
        df_test["urgency_level"] = df_test["urgency_level"].map(urgency_feature.int2str)
        logger.info("Test: {} exemples SFT", len(df_test))

        # ── Chargement unique du modèle ───────────────────────────────────────
        # Un seul modèle (base + SFT merged + DPO LoRA) est maintenu en VRAM.
        # Le swap SFT ↔ DPO se fait via model.disable_adapter() sans rechargement.
        logger.info("Chargement du modèle (base + SFT merged + DPO LoRA)...")
        model, tokenizer = load_sft_merged_for_dpo_eval(MODEL_NAME, SFT_CHECKPOINT, DPO_CHECKPOINT)

        # ── Évaluation SFT (adaptateur DPO désactivé) ────────────────────────
        sft_val = None
        if args.eval_val:
            logger.warning(
                "--eval-val activé : le val set a servi à sélectionner le checkpoint "
                "(load_best_model_at_end=True) — métriques biaisées, ~3h30 supplémentaires par modèle."
            )
            df_val = pd.DataFrame(sft["val"].to_pandas())
            df_val["urgency_level"] = df_val["urgency_level"].map(urgency_feature.int2str)
            with model.disable_adapter():  # type: ignore[reportAttributeAccessIssue]
                sft_val = evaluate_split(
                    model, tokenizer, df_val, "sft-val", args.n_eval, args.batch_size, logger
                )

        logger.info("Évaluation SFT sur test (adaptateur DPO désactivé)...")
        with model.disable_adapter():  # type: ignore[reportAttributeAccessIssue]
            sft_test = evaluate_split(
                model, tokenizer, df_test, "sft-test", args.n_eval, args.batch_size, logger
            )

        # ── Évaluation DPO (adaptateur actif) ────────────────────────────────
        dpo_val = None
        if args.eval_val:
            df_val = pd.DataFrame(sft["val"].to_pandas())
            df_val["urgency_level"] = df_val["urgency_level"].map(urgency_feature.int2str)
            dpo_val = evaluate_split(
                model, tokenizer, df_val, "dpo-val", args.n_eval, args.batch_size, logger
            )

        logger.info("Évaluation DPO sur test...")
        dpo_test = evaluate_split(
            model, tokenizer, df_test, "dpo-test", args.n_eval, args.batch_size, logger
        )

        # ── Comparaisons qualitatives ─────────────────────────────────────────
        logger.info("Génération de {} comparaisons qualitatives SFT vs DPO...", args.n_comparisons)
        comparisons = compare_responses_on_dpo_val(
            model, tokenizer, DPO_FINAL_DIR, n=args.n_comparisons, batch_size=args.batch_size
        )

        # ── Rapport ───────────────────────────────────────────────────────────
        report = generate_dpo_eval_report(sft_val, dpo_val, sft_test, dpo_test, comparisons)
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = REPORTS_DIR / f"eval_report_{run_timestamp}.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        logger.info("Rapport sauvegardé dans {}", report_path)

        metrics_to_log = {
            "sft_test_accuracy": sft_test["accuracy"],
            "sft_test_f1_macro": sft_test["f1_macro"],
            "sft_test_recall_macro": sft_test["recall_macro"],
            "sft_test_recall_max": sft_test["recall_max"],
            "sft_test_f2_macro": sft_test["f2_macro"],
            "dpo_test_accuracy": dpo_test["accuracy"],
            "dpo_test_f1_macro": dpo_test["f1_macro"],
            "dpo_test_recall_macro": dpo_test["recall_macro"],
            "dpo_test_recall_max": dpo_test["recall_max"],
            "dpo_test_f2_macro": dpo_test["f2_macro"],
            "dpo_test_format_compliance": dpo_test["format_compliance"],
            "accuracy_delta": dpo_test["accuracy"] - sft_test["accuracy"],
            "f1_macro_delta": dpo_test["f1_macro"] - sft_test["f1_macro"],
            "recall_macro_delta": dpo_test["recall_macro"] - sft_test["recall_macro"],
            "recall_max_delta": dpo_test["recall_max"] - sft_test["recall_max"],
            "f2_macro_delta": dpo_test["f2_macro"] - sft_test["f2_macro"],
        }
        if sft_val and dpo_val:
            metrics_to_log.update(
                {
                    "sft_val_accuracy": sft_val["accuracy"],
                    "sft_val_recall_macro": sft_val["recall_macro"],
                    "sft_val_recall_max": sft_val["recall_max"],
                    "sft_val_f2_macro": sft_val["f2_macro"],
                    "dpo_val_accuracy": dpo_val["accuracy"],
                    "dpo_val_recall_macro": dpo_val["recall_macro"],
                    "dpo_val_recall_max": dpo_val["recall_max"],
                    "dpo_val_f2_macro": dpo_val["f2_macro"],
                }
            )
        mlflow.log_metrics(metrics_to_log)
        mlflow.log_artifact(str(report_path))

    logger.info("=== Évaluation DPO terminée. ===")


if __name__ == "__main__":
    main()
