"""Script 10 — Préparation du tokenizer et formatage prompt/completion pour SFT."""

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from utils import format_chat_prompt, get_logger

PROJECT_ROOT = _SCRIPTS_DIR.parent

# ── Constantes ────────────────────────────────────────────────────────────────

MODEL_NAME = "unsloth/Qwen3-1.7B-Base"
MAX_SEQ_LENGTH = 1024

SFT_FINAL_DIR = PROJECT_ROOT / "data" / "final" / "sft"
TOKENIZED_DIR = PROJECT_ROOT / "data" / "processed" / "sft_tokenized"

CHAT_MARKER = "<|im_start|>assistant\n"


# ── Fonctions ─────────────────────────────────────────────────────────────────


def load_tokenizer(model_name: str) -> PreTrainedTokenizerFast:
    """Charge le tokenizer depuis HuggingFace Hub.

    Configure padding_side="right" (requis pour SFT causal).
    Assigne pad_token = eos_token si absent.

    Args:
        model_name: Identifiant du modèle sur HuggingFace Hub.

    Returns:
        Tokenizer configuré.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def format_to_prompt_completion(df: pd.DataFrame) -> Dataset:
    """Convertit un DataFrame SFT en Dataset HF au format prompt/completion.

    Le format prompt/completion est natif à TRL 0.29 SFTTrainer et active
    automatiquement le masquage de loss sur la partie prompt.

    Args:
        df: DataFrame avec colonnes instruction, response, urgency_level.

    Returns:
        HuggingFace Dataset avec colonnes prompt, completion, urgency_level.
    """
    records: list[dict[str, str]] = []
    for _, row in df.iterrows():
        full_text = format_chat_prompt(str(row["instruction"]), str(row["response"]))
        idx = full_text.index(CHAT_MARKER) + len(CHAT_MARKER)
        records.append(
            {
                "prompt": full_text[:idx],
                "completion": full_text[idx:],
                "urgency_level": str(row["urgency_level"]),
            }
        )
    return Dataset.from_list(records)


def analyze_lengths(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerFast,
    max_seq_length: int = MAX_SEQ_LENGTH,
) -> dict[str, int | float]:
    """Analyse la distribution des longueurs de tokens sur le dataset.

    Tokenise chaque exemple complet (système + instruction + réponse) sans
    troncature pour mesurer les longueurs réelles.

    Args:
        df: DataFrame SFT avec colonnes instruction et response.
        tokenizer: Tokenizer chargé.
        max_seq_length: Seuil pour compter les exemples tronqués.

    Returns:
        Dictionnaire avec p50, p75, p90, p95, p99, max, mean.
    """
    lengths: list[int] = []
    for _, row in df.iterrows():
        text = format_chat_prompt(str(row["instruction"]), str(row["response"]))
        tokens = tokenizer(text, truncation=False)["input_ids"]  # type: ignore[call-overload]
        lengths.append(len(tokens))  # type: ignore[reportArgumentType]

    arr = np.array(lengths)
    stats = {
        "p50": int(np.percentile(arr, 50)),
        "p75": int(np.percentile(arr, 75)),
        "p90": int(np.percentile(arr, 90)),
        "p95": int(np.percentile(arr, 95)),
        "p99": int(np.percentile(arr, 99)),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
    }

    n_truncated = int((arr > max_seq_length).sum())
    stats["n_truncated"] = n_truncated
    stats["pct_truncated"] = round(n_truncated / len(arr) * 100, 2)

    return stats


def main() -> None:
    """Pipeline de préparation : analyse des longueurs + formatage prompt/completion.

    Idempotent : skip si TOKENIZED_DIR/train existe déjà.
    """
    parser = argparse.ArgumentParser(description="Préparation tokenizer + formatage SFT")
    parser.add_argument("--verbose", action="store_true", help="Logging DEBUG")
    args = parser.parse_args()

    logger = get_logger("10_prepare_tokenizer", verbose=args.verbose)

    # Idempotence
    if (TOKENIZED_DIR / "train").exists():
        logger.info("Datasets tokenisés déjà présents dans %s — skip.", TOKENIZED_DIR)
        return

    # Vérification du dataset source
    if not SFT_FINAL_DIR.exists():
        logger.error("Dataset manquant : %s. Lancer le pipeline S1 d'abord.", SFT_FINAL_DIR)
        sys.exit(1)

    # Chargement du tokenizer
    logger.info("Chargement du tokenizer depuis %s...", MODEL_NAME)
    tokenizer = load_tokenizer(MODEL_NAME)
    logger.info(
        "Tokenizer chargé. Vocab size : %d, pad_token : '%s'",
        tokenizer.vocab_size,
        tokenizer.pad_token,
    )

    # Chargement des splits
    logger.info("Chargement des splits depuis %s...", SFT_FINAL_DIR)
    sft = DatasetDict(load_from_disk(str(SFT_FINAL_DIR)))  # type: ignore[arg-type]
    df_train = pd.DataFrame(sft["train"].to_pandas())
    df_val = pd.DataFrame(sft["val"].to_pandas())
    df_test = pd.DataFrame(sft["test"].to_pandas())
    logger.info("  train: %d | val: %d | test: %d", len(df_train), len(df_val), len(df_test))

    # Analyse des longueurs sur le train set
    logger.info("Analyse des longueurs de tokens (train set)...")
    stats = analyze_lengths(df_train, tokenizer, MAX_SEQ_LENGTH)
    logger.info("Distribution des longueurs :")
    for key in ["p50", "p75", "p90", "p95", "p99", "max", "mean"]:
        logger.info("  %s : %s", key, stats[key])

    if stats["n_truncated"] > 0:
        logger.warning(
            "%d exemples (%.1f%%) dépassent MAX_SEQ_LENGTH=%d et seront tronqués.",
            stats["n_truncated"],
            stats["pct_truncated"],
            MAX_SEQ_LENGTH,
        )
    else:
        logger.info("Aucun exemple ne dépasse MAX_SEQ_LENGTH=%d.", MAX_SEQ_LENGTH)

    # Recommandation de MAX_SEQ_LENGTH
    if stats["p95"] <= 512:
        logger.info("Recommandation : MAX_SEQ_LENGTH=512 suffirait (p95=%d).", stats["p95"])
    elif stats["p95"] <= 1024:
        logger.info("Recommandation : MAX_SEQ_LENGTH=1024 est adapté (p95=%d).", stats["p95"])
    else:
        logger.info("Recommandation : MAX_SEQ_LENGTH=2048 nécessaire (p95=%d).", stats["p95"])

    # Formatage prompt/completion
    logger.info("Formatage en prompt/completion...")
    splits = {
        "train": format_to_prompt_completion(df_train),
        "val": format_to_prompt_completion(df_val),
        "test": format_to_prompt_completion(df_test),
    }

    # Sauvegarde
    TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)
    for name, ds in splits.items():
        out_path = TOKENIZED_DIR / name
        ds.save_to_disk(str(out_path))
        logger.info("  %s : %d exemples sauvegardés dans %s", name, len(ds), out_path)

    logger.info("=== Préparation terminée. ===")


if __name__ == "__main__":
    main()
