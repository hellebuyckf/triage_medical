"""Script 02 — Build the unified SFT dataset (~5,000 instruction/response pairs)."""

import argparse
import sys
from functools import partial
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from utils import (
    SFT_COLUMNS,
    format_triage_response,
    get_logger,
    infer_urgency,
    is_valid_sft_row,
    load_datasets_config,
    md5_hash,
)

PROJECT_ROOT = _SCRIPTS_DIR.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "datasets.yaml"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sft_raw"

# Number of CPU cores used by .map() / .filter().
# Increase on multi-core machines for large datasets.
NUM_PROC = 4

# Sentinel dict returned by mappers for rows that fail validation.
# Arrow requires a consistent schema, so we can't return None from .map().
# Note: urgency_level / confidence / response are NOT computed here — they are
# added in a dedicated batched step after filtering (see _infer_and_format_batch).
_INVALID_ROW: dict = {
    "_keep": False,
    "instruction": "",
    "raw_response": "",
    "source": "",
    "language": "",
}


# ── Transform functions ───────────────────────────────────────────────────────


def transform_frenchmedmcqa(row: dict) -> dict | None:
    """FrenchMedMCQA → normalized (instruction, raw_response) pair.

    Columns: id, question, answer_a..answer_e, correct_answers (int64 index 0-4).

    Args:
        row: Raw dataset row.

    Returns:
        Normalized dict with keys instruction/raw_response/source/language,
        or ``None`` if the row is invalid.
    """
    question = row.get("question", "")
    correct_idx = row.get("correct_answers")

    if not question or correct_idx is None:
        return None

    letters = ["a", "b", "c", "d", "e"]
    if isinstance(correct_idx, int) and 0 <= correct_idx <= 4:
        letter = letters[correct_idx]
        correct_text = row.get(f"answer_{letter}", "")
    else:
        return None

    if not correct_text:
        return None

    return {
        "instruction": "Question médicale : " + question,
        "raw_response": "Réponse correcte : " + correct_text + ".",
        "source": "frenchmedmcqa",
        "language": "fr",
    }


def transform_medquad(row: dict) -> dict | None:
    """MedQuAD → normalized (instruction, raw_response) pair.

    Columns: qtype, Question, Answer (uppercase column names).

    Args:
        row: Raw dataset row.

    Returns:
        Normalized dict with keys instruction/raw_response/source/language,
        or ``None`` if the row is invalid.
    """
    instruction = row.get("Question", "")
    raw_response = row.get("Answer", "")

    if not instruction or not raw_response:
        return None

    return {
        "instruction": instruction,
        "raw_response": raw_response,
        "source": "medquad",
        "language": "en",
    }


def transform_mediql_mcqu(row: dict) -> dict | None:
    """MediQAl config mcqu → normalized (instruction, raw_response) pair.

    Columns: id, clinical_case, question, answer_a..answer_e,
    correct_answers (str "A".."E"), task, medical_subject, question_type.

    Args:
        row: Raw dataset row.

    Returns:
        Normalized dict with keys instruction/raw_response/source/language,
        or ``None`` if the row is invalid.
    """
    question = row.get("question", "")
    if not question:
        return None

    correct_answers = row.get("correct_answers", "")
    if not correct_answers:
        return None

    letter = correct_answers.strip().lower()
    response_text = row.get(f"answer_{letter}", "")
    if not response_text:
        return None

    clinical_case = row.get("clinical_case", "")
    if clinical_case:
        instruction = "Question médicale (examen) : " + clinical_case + "\n" + question
    else:
        instruction = "Question médicale (examen) : " + question

    return {
        "instruction": instruction,
        "raw_response": "Réponse correcte : " + response_text + ".",
        "source": "mediql_mcqu",
        "language": "fr",
    }


def transform_mediql_oeq(row: dict) -> dict | None:
    """MediQAl config oeq → normalized (instruction, raw_response) pair.

    Open-ended questions: question + direct answer.

    Args:
        row: Raw dataset row.

    Returns:
        Normalized dict with keys instruction/raw_response/source/language,
        or ``None`` if the row is invalid.
    """
    question = row.get("question", "")
    answer = row.get("answer", "") or row.get("correct_answer", "")

    if not question or not answer:
        return None

    return {
        "instruction": "Question médicale (examen) : " + question,
        "raw_response": str(answer),
        "source": "mediql_oeq",
        "language": "fr",
    }


# ── .map() / .filter() helpers ────────────────────────────────────────────────
# Module-level functions are picklable, which is required for num_proc > 1.


def _apply_transform(row: dict, transform_fn) -> dict:
    """Apply a transform function and return a schema-consistent output dict.

    Invalid rows (transform returns None) are marked with ``_keep=False`` so
    they can be removed in a subsequent ``.filter()`` call.

    Args:
        row: Raw dataset row passed by ``.map()``.
        transform_fn: Source-specific transform function.

    Returns:
        Output dict with SFT columns plus a boolean ``_keep`` flag.
    """
    result = transform_fn(row)
    if result is None:
        return _INVALID_ROW
    return {"_keep": True, **result}


def _is_valid(row: dict) -> bool:
    """Filter predicate: keep only rows flagged valid after content checks.

    Validation runs on ``raw_response`` (the extracted content before urgency
    formatting), which is the right signal for quality — the formatted response
    would always pass the length check since formatting adds text.

    Args:
        row: Mapped row containing ``_keep``, ``instruction``, and ``raw_response``.

    Returns:
        ``True`` if the row should be retained.
    """
    return row["_keep"] and is_valid_sft_row(row["instruction"], row["raw_response"])


def _infer_and_format_batch(batch: dict[str, list]) -> dict[str, list]:
    """Infer urgency and format responses for a batch of normalized rows.

    This is the single location where ``infer_urgency`` and
    ``format_triage_response`` are called. Keeping urgency inference in one
    batched step makes it trivial to swap the regex classifier for an ML model
    (e.g., a fine-tuned classifier or embedding model) without touching the
    extraction logic in the transform functions.

    Args:
        batch: Dict of lists with keys ``instruction`` and ``raw_response``,
            as provided by ``Dataset.map(batched=True)``.

    Returns:
        Dict of lists with new keys ``urgency_level``, ``confidence``,
        and ``response``. ``raw_response`` is kept so the caller can drop it
        via ``remove_columns``.
    """
    urgency_levels: list[str] = []
    confidences: list[float] = []
    responses: list[str] = []

    for instruction, raw_response in zip(batch["instruction"], batch["raw_response"], strict=True):
        level, conf = infer_urgency(instruction + " " + raw_response)
        urgency_levels.append(level)
        confidences.append(conf)
        responses.append(format_triage_response(level, raw_response))

    return {
        "urgency_level": urgency_levels,
        "confidence": confidences,
        "response": responses,
    }


# ── Pipeline ──────────────────────────────────────────────────────────────────

SOURCES = {
    "frenchmedmcqa": transform_frenchmedmcqa,
    "medquad": transform_medquad,
    "mediql_mcqu": transform_mediql_mcqu,
    "mediql_oeq": transform_mediql_oeq,
}


def load_and_transform(name: str, ds_config: dict, transform_fn, logger) -> Dataset | None:
    """Load a raw dataset via HuggingFace cache and apply the source-specific transform.

    Uses ``load_dataset(cache_dir=...)`` so the HF cache is the single source
    of truth — no second copy is created on disk. Uses ``Dataset.map()`` for
    Arrow-native batch processing and ``Dataset.filter()`` to discard invalid
    rows, avoiding Python-level loops and intermediate list allocations.

    Args:
        name: Dataset name, used for logging only.
        ds_config: Config dict with keys ``hf_id``, ``hf_config``, and ``cache_dir``.
        transform_fn: Source-specific transform function returning a SFT dict or None.
        logger: Logger instance.

    Returns:
        A ``Dataset`` with SFT columns, or ``None`` if loading fails.
    """
    logger.info(f"[{name}] Loading from HF cache ({ds_config['cache_dir']})...")
    kwargs: dict = {
        "path": ds_config["hf_id"],
        "cache_dir": str(ds_config["cache_dir"]),
    }
    if ds_config["hf_config"]:
        kwargs["name"] = ds_config["hf_config"]

    raw = load_dataset(**kwargs)
    ds: DatasetDict = DatasetDict({"train": raw}) if isinstance(raw, Dataset) else raw

    splits = []
    for split_name, split_ds in ds.items():
        logger.info(f"[{name}] Mapping split '{split_name}' ({len(split_ds)} examples)...")

        mapped = split_ds.map(
            partial(_apply_transform, transform_fn=transform_fn),
            remove_columns=split_ds.column_names,
            desc=f"{name}/{split_name}",
            num_proc=NUM_PROC,
        )
        valid = mapped.filter(
            _is_valid, desc=f"{name}/{split_name} filter", num_proc=NUM_PROC
        ).remove_columns(["_keep"])
        logger.info(f"[{name}/{split_name}] {len(valid)} valid examples after transform+filter.")
        splits.append(valid)

    if not splits:
        return None

    return concatenate_datasets(splits) if len(splits) > 1 else splits[0]


def deduplicate(ds: Dataset, logger) -> Dataset:
    """Remove duplicate rows based on MD5 hash of the (instruction, response) pair.

    Hashing the full pair rather than instruction alone preserves legitimate
    cases where the same question has multiple valid responses (e.g., different
    urgency levels from different source datasets).

    Deduplication is inherently sequential (stateful set lookup), so it is
    performed in pandas rather than via ``.map()``.

    Args:
        ds: Input dataset with ``instruction`` and ``response`` columns.
        logger: Logger instance.

    Returns:
        Deduplicated ``Dataset``.
    """
    df = pd.DataFrame(ds.to_pandas())
    df["_hash"] = (df["instruction"] + df["response"]).apply(md5_hash)
    before = len(df)
    df = df.drop_duplicates(subset=["_hash"]).drop(columns=["_hash"]).reset_index(drop=True)
    logger.info(f"Deduplication: {before - len(df)} duplicates removed, {len(df)} remaining.")
    return Dataset.from_pandas(df, preserve_index=False)


def balance_classes(
    df: pd.DataFrame, target_total: int = 5000, seed: int = 42, logger=None
) -> pd.DataFrame:
    """Undersample to balance urgency_level classes.

    Targets ``target_total // 3`` examples per class. If a class has fewer
    examples than the target, all of its examples are kept.

    Args:
        df: DataFrame with an ``urgency_level`` column.
        target_total: Desired total number of examples across all classes.
        seed: Random seed for reproducibility.
        logger: Optional logger instance.

    Returns:
        Balanced and shuffled DataFrame.
    """
    target_per_class = target_total // 3
    counts = df["urgency_level"].value_counts()
    if logger:
        logger.info(f"Distribution before balancing: {counts.to_dict()}")

    balanced_parts = []
    for level in ["max", "moderate", "deferred"]:
        subset = df[df["urgency_level"] == level]
        n = min(len(subset), target_per_class)
        balanced_parts.append(subset.sample(n=n, random_state=seed))

    result = (
        pd.concat(balanced_parts, ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    if logger:
        logger.info(
            f"Distribution after balancing: {result['urgency_level'].value_counts().to_dict()}"
        )
    return result


def build_sft(datasets_config: dict, logger) -> pd.DataFrame:
    """Full SFT dataset construction pipeline.

    Args:
        datasets_config: Full datasets config loaded from YAML (all usages).
            Only entries with ``usage == "sft"`` are processed.
        logger: Logger instance.

    Returns:
        Balanced SFT DataFrame ready to be saved.

    Raises:
        RuntimeError: If no source dataset could be loaded.
    """
    all_datasets = []
    for name, transform_fn in SOURCES.items():
        if name not in datasets_config:
            logger.warning(f"[{name}] Not found in config, skipping.")
            continue
        ds = load_and_transform(name, datasets_config[name], transform_fn, logger)
        if ds is not None:
            all_datasets.append(ds)

    if not all_datasets:
        raise RuntimeError("No source datasets could be loaded.")

    combined = concatenate_datasets(all_datasets)
    logger.info(f"Total raw: {len(combined)} examples.")

    # Urgency inference — single batched pass over all sources combined.
    # Swap _infer_and_format_batch for an ML model here without touching anything else.
    logger.info("Inferring urgency labels (batched)...")
    combined = combined.map(
        _infer_and_format_batch,
        batched=True,
        batch_size=1000,
        remove_columns=["raw_response"],
        desc="urgency inference",
        num_proc=NUM_PROC,
    )

    combined = deduplicate(combined, logger)
    df = pd.DataFrame(pd.DataFrame(combined.to_pandas())[SFT_COLUMNS])

    logger.info(f"Source distribution: {df['source'].value_counts().to_dict()}")
    logger.info(f"Language distribution: {df['language'].value_counts().to_dict()}")

    return balance_classes(df, target_total=6500, seed=42, logger=logger)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SFT dataset")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to datasets YAML config (default: {DEFAULT_CONFIG})",
    )
    args = parser.parse_args()

    logger = get_logger("02_build_sft", verbose=args.verbose)
    datasets_config = load_datasets_config(args.config, PROJECT_ROOT)

    if OUTPUT_PATH.exists():
        logger.info(f"SFT dataset already built at {OUTPUT_PATH}, skipping.")
        df = pd.DataFrame(Dataset.load_from_disk(str(OUTPUT_PATH)).to_pandas())
        logger.info(
            f"  {len(df)} examples, distribution: {df['urgency_level'].value_counts().to_dict()}"
        )
        return

    df = build_sft(datasets_config, logger)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_pandas(df).save_to_disk(str(OUTPUT_PATH))
    logger.info(f"SFT dataset: {len(df)} examples saved to {OUTPUT_PATH}.")
    logger.info(f"  Urgency: {df['urgency_level'].value_counts().to_string()}")
    logger.info(f"  Source:  {df['source'].value_counts().to_string()}")
    logger.info(f"  Language: {df['language'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
