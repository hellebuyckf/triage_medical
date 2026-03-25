"""Script 03 — Build the DPO dataset (~1,000 synthetic prompt/chosen/rejected pairs).

Strategy: synthetic pairs from the SFT train split.
  - chosen  : the correct triage response (ground-truth urgency label).
  - rejected : same clinical body, reformatted with a plausible wrong urgency label.

This approach ensures DPO directly corrects the SFT model's urgency-classification
errors instead of learning stylistic preferences from misaligned academic text
(UltraMedical-Preference was abandoned — see memory/project_dpo_strategy.md).

Urgency swap table (safety-first: rejected = most dangerous downgrade):
  max      → deferred   (most dangerous error: critical case sent home)
  moderate → deferred   (common over-reassurance error)
  deferred → moderate   (most common SFT mistake for deferred)

Note: "maximale" never appears in rejected, consistent with a safety-first policy
(we never want the model to learn that predicting max is wrong). "différée" appears
heavily in rejected (~600/900 pairs), correcting the DPO v1 structural bug where
"différée" was never rejected and became an over-predicted safe default.
"""

import argparse
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from datasets import Dataset, concatenate_datasets, load_from_disk
from utils import DPO_COLUMNS, format_triage_response, get_logger

PROJECT_ROOT = _SCRIPTS_DIR.parent
# sft_raw is produced by 02_build_sft.py and available before this script runs.
# data/final/sft is produced by 05_split_and_validate.py (downstream) — do not use here.
SFT_RAW_DIR = PROJECT_ROOT / "data" / "processed" / "sft_raw"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "dpo_raw"

DPO_SOURCE = "sft_synthetic"
SEED = 42

# Target number of DPO pairs after stratified undersampling.
DPO_TARGET_PAIRS = 1000

# Rejected urgency level for each correct level (safety-first policy).
# Each label appears in rejected ~equally often; "différée" is heavily penalised
# to prevent the model from using it as a safe default (v1 structural bug fix).
_URGENCY_SWAP: dict[str, str] = {
    "max": "deferred",
    "moderate": "deferred",
    "deferred": "moderate",
}

# Markers used to extract the clinical body from a formatted triage response.
_EVAL_MARKER = "Évaluation clinique : "
_RECO_MARKER = "\n\nRecommandations : "


# ── Helpers ───────────────────────────────────────────────────────────────────


def extract_clinical_body(response: str) -> str:
    """Extract the clinical evaluation body from a formatted triage response.

    The expected format is:
        URGENCE <LEVEL>

        Évaluation clinique : <body>

        Recommandations : <reco>

    Args:
        response: Full formatted triage response string.

    Returns:
        Clinical evaluation body, or the full response if markers are absent.
    """
    start = response.find(_EVAL_MARKER)
    if start == -1:
        return response
    start += len(_EVAL_MARKER)
    end = response.find(_RECO_MARKER, start)
    if end == -1:
        return response[start:]
    return response[start:end]


def build_synthetic_pair(row: dict) -> dict:
    """Build one DPO pair from an SFT row.

    chosen  = original response (correct urgency label).
    rejected = same clinical body reformatted with the swapped urgency label.

    Args:
        row: SFT raw dataset row with keys instruction, response, urgency_level
             (string: "max" / "moderate" / "deferred"), source, language.

    Returns:
        Dict with keys prompt, chosen, rejected, source, language.
    """
    urgency_level: str = row["urgency_level"]
    wrong_level = _URGENCY_SWAP[urgency_level]
    clinical_body = extract_clinical_body(row["response"])

    return {
        "prompt": row["instruction"],
        "chosen": row["response"],
        "rejected": format_triage_response(wrong_level, clinical_body),
        "source": DPO_SOURCE,
        "language": row["language"],
    }


def stratified_subsample(ds: Dataset, target: int, seed: int = SEED) -> Dataset:
    """Undersample while keeping urgency classes balanced.

    Splits the dataset by urgency_level, takes an equal share from each class,
    concatenates, and shuffles.

    Args:
        ds: Dataset with an ``urgency_level`` column (ClassLabel integers).
        target: Desired total number of pairs.
        seed: Random seed for reproducibility.

    Returns:
        Shuffled dataset of at most ``target`` pairs.
    """
    labels = sorted(set(ds["urgency_level"]))  # strings: "deferred", "max", "moderate"
    per_class = target // len(labels)
    splits: list[Dataset] = []

    for label in labels:
        subset = ds.filter(lambda x, lbl=label: x["urgency_level"] == lbl)
        n = min(per_class, len(subset))
        splits.append(subset.shuffle(seed=seed).select(range(n)))

    return concatenate_datasets(splits).shuffle(seed=seed)


# ── Pipeline ──────────────────────────────────────────────────────────────────


def main() -> None:
    """Build synthetic DPO pairs from the SFT train split and save to disk."""
    parser = argparse.ArgumentParser(description="Build synthetic DPO dataset from SFT train split")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    logger = get_logger("03_build_dpo", verbose=args.verbose)

    if OUTPUT_PATH.exists():
        logger.info("DPO dataset already built at {}, skipping.", OUTPUT_PATH)
        ds = Dataset.load_from_disk(str(OUTPUT_PATH))
        logger.info("  {} pairs.", len(ds))
        return

    if not SFT_RAW_DIR.exists():
        logger.error("SFT raw dataset not found at {}. Run 02_build_sft.py first.", SFT_RAW_DIR)
        return

    logger.info("Loading SFT raw dataset from {}...", SFT_RAW_DIR)
    ds_train = load_from_disk(str(SFT_RAW_DIR))
    logger.info("SFT raw: {} examples.", len(ds_train))

    # Stratified undersampling before transform (cheaper on raw SFT rows).
    logger.info("Stratified undersampling to {} pairs...", DPO_TARGET_PAIRS)
    ds_sampled = stratified_subsample(ds_train, target=DPO_TARGET_PAIRS, seed=SEED)
    logger.info("After undersampling: {} pairs.", len(ds_sampled))

    # Build synthetic pairs.
    ds_dpo = ds_sampled.map(
        build_synthetic_pair,
        remove_columns=ds_sampled.column_names,
        desc="Building synthetic DPO pairs",
    ).select_columns(DPO_COLUMNS)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ds_dpo.save_to_disk(str(OUTPUT_PATH))
    logger.info("DPO dataset: {} pairs saved to {}.", len(ds_dpo), OUTPUT_PATH)

    # Log urgency distribution for sanity check.
    prompts_sample = ds_dpo.select(range(min(10, len(ds_dpo))))
    logger.info("Sample chosen responses (first 80 chars):")
    for row in prompts_sample:
        logger.info("  chosen : {}", row["chosen"][:80].replace("\n", " "))
        logger.info("  rejected: {}", row["rejected"][:80].replace("\n", " "))


if __name__ == "__main__":
    main()
