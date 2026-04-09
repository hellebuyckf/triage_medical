"""Script 03 — Build the DPO dataset (~1,000 synthetic prompt/chosen/rejected pairs).

Strategy: synthetic pairs from the SFT train split.
  - chosen  : the correct triage response (ground-truth urgency label).
  - rejected : same clinical body, reformatted with a plausible wrong urgency label.

This approach ensures DPO directly corrects the SFT model's urgency-classification
errors instead of learning stylistic preferences from misaligned academic text
(UltraMedical-Preference was abandoned — see memory/project_dpo_strategy.md).

Urgency swap table (safety-first: rejected = most common real confusion):
  max      → moderate   (most frequent real confusion: 59 MAX→MOD vs 16 MAX→DEF)
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

from collections import Counter

from datasets import Dataset, concatenate_datasets, load_from_disk
from utils import DPO_COLUMNS, extract_urgency_from_response, format_triage_response, get_logger

PROJECT_ROOT = _SCRIPTS_DIR.parent
# sft_raw is produced by 02_build_sft.py and available before this script runs.
# data/final/sft is produced by 05_split_and_validate.py (downstream) — do not use here.
SFT_RAW_DIR = PROJECT_ROOT / "data" / "processed" / "sft_raw"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "dpo_raw"

DPO_SOURCE = "sft_synthetic"
SEED = 42

# Target number of synthetic DPO pairs after stratified undersampling.
# When hard negatives are available, synthetic pairs are capped at this value
# and merged with all available hard negatives.
DPO_TARGET_PAIRS = 500

HARD_NEGATIVES_PATH = PROJECT_ROOT / "data" / "processed" / "dpo_hard_negatives"

# Rejected urgency level for each correct level (safety-first policy).
# MAX → MODERATE: targets the dominant real confusion (59 MAX→MOD vs 16 MAX→DEF).
# DIFFÉRÉE appears in rejected ~167 times (from moderate pairs only), same as MODERATE
# (from deferred pairs), giving balanced penalisation and avoiding the v1 bias
# where DIFFÉRÉE was over-penalised (~334×) and became a model blind spot.
_URGENCY_SWAP: dict[str, str] = {
    "max": "moderate",
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


# ── Helpers ───────────────────────────────────────────────────────────────────


def deduplicate_on_prompt(ds: Dataset, seed: int = SEED) -> Dataset:
    """Remove duplicate prompts, keeping one pair per unique prompt.

    Duplicate prompts arise from anonymisation: different pathologies anonymised
    to the same ``<PERSON>`` placeholder produce identical prompts with different
    urgency labels, sending contradictory DPO gradients.

    When a prompt appears multiple times, the pair with the highest-urgency
    ``chosen`` label is kept (safety-first: prefer max > moderate > deferred).

    Args:
        ds: DPO dataset with ``prompt`` and ``chosen`` columns.
        seed: Random seed used to shuffle before deduplication.

    Returns:
        Dataset with at most one pair per unique prompt.
    """
    _urgency_rank = {"max": 2, "moderate": 1, "deferred": 0}

    # Shuffle first so that ties between same-rank pairs are broken randomly.
    ds = ds.shuffle(seed=seed)

    seen: set[str] = set()
    keep_indices: list[int] = []
    # Sort by urgency rank descending so the highest-urgency pair is encountered first.
    ranked = sorted(
        range(len(ds)),
        key=lambda i: _urgency_rank.get(
            extract_urgency_from_response(ds[i]["chosen"]) or "deferred", 0
        ),
        reverse=True,
    )
    for idx in ranked:
        prompt = ds[idx]["prompt"]
        if prompt not in seen:
            seen.add(prompt)
            keep_indices.append(idx)

    return ds.select(sorted(keep_indices))


# ── Pipeline ──────────────────────────────────────────────────────────────────


def _log_label_distribution(ds: Dataset, name: str, logger) -> None:  # type: ignore[no-untyped-def]
    """Log chosen/rejected label distribution for a dataset."""
    chosen_labels = [extract_urgency_from_response(r) or "unknown" for r in ds["chosen"]]
    rejected_labels = [extract_urgency_from_response(r) or "unknown" for r in ds["rejected"]]
    logger.info("  {} chosen  : {}", name, dict(Counter(chosen_labels)))
    logger.info("  {} rejected: {}", name, dict(Counter(rejected_labels)))


def main() -> None:
    """Build DPO dataset: synthetic pairs + optional hard negatives, deduplicated."""
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

    # ── Synthetic pairs ────────────────────────────────────────────────────────
    logger.info("Stratified undersampling to {} synthetic pairs...", DPO_TARGET_PAIRS)
    ds_sampled = stratified_subsample(ds_train, target=DPO_TARGET_PAIRS, seed=SEED)
    ds_synthetic = ds_sampled.map(
        build_synthetic_pair,
        remove_columns=ds_sampled.column_names,
        desc="Building synthetic DPO pairs",
    ).select_columns(DPO_COLUMNS)

    # Deduplicate on prompt to remove contradictory training signal from
    # anonymised prompts (e.g. "<PERSON> syndrome" mapping to 3 urgency levels).
    n_before = len(ds_synthetic)
    ds_synthetic = deduplicate_on_prompt(ds_synthetic, seed=SEED)
    logger.info(
        "Synthetic pairs: {} → {} after deduplication ({} removed).",
        n_before,
        len(ds_synthetic),
        n_before - len(ds_synthetic),
    )
    _log_label_distribution(ds_synthetic, "synthetic", logger)

    # ── Hard negatives (optional) ──────────────────────────────────────────────
    if HARD_NEGATIVES_PATH.exists():
        ds_hard = Dataset.load_from_disk(str(HARD_NEGATIVES_PATH))
        logger.info("Hard negatives loaded: {} pairs from {}.", len(ds_hard), HARD_NEGATIVES_PATH)
        # Cap hard negatives to match synthetic count — prevents HN from dominating the
        # training signal (1674 HN vs 495 synthetic = 77% HN caused max-collapse in Run C).
        if len(ds_hard) > DPO_TARGET_PAIRS:
            n_hn_before = len(ds_hard)
            ds_hard = ds_hard.shuffle(seed=SEED).select(range(DPO_TARGET_PAIRS))
            logger.info("Hard negatives capped at {} (was {}).", DPO_TARGET_PAIRS, n_hn_before)
        _log_label_distribution(ds_hard, "hard-neg", logger)
        ds_dpo = concatenate_datasets([ds_synthetic, ds_hard]).shuffle(seed=SEED)
        logger.info(
            "Final DPO dataset: {} synthetic + {} hard-neg = {} pairs.",
            len(ds_synthetic),
            len(ds_hard),
            len(ds_dpo),
        )
    else:
        logger.info(
            "No hard negatives found at {} — using synthetic pairs only.", HARD_NEGATIVES_PATH
        )
        logger.info("Run 'make sft-errors' after SFT training to generate hard negatives.")
        ds_dpo = ds_synthetic.shuffle(seed=SEED)

    # Deduplicate on the final merged dataset — synthetic dedup above only covers synthetic pairs;
    # the same prompt can appear in both synthetic and hard negatives with conflicting chosen labels.
    n_before = len(ds_dpo)
    ds_dpo = deduplicate_on_prompt(ds_dpo, seed=SEED)
    logger.info(
        "Final deduplication: {} → {} pairs ({} removed).",
        n_before,
        len(ds_dpo),
        n_before - len(ds_dpo),
    )
    _log_label_distribution(ds_dpo, "final", logger)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ds_dpo.save_to_disk(str(OUTPUT_PATH))
    logger.info("DPO dataset: {} pairs saved to {}.", len(ds_dpo), OUTPUT_PATH)


if __name__ == "__main__":
    main()
