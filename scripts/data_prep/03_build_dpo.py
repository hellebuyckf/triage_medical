"""Script 03 — Build the DPO dataset (~1,000 prompt/chosen/rejected pairs)."""

import argparse
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from datasets import Dataset, concatenate_datasets, load_dataset
from utils import DPO_COLUMNS, get_logger, load_datasets_config

PROJECT_ROOT = _SCRIPTS_DIR.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "datasets.yaml"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "dpo_raw"

DPO_SOURCE = "ultramedical_preference"

# Minimum number of words in chosen/rejected to pass quality filter.
MIN_WORDS_QUALITY = 20

# Target number of DPO pairs after undersampling.
DPO_TARGET_PAIRS = 1000


# ── Helpers ───────────────────────────────────────────────────────────────────


def extract_assistant_text(messages: list[dict]) -> str:
    """Extract the assistant turn text from a list of {content, role} messages.

    Args:
        messages: List of message dicts with ``role`` and ``content`` keys.

    Returns:
        Content of the first assistant message, or empty string if not found.
    """
    for msg in messages:
        if msg.get("role") == "assistant":
            return msg.get("content", "").strip()
    return ""


def filter_dpo_quality(row: dict) -> bool:
    """Check quality criteria for a DPO pair.

    chosen/rejected are lists of {content, role} message dicts.

    Args:
        row: Raw dataset row with ``chosen`` and ``rejected`` columns.

    Returns:
        ``True`` if the pair meets all quality thresholds.
    """
    chosen_text = extract_assistant_text(row.get("chosen", []))
    rejected_text = extract_assistant_text(row.get("rejected", []))

    if not chosen_text or not rejected_text:
        return False
    if len(chosen_text.split()) <= MIN_WORDS_QUALITY:
        return False
    if len(rejected_text.split()) <= MIN_WORDS_QUALITY:
        return False
    if chosen_text == rejected_text:
        return False
    return True


def subsample_by_label_type(ds: Dataset, target: int = 1000, seed: int = 42) -> Dataset:
    """Undersample while prioritising human-annotated pairs.

    Args:
        ds: Filtered dataset with a ``label_type`` column.
        target: Desired total number of pairs.
        seed: Random seed for reproducibility.

    Returns:
        Shuffled dataset of at most ``target`` pairs.
    """
    ds_human = ds.filter(lambda x: x.get("label_type") == "human", desc="filter human")
    ds_model = ds.filter(lambda x: x.get("label_type") != "human", desc="filter model")

    n_human = len(ds_human)
    n_model_needed = max(0, target - n_human)

    if n_model_needed > 0 and len(ds_model) > 0:
        ds_model = ds_model.shuffle(seed=seed).select(range(min(n_model_needed, len(ds_model))))
        result = concatenate_datasets([ds_human, ds_model])
    else:
        result = ds_human.select(range(min(target, n_human)))

    return result.shuffle(seed=seed)


def transform_ultramedical(row: dict) -> dict:
    """Transform one UltraMedical-Preference row to the DPO schema.

    Extracts assistant text from the chosen/rejected message lists and
    flattens them into plain strings.

    Args:
        row: Raw dataset row with ``prompt``, ``chosen``, and ``rejected`` columns.

    Returns:
        DPO dict with keys: prompt, chosen, rejected, source, language.
    """
    return {
        "prompt": row["prompt"],
        "chosen": extract_assistant_text(row["chosen"]),
        "rejected": extract_assistant_text(row["rejected"]),
        "source": DPO_SOURCE,
        "language": "en",
    }


# ── Pipeline ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DPO dataset")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to datasets YAML config (default: {DEFAULT_CONFIG})",
    )
    args = parser.parse_args()

    logger = get_logger("03_build_dpo", verbose=args.verbose)

    if OUTPUT_PATH.exists():
        logger.info(f"DPO dataset already built at {OUTPUT_PATH}, skipping.")
        ds = Dataset.load_from_disk(str(OUTPUT_PATH))
        logger.info(f"  {len(ds)} pairs.")
        return

    datasets_config = load_datasets_config(args.config, PROJECT_ROOT)
    if DPO_SOURCE not in datasets_config:
        logger.error(f"'{DPO_SOURCE}' not found in config {args.config}. Run 01_download.py first.")
        return

    ds_config = datasets_config[DPO_SOURCE]
    logger.info(f"Loading {DPO_SOURCE} from HF cache ({ds_config['cache_dir']})...")
    raw = load_dataset(
        ds_config["hf_id"],
        cache_dir=str(ds_config["cache_dir"]),
    )

    # Use the first available split (train expected)
    split_name = list(raw.keys())[0]
    ds_split = raw[split_name]
    logger.info(f"Split '{split_name}': {len(ds_split)} raw pairs.")

    # Quality filtering
    initial_count = len(ds_split)
    ds_filtered = ds_split.filter(filter_dpo_quality, desc="DPO quality filter")
    logger.info(
        f"Quality filter: {initial_count - len(ds_filtered)} pairs rejected, "
        f"{len(ds_filtered)} kept."
    )

    # Stratified undersampling (human-annotated pairs prioritised)
    ds_sampled = subsample_by_label_type(ds_filtered, target=DPO_TARGET_PAIRS, seed=42)
    logger.info(f"After undersampling: {len(ds_sampled)} pairs.")

    # Transform to DPO schema — stays entirely in Arrow, no Python list or DataFrame
    ds_dpo = ds_sampled.map(
        transform_ultramedical,
        remove_columns=ds_sampled.column_names,
        desc="DPO transform",
    ).select_columns(DPO_COLUMNS)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ds_dpo.save_to_disk(str(OUTPUT_PATH))
    logger.info(f"DPO dataset: {len(ds_dpo)} pairs saved to {OUTPUT_PATH}.")


if __name__ == "__main__":
    main()
