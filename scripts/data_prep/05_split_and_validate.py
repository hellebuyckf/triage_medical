"""Script 05 — Train/val/test split, validation and final report."""

import argparse
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from datasets import Dataset, DatasetDict
from utils import DPO_COLUMNS, SFT_COLUMNS, get_logger, md5_hash

PROJECT_ROOT = _SCRIPTS_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FINAL_DIR = PROJECT_ROOT / "data" / "final"

SFT_INPUT = PROCESSED_DIR / "sft_anonymized"
DPO_INPUT = PROCESSED_DIR / "dpo_anonymized"
RGPD_REPORT_SRC = PROCESSED_DIR / "rgpd_report.md"


# ── Split ─────────────────────────────────────────────────────────────────────


def stratified_split_sft(
    ds: Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Dataset, Dataset, Dataset]:
    """Three-way stratified split on urgency_level using HF Dataset.

    Operates on Arrow data directly — no pandas conversion, no RAM duplication.

    Args:
        ds: Input dataset with an ``urgency_level`` column.
        train_ratio: Fraction for training split.
        val_ratio: Fraction for validation split (test gets the remainder).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train, val, test) Dataset objects.
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    split1 = ds.train_test_split(
        test_size=(val_ratio + test_ratio),
        stratify_by_column="urgency_level",
        seed=seed,
    )
    relative_test = test_ratio / (val_ratio + test_ratio)
    split2 = split1["test"].train_test_split(
        test_size=relative_test,
        stratify_by_column="urgency_level",
        seed=seed,
    )
    return split1["train"], split2["train"], split2["test"]


def split_dpo(
    ds: Dataset,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """Simple random split for DPO (no stratification needed).

    Args:
        ds: Input dataset.
        train_ratio: Fraction for the training split.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train, val) Dataset objects.
    """
    splits = ds.train_test_split(test_size=(1.0 - train_ratio), seed=seed)
    return splits["train"], splits["test"]


# ── Deduplication ─────────────────────────────────────────────────────────────


def deduplicate(ds: Dataset, key_col: str, logger) -> Dataset:
    """Remove duplicate rows post-anonymization using Arrow-native select.

    Only the hash column is loaded into Python memory, keeping RAM usage
    proportional to the number of rows (not text length).

    Args:
        ds: Input dataset.
        key_col: Column to hash for duplicate detection.
        logger: Logger instance.

    Returns:
        Deduplicated dataset.
    """
    hashes = [md5_hash(text) for text in ds[key_col]]
    seen: set[str] = set()
    indices: list[int] = []
    for i, h in enumerate(hashes):
        if h not in seen:
            seen.add(h)
            indices.append(i)

    removed = len(ds) - len(indices)
    if removed:
        logger.info(
            f"Post-anonymization dedup on '{key_col}': "
            f"{removed} duplicates removed, {len(indices)} remaining."
        )
        return ds.select(indices)
    return ds


# ── Validation ────────────────────────────────────────────────────────────────


def validate_schema(ds: Dataset, expected_columns: list[str], name: str, logger) -> bool:
    """Check that all expected columns are present and contain no empty values.

    Args:
        ds: Dataset to validate.
        expected_columns: Column names that must be present and non-empty.
        name: Label used in log messages.
        logger: Logger instance.

    Returns:
        ``True`` if the schema is valid.
    """
    missing = set(expected_columns) - set(ds.column_names)
    if missing:
        logger.error(f"[{name}] Missing columns: {missing}")
        return False

    null_counts = {col: sum(1 for v in ds[col] if not v) for col in expected_columns}
    cols_with_nulls = {col: n for col, n in null_counts.items() if n > 0}
    if cols_with_nulls:
        logger.error(f"[{name}] Null/empty values found: {cols_with_nulls}")
        return False

    return True


def validate_no_leakage(train_ds: Dataset, test_ds: Dataset, key_col: str, logger) -> bool:
    """Check there is no MD5-hash overlap between train and test on key_col.

    Args:
        train_ds: Training split.
        test_ds: Test split.
        key_col: Column to hash for leak detection.
        logger: Logger instance.

    Returns:
        ``True`` if there is no overlap.
    """
    train_hashes = {md5_hash(t) for t in train_ds[key_col]}
    test_hashes = {md5_hash(t) for t in test_ds[key_col]}
    overlap = train_hashes & test_hashes
    if overlap:
        logger.error(f"Train/test leakage: {len(overlap)} duplicates detected!")
        return False
    return True


def validate_distribution(ds: Dataset, name: str, logger) -> bool:
    """Check that urgency_level distribution is balanced (gap < 10%).

    Args:
        ds: Dataset with an optional ``urgency_level`` column.
        name: Label used in log messages.
        logger: Logger instance.

    Returns:
        ``True`` if the distribution is balanced (or the column is absent).
    """
    if "urgency_level" not in ds.column_names:
        return True

    counts = Counter(ds["urgency_level"])
    total = len(ds)
    dist = {k: v / total * 100 for k, v in counts.items()}
    logger.info(f"[{name}] Urgency distribution: { {k: f'{v:.1f}%' for k, v in dist.items()} }")

    gap = max(dist.values()) - min(dist.values())
    if gap > 10:
        logger.warning(f"[{name}] Imbalanced distribution: gap = {gap:.1f}%")
        return False
    return True


# ── Report ────────────────────────────────────────────────────────────────────


def compute_split_info(ds: Dataset, name: str) -> dict:
    """Compute statistics for a single split.

    Accesses columns one at a time via Arrow to avoid loading the full dataset
    into RAM as a pandas DataFrame.

    Args:
        ds: Dataset split.
        name: Split name (unused, kept for call-site clarity).

    Returns:
        Dict with count and optional distribution/length statistics.
    """
    info: dict[str, object] = {"count": len(ds)}

    if "urgency_level" in ds.column_names:
        info["urgency_dist"] = dict(Counter(ds["urgency_level"]))
        info["source_dist"] = dict(Counter(ds["source"]))
        info["language_dist"] = dict(Counter(ds["language"]))
        instructions = ds["instruction"]
        responses = ds["response"]
        info["avg_instruction_len"] = sum(len(t.split()) for t in instructions) / len(ds)
        info["avg_response_len"] = sum(len(t.split()) for t in responses) / len(ds)
    elif "prompt" in ds.column_names:
        info["avg_prompt_len"] = sum(len(t.split()) for t in ds["prompt"]) / len(ds)
        info["avg_chosen_len"] = sum(len(t.split()) for t in ds["chosen"]) / len(ds)
        info["avg_rejected_len"] = sum(len(t.split()) for t in ds["rejected"]) / len(ds)

    return info


def generate_stats_report(splits_info: dict) -> str:
    """Generate data/final/stats_report.md.

    Args:
        splits_info: Dict mapping split names to their computed statistics.

    Returns:
        Markdown report as a string.
    """
    report = f"""# Rapport de Statistiques — Datasets Finaux
## project14 — Agent de Triage Médical

**Date de génération** : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Récapitulatif des splits

| Split | Exemples |
|---|---|
"""
    for name, info in splits_info.items():
        report += f"| {name} | {info['count']} |\n"

    report += "\n---\n\n## Distribution SFT\n\n"
    for name in ["sft_train", "sft_val", "sft_test"]:
        if name in splits_info:
            info = splits_info[name]
            report += f"### {name}\n\n"
            if "urgency_dist" in info:
                report += "| Urgency Level | Count | % |\n|---|---|---|\n"
                for level, count in info["urgency_dist"].items():
                    pct = count / info["count"] * 100
                    report += f"| {level} | {count} | {pct:.1f}% |\n"
            if "source_dist" in info:
                report += f"\n**Sources** : {info['source_dist']}\n\n"
            if "language_dist" in info:
                report += f"**Langues** : {info['language_dist']}\n\n"
            if "avg_instruction_len" in info:
                report += f"**Longueur moyenne instruction** : {info['avg_instruction_len']:.0f} tokens\n\n"
                report += (
                    f"**Longueur moyenne response** : {info['avg_response_len']:.0f} tokens\n\n"
                )

    report += "---\n\n## Distribution DPO\n\n"
    for name in ["dpo_train", "dpo_val"]:
        if name in splits_info:
            info = splits_info[name]
            report += f"### {name}\n\n"
            report += f"**Exemples** : {info['count']}\n\n"
            if "avg_prompt_len" in info:
                report += f"**Longueur moyenne prompt** : {info['avg_prompt_len']:.0f} tokens\n\n"
                report += f"**Longueur moyenne chosen** : {info['avg_chosen_len']:.0f} tokens\n\n"
                report += (
                    f"**Longueur moyenne rejected** : {info['avg_rejected_len']:.0f} tokens\n\n"
                )

    report += "---\n\n## Validation\n\n"
    report += "Voir les logs du script `05_split_and_validate.py` pour les résultats de validation détaillés.\n"
    return report


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Split and final validation")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    logger = get_logger("05_split_validate", verbose=args.verbose)

    expected_dirs = [
        FINAL_DIR / "sft" / "dataset_dict.json",
        FINAL_DIR / "dpo" / "dataset_dict.json",
    ]
    if all(f.exists() for f in expected_dirs):
        logger.info("All final files already exist, skipping.")
        return

    if not SFT_INPUT.exists() or not DPO_INPUT.exists():
        logger.error("Anonymized inputs missing. Run 04_anonymize.py first.")
        sys.exit(1)

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    checks_passed = True
    splits_info = {}

    # ── SFT ──
    logger.info("Loading anonymized SFT dataset...")
    ds_sft = Dataset.load_from_disk(str(SFT_INPUT))

    if not validate_schema(ds_sft, SFT_COLUMNS, "SFT", logger):
        checks_passed = False

    # Post-anonymization dedup — anonymization can merge distinct rows into identical ones
    ds_sft = deduplicate(ds_sft, "instruction", logger)

    sft_train, sft_val, sft_test = stratified_split_sft(ds_sft, seed=42)

    size_checks = [
        (len(sft_train) >= 4000, f"SFT train: {len(sft_train)} examples (≥4000)"),
        (len(sft_val) >= 500, f"SFT val: {len(sft_val)} examples (≥500)"),
        (len(sft_test) >= 500, f"SFT test: {len(sft_test)} examples (≥500)"),
    ]
    for ok, msg in size_checks:
        if ok:
            logger.info(f"[✓] {msg}")
        else:
            logger.error(f"[✗] {msg} — below threshold!")
            checks_passed = False

    validate_distribution(sft_train, "SFT train", logger)
    validate_distribution(sft_val, "SFT val", logger)

    if validate_no_leakage(sft_train, sft_test, "instruction", logger):
        logger.info("[✓] No train/test leakage detected (MD5 check on instruction)")
    else:
        checks_passed = False

    DatasetDict({"train": sft_train, "val": sft_val, "test": sft_test}).save_to_disk(
        str(FINAL_DIR / "sft")
    )
    logger.info(f"SFT splits saved to {FINAL_DIR}/sft/")

    splits_info["sft_train"] = compute_split_info(sft_train, "sft_train")
    splits_info["sft_val"] = compute_split_info(sft_val, "sft_val")
    splits_info["sft_test"] = compute_split_info(sft_test, "sft_test")

    # ── DPO ──
    logger.info("Loading anonymized DPO dataset...")
    ds_dpo = Dataset.load_from_disk(str(DPO_INPUT))

    if not validate_schema(ds_dpo, DPO_COLUMNS, "DPO", logger):
        checks_passed = False

    dpo_train, dpo_val = split_dpo(ds_dpo, seed=42)

    identical_count = sum(
        1 for c, r in zip(dpo_train["chosen"], dpo_train["rejected"], strict=True) if c == r
    )
    if identical_count == 0:
        logger.info("[✓] DPO train: chosen != rejected on all examples")
    else:
        logger.error(f"[✗] DPO train: {identical_count} pairs with chosen == rejected!")
        checks_passed = False

    DatasetDict({"train": dpo_train, "val": dpo_val}).save_to_disk(str(FINAL_DIR / "dpo"))
    logger.info(f"DPO splits saved to {FINAL_DIR}/dpo/")

    splits_info["dpo_train"] = compute_split_info(dpo_train, "dpo_train")
    splits_info["dpo_val"] = compute_split_info(dpo_val, "dpo_val")

    # ── Report ──
    report = generate_stats_report(splits_info)
    (FINAL_DIR / "stats_report.md").write_text(report, encoding="utf-8")
    logger.info(f"Stats report generated at {FINAL_DIR}/stats_report.md")

    if RGPD_REPORT_SRC.exists():
        shutil.copy2(RGPD_REPORT_SRC, FINAL_DIR / "rgpd_report.md")
        logger.info(f"RGPD report copied to {FINAL_DIR}/rgpd_report.md")
        logger.info("[!] Low-confidence PII detections require manual review: see rgpd_report.md")

    if not checks_passed:
        logger.error("Some checks failed — review logs above.")
        sys.exit(1)

    logger.info("=== All checks passed. Pipeline completed successfully. ===")


if __name__ == "__main__":
    main()
