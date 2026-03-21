"""Script 01 — Download HuggingFace datasets to data/raw/."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loguru import Logger

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from datasets import Dataset, DatasetDict, load_dataset
from utils import get_logger, load_datasets_config

PROJECT_ROOT = _SCRIPTS_DIR.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "datasets.yaml"


def download_dataset(name: str, config: dict, logger: Logger) -> DatasetDict | None:
    """Load a dataset from HuggingFace, using cache_dir for persistence.

    HuggingFace handles caching automatically: if the dataset is already
    present in ``cache_dir``, it is loaded from disk without re-downloading.

    Args:
        name: Human-readable dataset name used for logging.
        config: Dataset config dict with keys ``hf_id``, ``hf_config``,
            ``cache_dir``, and ``usage``.
        logger: Logger instance.

    Returns:
        The loaded ``DatasetDict``, or ``None`` if an error occurred.
    """
    logger.info(f"[{name}] Loading from {config['hf_id']} (cache: {config['cache_dir']})...")
    kwargs: dict = {
        "path": config["hf_id"],
        "cache_dir": str(config["cache_dir"]),
    }
    if config["hf_config"]:
        kwargs["name"] = config["hf_config"]

    result = load_dataset(**kwargs)
    if isinstance(result, Dataset):
        return DatasetDict({"train": result})
    return result


def print_stats(name: str, ds: DatasetDict, logger: Logger) -> None:
    """Log splits, columns, and 2 random examples per split.

    Args:
        name: Dataset name used as log prefix.
        ds: Loaded ``DatasetDict``.
        logger: Logger instance.
    """
    for split_name, split_ds in ds.items():
        logger.info(f"[{name}] Split '{split_name}': {len(split_ds)} examples")
        logger.info(f"  Columns: {list(split_ds.features.keys())}")
        logger.info(f"  Features: {split_ds.features}")
        indices = random.Random(42).sample(range(len(split_ds)), min(2, len(split_ds)))
        for i in indices:
            logger.info(f"  Example {i}: {split_ds[i]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download HuggingFace datasets")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to datasets YAML config (default: {DEFAULT_CONFIG})",
    )
    args = parser.parse_args()

    logger = get_logger("01_download", verbose=args.verbose)
    logger.info(f"Loading config from {args.config}")
    datasets_config = load_datasets_config(args.config, PROJECT_ROOT)

    successes, failures = [], []

    for name, config in datasets_config.items():
        try:
            ds = download_dataset(name, config, logger)
            if ds is None:
                failures.append(name)
                continue
            print_stats(name, ds, logger)
            successes.append(name)
        except Exception as e:
            logger.error(f"[{name}] Error: {e}")
            failures.append(name)

    logger.info("=== Summary ===")
    logger.info(f"Success ({len(successes)}): {', '.join(successes)}")
    if failures:
        logger.warning(f"Failures ({len(failures)}): {', '.join(failures)}")


if __name__ == "__main__":
    main()
