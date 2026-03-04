"""Script 01 — Téléchargement des datasets HuggingFace vers data/raw/."""

import argparse
import random
from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk

from utils import get_logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASETS_CONFIG = {
    "frenchmedmcqa": {
        "hf_id": "nthngdy/frenchmedmcqa",
        "hf_config": None,
        "local_path": PROJECT_ROOT / "data" / "raw" / "frenchmedmcqa",
        "usage": "sft",
    },
    "medquad": {
        "hf_id": "keivalya/MedQuad-MedicalQnADataset",
        "hf_config": None,
        "local_path": PROJECT_ROOT / "data" / "raw" / "medquad",
        "usage": "sft",
    },
    "mediql_mcqu": {
        "hf_id": "ANR-MALADES/MediQAl",
        "hf_config": "mcqu",
        "local_path": PROJECT_ROOT / "data" / "raw" / "mediql_mcqu",
        "usage": "sft",
    },
    "mediql_oeq": {
        "hf_id": "ANR-MALADES/MediQAl",
        "hf_config": "oeq",
        "local_path": PROJECT_ROOT / "data" / "raw" / "mediql_oeq",
        "usage": "sft",
    },
    "ultramedical_preference": {
        "hf_id": "TsinghuaC3I/UltraMedical-Preference",
        "hf_config": None,
        "local_path": PROJECT_ROOT / "data" / "raw" / "ultramedical_preference",
        "usage": "dpo",
    },
}


def download_dataset(name: str, config: dict, logger) -> DatasetDict | None:
    """Charge un dataset depuis HuggingFace. Retourne None en cas d'échec."""
    path = config["local_path"]

    if path.exists():
        logger.info(f"[{name}] Déjà téléchargé, skip.")
        return load_from_disk(str(path))

    logger.info(f"[{name}] Téléchargement depuis {config['hf_id']}...")
    kwargs = {"path": config["hf_id"]}
    if config["hf_config"]:
        kwargs["name"] = config["hf_config"]

    ds = load_dataset(**kwargs)
    return ds


def print_stats(name: str, ds: DatasetDict, logger) -> None:
    """Affiche les splits, colonnes et 2 exemples aléatoires par split."""
    for split_name, split_ds in ds.items():
        logger.info(f"[{name}] Split '{split_name}': {len(split_ds)} exemples")
        logger.info(f"  Colonnes: {list(split_ds.features.keys())}")
        logger.info(f"  Features: {split_ds.features}")
        indices = random.Random(42).sample(range(len(split_ds)), min(2, len(split_ds)))
        for i in indices:
            logger.info(f"  Exemple {i}: {split_ds[i]}")


def save_dataset(ds: DatasetDict, path: Path, logger) -> None:
    """Sauvegarde le dataset en format Arrow (save_to_disk)."""
    if path.exists():
        logger.info(f"  Déjà sauvegardé dans {path}, skip.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(path))
    logger.info(f"  Sauvegardé dans {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Téléchargement des datasets")
    parser.add_argument("--verbose", action="store_true", help="Logging DEBUG")
    args = parser.parse_args()

    logger = get_logger("01_download", verbose=args.verbose)
    successes, failures = [], []

    for name, config in DATASETS_CONFIG.items():
        try:
            ds = download_dataset(name, config, logger)
            if ds is None:
                failures.append(name)
                continue
            print_stats(name, ds, logger)
            save_dataset(ds, config["local_path"], logger)
            successes.append(name)
        except Exception as e:
            logger.error(f"[{name}] Erreur: {e}")
            failures.append(name)

    logger.info(f"\n=== Résumé ===")
    logger.info(f"Succès ({len(successes)}): {', '.join(successes)}")
    if failures:
        logger.warning(f"Échecs ({len(failures)}): {', '.join(failures)}")


if __name__ == "__main__":
    main()
