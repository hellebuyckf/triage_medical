"""Script 06 — Publication des datasets sur HuggingFace Hub.

Pousse les datasets finaux (et optionnellement intermédiaires) depuis
data/ vers l'espace HuggingFace de l'utilisateur.

Datasets finaux (DatasetDict) :
    {username}/project14-sft  →  splits train / val / test
    {username}/project14-dpo  →  splits train / val

Datasets intermédiaires (Dataset simple, --include-processed) :
    {username}/project14-sft-raw
    {username}/project14-dpo-raw
    {username}/project14-sft-anonymized
    {username}/project14-dpo-anonymized

Usage :
    uv run python scripts/data_prep/06_push_to_hub.py --username <hf_username>
    uv run python scripts/data_prep/06_push_to_hub.py --username <hf_username> --private
    uv run python scripts/data_prep/06_push_to_hub.py --username <hf_username> --include-processed
"""

import argparse
import os
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from datasets import Dataset, DatasetDict, load_from_disk

from utils import get_logger

PROJECT_ROOT = _SCRIPTS_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FINAL_DIR = PROJECT_ROOT / "data" / "final"


def push_dataset_dict(
    path: Path,
    repo_id: str,
    private: bool,
    logger,
) -> None:
    """Charge un DatasetDict depuis le disque et le pousse vers HuggingFace Hub.

    Args:
        path: Répertoire contenant le DatasetDict (dataset_dict.json).
        repo_id: Identifiant HuggingFace du dépôt cible (username/repo-name).
        private: Si True, le dépôt est créé en mode privé.
        logger: Logger pour les messages.
    """
    if not path.exists():
        logger.warning("Dataset non trouvé : %s — skip.", path)
        return

    logger.info("Chargement de %s...", path)
    dataset_dict: DatasetDict = load_from_disk(str(path))

    splits_info = {name: len(ds) for name, ds in dataset_dict.items()}
    logger.info("  Splits : %s", splits_info)

    logger.info("Push vers hub : %s (private=%s)...", repo_id, private)
    dataset_dict.push_to_hub(repo_id, private=private)
    logger.info("  ✓ %s publié.", repo_id)


def push_single_dataset(
    path: Path,
    repo_id: str,
    private: bool,
    logger,
) -> None:
    """Charge un Dataset simple depuis le disque et le pousse vers HuggingFace Hub.

    Args:
        path: Répertoire contenant le Dataset HF Arrow.
        repo_id: Identifiant HuggingFace du dépôt cible (username/repo-name).
        private: Si True, le dépôt est créé en mode privé.
        logger: Logger pour les messages.
    """
    if not path.exists():
        logger.warning("Dataset non trouvé : %s — skip.", path)
        return

    logger.info("Chargement de %s...", path)
    ds: Dataset = load_from_disk(str(path))
    logger.info("  %d exemples.", len(ds))

    logger.info("Push vers hub : %s (private=%s)...", repo_id, private)
    ds.push_to_hub(repo_id, private=private)
    logger.info("  ✓ %s publié.", repo_id)


def main() -> None:
    """Pipeline de publication vers HuggingFace Hub."""
    parser = argparse.ArgumentParser(
        description="Publication des datasets project14 sur HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Authentification :\n"
            "  La variable d'environnement HF_TOKEN doit être définie,\n"
            "  ou l'utilisateur doit être connecté via : huggingface-cli login\n\n"
            "Exemples :\n"
            "  uv run python scripts/data_prep/06_push_to_hub.py --username johndoe\n"
            "  uv run python scripts/data_prep/06_push_to_hub.py --username johndoe --private\n"
            "  uv run python scripts/data_prep/06_push_to_hub.py --username johndoe --include-processed\n"
        ),
    )
    parser.add_argument(
        "--username",
        type=str,
        default=os.environ.get("HF_USERNAME", ""),
        help="Username HuggingFace (ou variable d'env HF_USERNAME).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Crée les dépôts en mode privé (par défaut : public).",
    )
    parser.add_argument(
        "--include-processed",
        action="store_true",
        help="Publie aussi les datasets intermédiaires (sft_raw, dpo_raw, anonymized).",
    )
    parser.add_argument("--verbose", action="store_true", help="Logging DEBUG.")
    args = parser.parse_args()

    logger = get_logger("06_push_to_hub", verbose=args.verbose)

    if not args.username:
        logger.error(
            "Username HuggingFace requis. "
            "Utiliser --username <hf_username> ou définir HF_USERNAME."
        )
        sys.exit(1)

    username = args.username

    # ── Datasets finaux (DatasetDict) ─────────────────────────────────────────

    final_datasets = [
        (FINAL_DIR / "sft", f"{username}/project14-sft"),
        (FINAL_DIR / "dpo", f"{username}/project14-dpo"),
    ]

    for path, repo_id in final_datasets:
        push_dataset_dict(path, repo_id, args.private, logger)

    # ── Datasets intermédiaires (Dataset simple) ──────────────────────────────

    if args.include_processed:
        processed_datasets = [
            (PROCESSED_DIR / "sft_raw", f"{username}/project14-sft-raw"),
            (PROCESSED_DIR / "dpo_raw", f"{username}/project14-dpo-raw"),
            (PROCESSED_DIR / "sft_anonymized", f"{username}/project14-sft-anonymized"),
            (PROCESSED_DIR / "dpo_anonymized", f"{username}/project14-dpo-anonymized"),
        ]
        for path, repo_id in processed_datasets:
            push_single_dataset(path, repo_id, args.private, logger)

    logger.info("=== Publication terminée. ===")


if __name__ == "__main__":
    main()
