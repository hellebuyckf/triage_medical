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

Authentification :
    Le token HuggingFace est lu depuis .env (HF_TOKEN=hf_xxx) à la racine du projet.
    Fallback : variable d'environnement HF_TOKEN déjà définie dans le shell.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Charge HF_TOKEN depuis PROJECT_ROOT/.env (sans écraser une valeur déjà présente dans
# le shell — override=False). Silencieux si le fichier est absent.
_PROJECT_ROOT = _SCRIPTS_DIR.parent
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=False)

from datasets import Dataset, DatasetDict, load_from_disk
from huggingface_hub import DatasetCard, DatasetCardData
from utils import get_logger

PROJECT_ROOT = _SCRIPTS_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FINAL_DIR = PROJECT_ROOT / "data" / "final"

# ── Dataset card metadata ──────────────────────────────────────────────────────
# Each entry maps a repo-name suffix to its card metadata.
# Tags follow the HF taxonomy: https://huggingface.co/docs/hub/datasets-cards

_COMMON_TAGS = ["medical", "triage", "project14", "openclassrooms"]

DATASET_METADATA: dict[str, dict] = {
    "project14-sft": {
        "tags": _COMMON_TAGS + ["sft", "instruction-tuning", "multilingual"],
        "language": ["fr", "en"],
        "description": (
            "Supervised Fine-Tuning dataset for a medical triage agent. "
            "~5,000 instruction/response pairs in French and English, "
            "stratified across three urgency levels (max / moderate / deferred). "
            "Sources: FrenchMedMCQA, MedQuAD, MediQAl. "
            "PII anonymised with Presidio (RGPD compliant)."
        ),
    },
    "project14-dpo": {
        "tags": _COMMON_TAGS + ["dpo", "preference-learning", "rlhf"],
        "language": ["en"],
        "description": (
            "Direct Preference Optimisation dataset for a medical triage agent. "
            "~1,000 prompt/chosen/rejected pairs sourced from UltraMedical-Preference. "
            "Human-annotated pairs prioritised during undersampling. "
            "PII anonymised with Presidio (RGPD compliant)."
        ),
    },
    "project14-sft-raw": {
        "tags": _COMMON_TAGS + ["sft", "raw", "intermediate"],
        "language": ["fr", "en"],
        "description": "Intermediate SFT dataset before RGPD anonymisation (project14 pipeline).",
    },
    "project14-dpo-raw": {
        "tags": _COMMON_TAGS + ["dpo", "raw", "intermediate"],
        "language": ["en"],
        "description": "Intermediate DPO dataset before RGPD anonymisation (project14 pipeline).",
    },
    "project14-sft-anonymized": {
        "tags": _COMMON_TAGS + ["sft", "anonymized", "intermediate"],
        "language": ["fr", "en"],
        "description": "Anonymised SFT dataset after Presidio pass, before train/val/test split.",
    },
    "project14-dpo-anonymized": {
        "tags": _COMMON_TAGS + ["dpo", "anonymized", "intermediate"],
        "language": ["en"],
        "description": "Anonymised DPO dataset after Presidio pass, before train/val split.",
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────


def build_dataset_card(
    repo_id: str, tags: list[str], language: list[str], description: str
) -> DatasetCard:
    """Build a minimal HuggingFace DatasetCard with YAML front-matter.

    The front-matter is parsed by the HF Hub search engine to populate
    language filters, task categories, and tags on the dataset page.

    Args:
        repo_id: Full HuggingFace repository identifier (``username/repo-name``).
        tags: List of dataset tags following HF taxonomy.
        language: ISO 639-1 language codes (e.g. ``["fr", "en"]``).
        description: Short description displayed on the dataset page.

    Returns:
        A :class:`huggingface_hub.DatasetCard` ready to be pushed.
    """
    card_data = DatasetCardData(
        language=language,
        license="mit",
        tags=tags,
        task_categories=["text-generation"],
        pretty_name=repo_id.split("/")[-1],
    )
    content = f"---\n{card_data.to_yaml()}---\n\n# {repo_id.split('/')[-1]}\n\n{description}\n"
    return DatasetCard(content)


def push_dataset(
    path: Path,
    repo_id: str,
    private: bool,
    logger,
) -> bool:
    """Load a Dataset or DatasetDict from disk and push it to HuggingFace Hub.

    Works for both simple datasets and dataset dicts — both expose the same
    ``.push_to_hub()`` interface. The type is inferred automatically from the
    Arrow files on disk.

    After a successful data push, a DatasetCard (README.md with YAML front-matter)
    is pushed so the repository is discoverable via HF search and filters.

    Network errors (timeout, HTTP 5xx, connection drop) are caught so that a
    single failing push does not abort the rest of the pipeline.

    Args:
        path: Directory containing the HF Arrow dataset (Dataset or DatasetDict).
        repo_id: HuggingFace repository identifier (``username/repo-name``).
        private: If True, the repository is created in private mode.
        logger: Logger instance.

    Returns:
        ``True`` if the dataset was pushed successfully, ``False`` otherwise.
    """
    if not path.exists():
        logger.warning("Dataset not found: {} — skip.", path)
        return False

    logger.info("Loading {}...", path)
    ds: Dataset | DatasetDict = load_from_disk(str(path))  # type: ignore[assignment]

    if isinstance(ds, DatasetDict):
        logger.info("  Splits: {}", {name: len(split) for name, split in ds.items()})
    else:
        logger.info("  {} examples.", len(ds))

    logger.info("Pushing to hub: {} (private={})...", repo_id, private)
    try:
        ds.push_to_hub(repo_id, private=private)
    except Exception as exc:
        logger.error("  ✗ Push failed for {}: {}", repo_id, exc)
        return False

    logger.info("  ✓ {} published.", repo_id)

    # ── Dataset card ──────────────────────────────────────────────────────────
    repo_suffix = repo_id.split("/")[-1]
    meta = DATASET_METADATA.get(repo_suffix)
    if meta:
        try:
            card = build_dataset_card(
                repo_id=repo_id,
                tags=meta["tags"],
                language=meta["language"],
                description=meta["description"],
            )
            card.push_to_hub(repo_id)
            logger.info("  ✓ Dataset card pushed for {}.", repo_id)
        except Exception as exc:
            # Card failure is non-blocking: data is already on the hub.
            logger.warning("  ⚠ Dataset card push failed for {}: {}", repo_id, exc)
    else:
        logger.warning("  No metadata found for '{}' — card skipped.", repo_suffix)

    return True


def main() -> None:
    """Pipeline de publication vers HuggingFace Hub."""
    parser = argparse.ArgumentParser(
        description="Publication des datasets project14 sur HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Authentification :\n"
            "  Le token est lu depuis .env à la racine du projet (HF_TOKEN=hf_xxx).\n"
            "  Fallback : variable d'env HF_TOKEN déjà définie dans le shell.\n"
            "  Dernier recours : huggingface-cli login\n\n"
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
            "Username HuggingFace requis. Utiliser --username <hf_username> ou définir HF_USERNAME."
        )
        sys.exit(1)

    # Vérification du token avant tout push
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        logger.info("Token HF_TOKEN chargé ({}...{}).", hf_token[:6], hf_token[-4:])
    else:
        logger.warning(
            "HF_TOKEN introuvable dans .env ni dans l'environnement. "
            "Le push utilisera les credentials stockés par huggingface-cli login. "
            "En cas d'erreur 401, ajouter HF_TOKEN=hf_xxx dans .env à la racine du projet."
        )

    username = args.username

    # ── Datasets finaux (DatasetDict) ─────────────────────────────────────────

    final_datasets = [
        (FINAL_DIR / "sft", f"{username}/project14-sft"),
        (FINAL_DIR / "dpo", f"{username}/project14-dpo"),
    ]

    failed: list[str] = []

    for path, repo_id in final_datasets:
        if not push_dataset(path, repo_id, args.private, logger):
            failed.append(repo_id)

    # ── Datasets intermédiaires (Dataset simple) ──────────────────────────────

    if args.include_processed:
        processed_datasets = [
            (PROCESSED_DIR / "sft_raw", f"{username}/project14-sft-raw"),
            (PROCESSED_DIR / "dpo_raw", f"{username}/project14-dpo-raw"),
            (PROCESSED_DIR / "sft_anonymized", f"{username}/project14-sft-anonymized"),
            (PROCESSED_DIR / "dpo_anonymized", f"{username}/project14-dpo-anonymized"),
        ]
        for path, repo_id in processed_datasets:
            if not push_dataset(path, repo_id, args.private, logger):
                failed.append(repo_id)

    if failed:
        logger.error("=== {} push(es) failed: {} ===", len(failed), failed)
        sys.exit(1)

    logger.info("=== All datasets published successfully. ===")


if __name__ == "__main__":
    main()
