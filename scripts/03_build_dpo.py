"""Script 03 — Construction du dataset DPO (~1 000 paires prompt/chosen/rejected)."""

import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from tqdm import tqdm

from utils import DPO_COLUMNS, get_logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ultramedical_preference"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "dpo_raw.parquet"


def extract_assistant_text(messages: list[dict]) -> str:
    """Extrait le texte de l'assistant depuis une liste de messages {content, role}."""
    for msg in messages:
        if msg.get("role") == "assistant":
            return msg.get("content", "").strip()
    return ""


def filter_dpo_quality(row: dict) -> bool:
    """Vérifie les critères qualité d'une paire DPO.

    chosen/rejected sont des listes de {content, role}.
    """
    chosen_text = extract_assistant_text(row.get("chosen", []))
    rejected_text = extract_assistant_text(row.get("rejected", []))

    if not chosen_text or not rejected_text:
        return False
    if len(chosen_text.split()) <= 20:
        return False
    if len(rejected_text.split()) <= 20:
        return False
    if chosen_text == rejected_text:
        return False
    if abs(len(chosen_text.split()) - len(rejected_text.split())) <= 5:
        return False
    return True


def subsample_by_label_type(ds: Dataset, target: int = 1000, seed: int = 42) -> Dataset:
    """Sous-échantillonne en priorisant les paires annotées humainement."""
    ds_human = ds.filter(lambda x: x.get("label_type") == "human", desc="Filtre human")
    ds_model = ds.filter(lambda x: x.get("label_type") != "human", desc="Filtre model")

    n_human = len(ds_human)
    n_model_needed = max(0, target - n_human)

    if n_model_needed > 0 and len(ds_model) > 0:
        ds_model = ds_model.shuffle(seed=seed).select(range(min(n_model_needed, len(ds_model))))
        result = concatenate_datasets([ds_human, ds_model])
    else:
        result = ds_human.select(range(min(target, n_human)))

    return result.shuffle(seed=seed)


def transform_ultramedical(row: dict) -> dict:
    """Transforme une ligne UltraMedical-Preference vers le schéma DPO.

    Extrait le texte de l'assistant depuis les listes de messages.
    """
    return {
        "prompt": row["prompt"],
        "chosen": extract_assistant_text(row["chosen"]),
        "rejected": extract_assistant_text(row["rejected"]),
        "source": "ultramedical_preference",
        "language": "en",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Construction du dataset DPO")
    parser.add_argument("--verbose", action="store_true", help="Logging DEBUG")
    args = parser.parse_args()

    logger = get_logger("03_build_dpo", verbose=args.verbose)

    if OUTPUT_PATH.exists():
        logger.info(f"Dataset DPO déjà construit dans {OUTPUT_PATH}, skip.")
        df = pd.read_parquet(OUTPUT_PATH)
        logger.info(f"  {len(df)} paires.")
        return

    if not RAW_DIR.exists():
        logger.error(f"Données brutes non trouvées dans {RAW_DIR}. Lancer d'abord 01_download.py.")
        return

    logger.info("Chargement du dataset UltraMedical-Preference...")
    ds = load_from_disk(str(RAW_DIR))

    # Utiliser le split train (le seul attendu)
    split_name = list(ds.keys())[0]
    ds_split = ds[split_name]
    logger.info(f"Split '{split_name}': {len(ds_split)} paires brutes.")

    # Filtrage qualité
    initial_count = len(ds_split)
    ds_filtered = ds_split.filter(filter_dpo_quality, desc="Filtre qualité DPO")
    rejected_count = initial_count - len(ds_filtered)
    logger.info(f"Filtre qualité : {rejected_count} paires rejetées, {len(ds_filtered)} conservées.")

    # Sous-échantillonnage
    ds_sampled = subsample_by_label_type(ds_filtered, target=1000, seed=42)
    logger.info(f"Après sous-échantillonnage : {len(ds_sampled)} paires.")

    # Transformation vers le schéma DPO
    rows = []
    for row in tqdm(ds_sampled, desc="Transformation DPO"):
        rows.append(transform_ultramedical(row))

    df = pd.DataFrame(rows, columns=DPO_COLUMNS)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"DPO dataset: {len(df)} paires sauvegardées dans {OUTPUT_PATH}.")


if __name__ == "__main__":
    main()
