"""Script 02 — Construction du dataset SFT unifié (~5 000 paires instruction/response)."""

import argparse
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import pandas as pd
from datasets import Dataset, load_from_disk
from tqdm import tqdm

from utils import (
    SFT_COLUMNS,
    format_triage_response,
    get_logger,
    infer_urgency,
    is_valid_sft_row,
    md5_hash,
)

PROJECT_ROOT = _SCRIPTS_DIR.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sft_raw"


# ── Transformations par source ───────────────────────────────────────────────


def transform_frenchmedmcqa(row: dict) -> dict | None:
    """FrenchMedMCQA → SFT.

    Colonnes: id, question, answer_a..answer_e, correct_answers (int64 index 0-4).
    """
    question = row.get("question", "")
    correct_idx = row.get("correct_answers")

    if not question or correct_idx is None:
        return None

    # correct_answers est un int64 (0=a, 1=b, 2=c, 3=d, 4=e)
    letters = ["a", "b", "c", "d", "e"]
    if isinstance(correct_idx, int) and 0 <= correct_idx <= 4:
        letter = letters[correct_idx]
        correct_text = row.get(f"answer_{letter}", "")
    else:
        return None

    if not correct_text:
        return None

    instruction = "Question médicale : " + question
    raw_response = "Réponse correcte : " + correct_text + "."

    urgency_level, confidence = infer_urgency(instruction + " " + raw_response)

    return {
        "instruction": instruction,
        "response": format_triage_response(urgency_level, raw_response),
        "source": "frenchmedmcqa",
        "language": "fr",
        "urgency_level": urgency_level,
        "confidence": confidence,
    }


def transform_medquad(row: dict) -> dict | None:
    """MedQuAD → SFT.

    Colonnes: qtype, Question, Answer (majuscules !).
    """
    instruction = row.get("Question", "")
    response = row.get("Answer", "")

    if not instruction or not response:
        return None

    urgency_level, confidence = infer_urgency(instruction + " " + response)

    return {
        "instruction": instruction,
        "response": format_triage_response(urgency_level, response),
        "source": "medquad",
        "language": "en",
        "urgency_level": urgency_level,
        "confidence": confidence,
    }


def transform_mediql_mcqu(row: dict) -> dict | None:
    """MediQAl config mcqu → SFT.

    Colonnes: id, clinical_case, question, answer_a..answer_e, correct_answers (str "A".."E"),
              task, medical_subject, question_type.
    """
    question = row.get("question", "")
    if not question:
        return None

    correct_answers = row.get("correct_answers", "")
    if not correct_answers:
        return None

    # correct_answers est une lettre comme "A", "B", etc.
    letter = correct_answers.strip().lower()
    response_text = row.get(f"answer_{letter}", "")

    if not response_text:
        return None

    # Inclure le cas clinique dans l'instruction si disponible
    clinical_case = row.get("clinical_case", "")
    if clinical_case:
        instruction = "Question médicale (examen) : " + clinical_case + "\n" + question
    else:
        instruction = "Question médicale (examen) : " + question

    raw_response = "Réponse correcte : " + response_text + "."

    urgency_level, confidence = infer_urgency(instruction + " " + raw_response)

    return {
        "instruction": instruction,
        "response": format_triage_response(urgency_level, raw_response),
        "source": "mediql_mcqu",
        "language": "fr",
        "urgency_level": urgency_level,
        "confidence": confidence,
    }


def transform_mediql_oeq(row: dict) -> dict | None:
    """MediQAl config oeq → SFT.

    Questions ouvertes : question + answer directe.
    """
    question = row.get("question", "")
    answer = row.get("answer", "") or row.get("correct_answer", "")

    if not question or not answer:
        return None

    instruction = "Question médicale (examen) : " + question
    raw_response = str(answer)

    urgency_level, confidence = infer_urgency(instruction + " " + raw_response)

    return {
        "instruction": instruction,
        "response": format_triage_response(urgency_level, raw_response),
        "source": "mediql_oeq",
        "language": "fr",
        "urgency_level": urgency_level,
        "confidence": confidence,
    }


# ── Pipeline ─────────────────────────────────────────────────────────────────

SOURCES = {
    "frenchmedmcqa": transform_frenchmedmcqa,
    "medquad": transform_medquad,
    "mediql_mcqu": transform_mediql_mcqu,
    "mediql_oeq": transform_mediql_oeq,
}


def load_and_transform(name: str, transform_fn, logger) -> list[dict]:
    """Charge un dataset depuis data/raw/ et applique la transformation."""
    path = RAW_DIR / name
    if not path.exists():
        logger.warning(f"[{name}] Données brutes non trouvées dans {path}, skip.")
        return []

    logger.info(f"[{name}] Chargement depuis {path}...")
    ds = load_from_disk(str(path))

    rows = []
    # Utiliser tous les splits disponibles
    for split_name in ds:
        split_ds = ds[split_name]
        logger.info(f"[{name}] Transformation du split '{split_name}' ({len(split_ds)} ex.)...")
        for row in tqdm(split_ds, desc=f"{name}/{split_name}", leave=False):
            result = transform_fn(row)
            if result and is_valid_sft_row(result["instruction"], result["response"]):
                rows.append(result)

    logger.info(f"[{name}] {len(rows)} exemples valides après transformation et filtrage.")
    return rows


def deduplicate(rows: list[dict], logger) -> list[dict]:
    """Déduplique sur MD5(instruction)."""
    seen = set()
    unique = []
    for row in rows:
        h = md5_hash(row["instruction"])
        if h not in seen:
            seen.add(h)
            unique.append(row)
    removed = len(rows) - len(unique)
    logger.info(f"Déduplication : {removed} doublons supprimés, {len(unique)} restants.")
    return unique


def balance_classes(df: pd.DataFrame, target_total: int = 5000, seed: int = 42, logger=None) -> pd.DataFrame:
    """Sous-échantillonne pour équilibrer urgency_level.

    Cible : target_total // 3 par classe.
    Si une classe a moins d'exemples que la cible, on garde tout
    et on réduit les autres classes à la même taille que la plus petite.
    """
    target_per_class = target_total // 3
    counts = df["urgency_level"].value_counts()
    if logger:
        logger.info(f"Distribution avant équilibrage : {counts.to_dict()}")

    # Cap chaque classe à target_per_class, garde tout si en dessous
    balanced_parts = []
    for level in ["max", "moderate", "deferred"]:
        subset = df[df["urgency_level"] == level]
        n = min(len(subset), target_per_class)
        balanced_parts.append(subset.sample(n=n, random_state=seed))

    result = pd.concat(balanced_parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

    if logger:
        logger.info(f"Distribution après équilibrage : {result['urgency_level'].value_counts().to_dict()}")
    return result


def build_sft(logger) -> pd.DataFrame:
    """Pipeline complet de construction du dataset SFT."""
    all_rows = []
    for name, transform_fn in SOURCES.items():
        rows = load_and_transform(name, transform_fn, logger)
        all_rows.extend(rows)

    logger.info(f"Total brut : {len(all_rows)} exemples.")

    all_rows = deduplicate(all_rows, logger)

    df = pd.DataFrame(all_rows, columns=SFT_COLUMNS)
    logger.info(f"Distribution par source : {df['source'].value_counts().to_dict()}")
    logger.info(f"Distribution par langue : {df['language'].value_counts().to_dict()}")

    df = balance_classes(df, target_total=6500, seed=42, logger=logger)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Construction du dataset SFT")
    parser.add_argument("--verbose", action="store_true", help="Logging DEBUG")
    args = parser.parse_args()

    logger = get_logger("02_build_sft", verbose=args.verbose)

    if OUTPUT_PATH.exists():
        logger.info(f"Dataset SFT déjà construit dans {OUTPUT_PATH}, skip.")
        df = load_from_disk(str(OUTPUT_PATH)).to_pandas()
        logger.info(f"  {len(df)} exemples, distribution : {df['urgency_level'].value_counts().to_dict()}")
        return

    df = build_sft(logger)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_pandas(df).save_to_disk(str(OUTPUT_PATH))
    logger.info(f"SFT dataset: {len(df)} exemples sauvegardés dans {OUTPUT_PATH}.")
    logger.info(f"  Urgency: {df['urgency_level'].value_counts().to_string()}")
    logger.info(f"  Source:  {df['source'].value_counts().to_string()}")
    logger.info(f"  Langue:  {df['language'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
