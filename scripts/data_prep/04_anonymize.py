"""Script 04 — Anonymisation RGPD avec Presidio."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# spacy.prefer_gpu() must be called before any spaCy model is loaded (i.e. before
# NlpEngineProvider instantiation). Keeping it in its own import block prevents
# ruff/isort from reordering it after the presidio imports.
import spacy

spacy.prefer_gpu()  # Activates GPU support for spaCy if a GPU is available

import pandas as pd
from datasets import Dataset
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from tqdm import tqdm
from utils import filter_presidio_false_positives, get_logger

PROJECT_ROOT = _SCRIPTS_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SFT_INPUT = PROCESSED_DIR / "sft_raw"
DPO_INPUT = PROCESSED_DIR / "dpo_raw"
SFT_OUTPUT = PROCESSED_DIR / "sft_anonymized"
DPO_OUTPUT = PROCESSED_DIR / "dpo_anonymized"
RGPD_REPORT = PROCESSED_DIR / "rgpd_report.md"

ENTITIES = [
    "PERSON",
    "LOCATION",
    "DATE_TIME",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "NRP",
]


def load_presidio_engines() -> tuple[AnalyzerEngine, AnonymizerEngine]:
    """Initialise Presidio avec les modèles spaCy FR et EN."""
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "fr", "model_name": "fr_core_news_md"},
            {"lang_code": "en", "model_name": "en_core_web_md"},
        ],
    }
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()

    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["fr", "en"])
    anonymizer = AnonymizerEngine()

    return analyzer, anonymizer


def anonymize_text(
    text: str,
    analyzer: AnalyzerEngine,
    anonymizer: AnonymizerEngine,
    language: str,
) -> tuple[str, list[dict]]:
    """Analyse et anonymise un texte. Retourne (texte_anonymisé, entités_détectées)."""
    if not text or not text.strip():
        return text, []

    results = analyzer.analyze(text=text, entities=ENTITIES, language=language)
    results = filter_presidio_false_positives(results, text)

    entities_found = [
        {
            "type": r.entity_type,
            "start": r.start,
            "end": r.end,
            "score": r.score,
            "text": text[r.start : r.end],
        }
        for r in results
    ]

    if not results:
        return text, []

    anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized.text, entities_found


def anonymize_dataset(
    df: pd.DataFrame,
    columns: list[str],
    analyzer: AnalyzerEngine,
    anonymizer: AnonymizerEngine,
    desc: str = "Anonymisation",
) -> tuple[pd.DataFrame, dict]:
    """Anonymize the specified text columns of a DataFrame using Presidio.

    Generic function used for both SFT (instruction, response) and DPO
    (prompt, chosen, rejected) datasets. The ``language`` column is used
    to select the correct spaCy model per row.

    Args:
        df: Input DataFrame. Must have a ``language`` column and all columns
            listed in ``columns``.
        columns: Text columns to anonymize.
        analyzer: Presidio AnalyzerEngine instance.
        anonymizer: Presidio AnonymizerEngine instance.
        desc: Label shown in the tqdm progress bar.

    Returns:
        Tuple of (anonymized DataFrame copy, stats dict).
    """
    stats: dict = {
        "total_rows": len(df),
        "rows_with_pii": 0,
        "total_entities_found": 0,
        "entity_type_counts": {},
        "low_confidence_examples": [],
        "examples": [],
    }

    df = df.copy()
    for idx in tqdm(range(len(df)), desc=desc):
        row = df.iloc[idx]
        lang = row["language"]
        all_entities: list[dict] = []
        anonymized_values: dict[str, str] = {}

        for col in columns:
            anon_text, entities = anonymize_text(str(row[col]), analyzer, anonymizer, lang)
            df.at[df.index[idx], col] = anon_text
            anonymized_values[col] = anon_text
            all_entities.extend(entities)

        if all_entities:
            stats["rows_with_pii"] += 1
            stats["total_entities_found"] += len(all_entities)

            for ent in all_entities:
                etype = ent["type"]
                stats["entity_type_counts"][etype] = stats["entity_type_counts"].get(etype, 0) + 1
                if ent["score"] < 0.7:
                    stats["low_confidence_examples"].append(
                        {
                            "row_idx": idx,
                            "entity_type": etype,
                            "text": ent["text"],
                            "score": ent["score"],
                        }
                    )

            if len(stats["examples"]) < 10:
                example: dict = {"entities": all_entities}
                for col in columns:
                    example[f"original_{col}"] = str(row[col])
                    example[f"anonymized_{col}"] = anonymized_values[col]
                stats["examples"].append(example)

    return df, stats


def generate_rgpd_report(sft_stats: dict, dpo_stats: dict) -> str:
    """Génère le rapport RGPD en Markdown."""
    total_rows = sft_stats["total_rows"] + dpo_stats["total_rows"]
    total_pii = sft_stats["rows_with_pii"] + dpo_stats["rows_with_pii"]
    total_entities = sft_stats["total_entities_found"] + dpo_stats["total_entities_found"]

    # Fusionner les compteurs d'entités
    merged_counts = {}
    for counts in [sft_stats["entity_type_counts"], dpo_stats["entity_type_counts"]]:
        for etype, count in counts.items():
            merged_counts[etype] = merged_counts.get(etype, 0) + count

    report = f"""# Rapport d'Anonymisation RGPD
## project14 — Agent de Triage Médical

**Date d'exécution** : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 1. Résumé global

| Métrique | Valeur |
|---|---|
| Exemples traités (SFT + DPO) | {total_rows} |
| Exemples contenant au moins une entité PII | {total_pii} |
| Nombre total d'entités masquées | {total_entities} |

---

## 2. Détail par type d'entité

| Type d'entité | Détections | % des exemples touchés |
|---|---|---|
"""
    for etype, count in sorted(merged_counts.items(), key=lambda x: -x[1]):
        pct = (count / total_rows * 100) if total_rows > 0 else 0
        report += f"| {etype} | {count} | {pct:.1f}% |\n"

    report += """
---

## 3. Exemples de masquages (SFT)

"""
    for i, ex in enumerate(sft_stats["examples"][:10], 1):
        report += f"### Exemple {i}\n\n"
        report += f"**Instruction avant** : {ex.get('original_instruction', '')[:200]}...\n\n"
        report += f"**Instruction après** : {ex.get('anonymized_instruction', '')[:200]}...\n\n"
        report += f"**Entités détectées** : {', '.join(e['type'] + ' (' + e['text'] + ')' for e in ex['entities'][:5])}\n\n"

    # Avertissements
    all_low = sft_stats["low_confidence_examples"] + dpo_stats["low_confidence_examples"]
    report += f"""---

## 4. Avertissements (score de confiance < 0.7)

**{len(all_low)} détections à faible confiance** nécessitant relecture manuelle.

"""
    for item in all_low[:20]:
        report += f'- Ligne {item["row_idx"]}: `{item["entity_type"]}` = "{item["text"]}" (score={item["score"]:.2f})\n'

    report += """
---

## Déclaration de conformité

Toutes les données ont été anonymisées avant utilisation pour l'entraînement du modèle.
Les entités personnelles identifiables (PII) ont été détectées et remplacées par des marqueurs de type
(ex: `<PERSON>`, `<LOCATION>`) conformément au RGPD.
"""

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Anonymisation RGPD")
    parser.add_argument("--verbose", action="store_true", help="Logging DEBUG")
    args = parser.parse_args()

    logger = get_logger("04_anonymize", verbose=args.verbose)

    if SFT_OUTPUT.exists() and DPO_OUTPUT.exists():
        logger.info("Datasets déjà anonymisés, skip.")
        return

    if not SFT_INPUT.exists() or not DPO_INPUT.exists():
        logger.error(
            "Fichiers d'entrée manquants. Lancer 02_build_sft.py et 03_build_dpo.py d'abord."
        )
        return

    logger.info("Initialisation de Presidio...")
    analyzer, anonymizer = load_presidio_engines()

    # SFT
    logger.info("Anonymisation du dataset SFT...")
    df_sft = pd.DataFrame(Dataset.load_from_disk(str(SFT_INPUT)).to_pandas())
    df_sft_anon, sft_stats = anonymize_dataset(
        df_sft, ["instruction", "response"], analyzer, anonymizer, desc="Anonymisation SFT"
    )
    Dataset.from_pandas(df_sft_anon).save_to_disk(str(SFT_OUTPUT))
    logger.info(
        f"SFT anonymisé: {sft_stats['rows_with_pii']} lignes avec PII, "
        f"{sft_stats['total_entities_found']} entités masquées."
    )

    # DPO
    logger.info("Anonymisation du dataset DPO...")
    df_dpo = pd.DataFrame(Dataset.load_from_disk(str(DPO_INPUT)).to_pandas())
    df_dpo_anon, dpo_stats = anonymize_dataset(
        df_dpo, ["prompt", "chosen", "rejected"], analyzer, anonymizer, desc="Anonymisation DPO"
    )
    Dataset.from_pandas(df_dpo_anon).save_to_disk(str(DPO_OUTPUT))
    logger.info(
        f"DPO anonymisé: {dpo_stats['rows_with_pii']} lignes avec PII, "
        f"{dpo_stats['total_entities_found']} entités masquées."
    )

    # Rapport RGPD
    report = generate_rgpd_report(sft_stats, dpo_stats)
    RGPD_REPORT.write_text(report, encoding="utf-8")
    logger.info(f"Rapport RGPD généré dans {RGPD_REPORT}.")


if __name__ == "__main__":
    main()
