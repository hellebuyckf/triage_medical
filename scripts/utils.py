"""Module partagé — constantes, filtres qualité, inférence d'urgence, logging."""

from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from loguru import logger as _loguru_logger

if TYPE_CHECKING:
    from loguru import Logger as _LoguruLogger

# ── Schémas de données ───────────────────────────────────────────────────────

SFT_COLUMNS = ["instruction", "response", "source", "language", "urgency_level", "confidence"]
DPO_COLUMNS = ["prompt", "chosen", "rejected", "source", "language"]
URGENCY_LEVELS = ["max", "moderate", "deferred"]

# ── Mots-clés d'urgence (bilingues) ──────────────────────────────────────────

URGENCY_MAX_KEYWORDS = [
    # FR
    "douleur thoracique",
    "difficultés respiratoires",
    "perte de conscience",
    "arrêt cardiaque",
    "hémorragie",
    "anaphylaxie",
    "AVC",
    "urgence vitale",
    "danger vital",
    "convulsions",
    "coma",
    "détresse respiratoire",
    # EN
    "chest pain",
    "difficulty breathing",
    "loss of consciousness",
    "cardiac arrest",
    "hemorrhage",
    "anaphylaxis",
    "stroke",
    "life-threatening",
    "emergency",
    "seizure",
    "unconscious",
]

URGENCY_DEFERRED_KEYWORDS = [
    # FR
    "rhume",
    "légère douleur",
    "fatigue chronique",
    "médecin traitant",
    "rendez-vous",
    "peut attendre",
    "suivi régulier",
    "vaccin",
    "prévention",
    "dépistage",
    "bilan",
    "contrôle",
    "consultation",
    "vitamines",
    "nutrition",
    "hygiène",
    "allergie saisonnière",
    "eczéma",
    "acné",
    "insomnie",
    "constipation",
    "régime",
    # EN
    "cold",
    "mild pain",
    "chronic fatigue",
    "general practitioner",
    "appointment",
    "can wait",
    "routine",
    "follow-up",
    "vaccine",
    "prevention",
    "screening",
    "check-up",
    "annual",
    "wellness",
    "vitamins",
    "nutrition",
    "hygiene",
    "seasonal allergy",
    "eczema",
    "acne",
    "insomnia",
    "constipation",
    "diet",
    "supplement",
    "lifestyle",
    "rehabilitation",
    "physical therapy",
    "counseling",
    "medication management",
    "refill",
]

# Pré-compilation des patterns pour la performance
_MAX_PATTERN = re.compile("|".join(re.escape(kw) for kw in URGENCY_MAX_KEYWORDS), re.IGNORECASE)
_DEFERRED_PATTERN = re.compile(
    "|".join(re.escape(kw) for kw in URGENCY_DEFERRED_KEYWORDS), re.IGNORECASE
)

# Patterns pour le filtrage qualité
_HTML_PATTERN = re.compile(r"<[^>]+>|&[a-z]+;", re.IGNORECASE)
_URL_PATTERN = re.compile(r"https?://|www\.", re.IGNORECASE)


# ── Fonctions ────────────────────────────────────────────────────────────────


def infer_urgency(text: str) -> tuple[str, float]:
    """Infère le niveau d'urgence depuis le texte combiné instruction+response.

    Returns:
        (urgency_level, confidence) — "max"/0.8, "deferred"/0.8, ou "moderate"/0.7.
    """
    text_lower = text.lower()
    if _MAX_PATTERN.search(text_lower):
        return "max", 0.8
    if _DEFERRED_PATTERN.search(text_lower):
        return "deferred", 0.8
    return "moderate", 0.7


def is_valid_sft_row(instruction: str, response: str, min_tokens: int = 20) -> bool:
    """Vérifie qu'une paire SFT respecte les critères qualité.

    - instruction non vide
    - response >= min_tokens mots
    - pas de HTML résiduel
    - pas d'URL
    """
    if not instruction or not instruction.strip():
        return False
    if not response or len(response.split()) < min_tokens:
        return False
    combined = instruction + " " + response
    if _HTML_PATTERN.search(combined):
        return False
    if _URL_PATTERN.search(combined):
        return False
    return True


def md5_hash(text: str) -> str:
    """Hash MD5 pour détection de doublons sur instruction."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ── Triage response formatting (S2 — SFT dataset) ────────────────────────────

_URGENCY_LABELS: dict[str, str] = {
    "max": "URGENCE MAXIMALE",
    "moderate": "URGENCE MODÉRÉE",
    "deferred": "URGENCE DIFFÉRÉE",
}

_URGENCY_RECOMMENDATIONS: dict[str, str] = {
    "max": "Appelez le 15 (SAMU) ou rendez-vous aux urgences immédiatement. Ne restez pas seul.",
    "moderate": "Consultez un médecin ou une unité de soins urgents dans les 24 à 48 heures.",
    "deferred": "Prenez rendez-vous avec votre médecin traitant dans les prochains jours.",
}

_MAX_EVAL_CHARS = 500


def format_triage_response(urgency_level: str, source_response: str) -> str:
    """Formate une réponse de triage médical au format attendu par le modèle.

    Transforme une réponse brute (Wikipedia-style) en réponse structurée incluant :
    - Le niveau d'urgence (URGENCE MAXIMALE / MODÉRÉE / DIFFÉRÉE)
    - Une évaluation clinique (extraite de la réponse source)
    - Des recommandations adaptées au niveau d'urgence

    Ce format est indispensable pour que ``extract_urgency_from_response()``
    puisse parser les prédictions lors de l'évaluation.

    Args:
        urgency_level: Niveau d'urgence inféré ("max", "moderate", "deferred").
        source_response: Réponse brute issue des datasets sources (MedQuAD, etc.).

    Returns:
        Réponse structurée au format triage, prête pour le fine-tuning SFT.

    Example:
        >>> format_triage_response("max", "Chest pain can indicate a myocardial infarction.")
        'URGENCE MAXIMALE\\n\\nÉvaluation clinique : Chest pain can indicate...\\n\\nRecommandations : Appelez le 15...'
    """
    label = _URGENCY_LABELS.get(urgency_level, _URGENCY_LABELS["moderate"])
    reco = _URGENCY_RECOMMENDATIONS.get(urgency_level, _URGENCY_RECOMMENDATIONS["moderate"])

    eval_text = source_response.strip()
    if len(eval_text) > _MAX_EVAL_CHARS:
        eval_text = eval_text[: _MAX_EVAL_CHARS - 3] + "..."

    return f"{label}\n\nÉvaluation clinique : {eval_text}\n\nRecommandations : {reco}"


# ── Prompt formatting (S2 — SFT) ─────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un agent de triage médical pour le Centre Hospitalier Saint-Aurélien.
Analyse les symptômes décrits et fournis :
1. Le niveau d'urgence : URGENCE MAXIMALE / URGENCE MODÉRÉE / URGENCE DIFFÉRÉE
2. Une évaluation clinique brève
3. Des recommandations concrètes

Règles absolues :
- Réponds TOUJOURS en français, même si les symptômes sont décrits en anglais.
- N'utilise jamais de marqueurs d'anonymisation comme <PERSON>, <LOCATION>, <DATE>, etc.
- Si tu ne connais pas un nom propre, omets-le simplement.

⚠️ Cet agent est un outil d'aide au triage, pas un diagnostic médical."""


def format_chat_prompt(instruction: str, response: str = "") -> str:
    """Formate un exemple SFT au format ChatML Qwen3.

    Args:
        instruction: Le texte de la question/instruction utilisateur.
        response: La réponse attendue. Si vide, génère un prompt d'inférence
            (sans le tour assistant), utilisé pour la génération.

    Returns:
        Le prompt formaté au format ChatML complet ou partiel.
    """
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n{instruction}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    if response:
        prompt += f"{response}<|im_end|>"
    return prompt


# ── Urgency extraction (S2 — évaluation) ─────────────────────────────────────

_URGENCY_PATTERNS: dict[str, re.Pattern] = {
    "max": re.compile(r"MAXIMALE|maximale|urgence\s+max\b", re.IGNORECASE),
    "moderate": re.compile(r"MOD[ÉE]R[ÉE]E|mod[eé]r[eé]e|urgence\s+mod", re.IGNORECASE),
    "deferred": re.compile(r"DIFF[ÉE]R[ÉE]E|diff[eé]r[eé]e|deferred", re.IGNORECASE),
}


def extract_urgency_from_response(text: str) -> str | None:
    """Extrait le niveau d'urgence depuis une réponse générée par le modèle.

    Cherche les patterns uniquement dans les 150 premiers caractères :
    le label d'urgence doit apparaître en tête de réponse (première ligne).
    Limiter la recherche évite de capter des labels spurieux générés dans
    un second tour "user/assistant" en cas d'arrêt EOS défaillant.

    Patterns recherchés :
    - "URGENCE MAXIMALE" / "maximale" / "urgence max" → "max"
    - "URGENCE MODÉRÉE" / "modérée" / "urgence mod" → "moderate"
    - "URGENCE DIFFÉRÉE" / "différée" / "deferred" → "deferred"

    Args:
        text: Le texte de la réponse générée.

    Returns:
        "max", "moderate", "deferred", ou None si non trouvé.
    """
    # Limiter la recherche au début de la réponse où le label doit se trouver.
    # "URGENCE MAXIMALE\n\nÉvaluation clinique..." → label dans les 20 premiers chars.
    # Marge de 150 chars pour absorber d'éventuels espaces ou tokens parasites.
    search_text = text[:150]
    for level, pattern in _URGENCY_PATTERNS.items():
        if pattern.search(search_text):
            return level
    return None


# ── Anonymisation — filtre faux positifs Presidio (S1) ───────────────────────

# Seuil de confiance minimum pour les entités PERSON.
# Les datasets sources (MedQuAD, FrenchMedMCQA, MediQAl) sont des bases de
# connaissances médicales, pas des données patient. Le NER spaCy génère de nombreux
# faux positifs sur les éponymes médicaux (noms de syndromes, maladies, médicaments).
# Un seuil de 0.85 élimine la plupart sans sacrifier les vrais noms de personnes.
PERSON_CONFIDENCE_THRESHOLD = 0.85

# Termes médicaux à ne jamais masquer même si détectés comme PERSON.
# Inclut : éponymes (syndromes, maladies), organes mal détectés, abréviations médicales.
MEDICAL_TERMS_ALLOWLIST: frozenset[str] = frozenset(
    {
        # Éponymes — maladies et syndromes (EN)
        "alzheimer",
        "parkinson",
        "huntington",
        "crohn",
        "hodgkin",
        "addison",
        "cushing",
        "graves",
        "hashimoto",
        "wilson",
        "marfan",
        "turner",
        "down",
        "raynaud",
        "behcet",
        "sjogren",
        "sjögren",
        "brugada",
        "wolff",
        "klinefelter",
        "noonan",
        "prader",
        "willi",
        "angelman",
        "rett",
        "duchenne",
        "becker",
        "charcot",
        "marie",
        "tooth",
        "gaucher",
        "fabry",
        "niemann",
        "pick",
        "pompe",
        "hurler",
        "hunter",
        "sanfilippo",
        "morquio",
        "tay",
        "sachs",
        "canavan",
        "krabbe",
        "batten",
        "spielmeyer",
        "vogt",
        "aicardi",
        "goutières",
        "aicardi-goutières",
        "lennox",
        "gastaut",
        "dravet",
        "landau",
        "kleffner",
        "sturge",
        "weber",
        "von hippel",
        "lindau",
        "neurofibromatosis",
        "tuberous",
        "sézary",
        "szary",
        "paget",
        "bowen",
        "kaposi",
        "burkitt",
        "wilms",
        "ewing",
        "pott",
        "bright",
        "berger",
        "henoch",
        "schönlein",
        "wegener",
        "goodpasture",
        "buerger",
        "takayasu",
        "horton",
        "kawasaki",
        "still",
        "felty",
        "reiter",
        "behçet",
        "whipple",
        "menetrier",
        "zollinger",
        "ellison",
        "sipple",
        "wermer",
        "verner",
        "morrison",
        "ogilvie",
        "hirschsprung",
        "meckel",
        "peutz",
        "jeghers",
        "lynch",
        "cowden",
        "bannayan",
        "riley",
        "day",
        "fanconi",
        "blackfan",
        "diamond",
        "shwachman",
        "kostmann",
        "chediak",
        "higashi",
        "wiskott",
        "aldrich",
        "bruton",
        "digeorge",
        "treacher",
        "collins",
        "pierre",
        "robin",
        "goldenhar",
        "charge",
        "vacter",
        "vacterl",
        # Éponymes — maladies et syndromes (FR)
        "basedow",
        "quincke",
        "biermer",
        "leriche",
        "osler",
        "rendu",
        "barre",
        "guillain",
        "millard",
        "gubler",
        # Procédures / examens
        "x-ray",
        "x ray",
        "mri",
        "ct scan",
        "ercp",
        # Organes / termes anatomiques mal détectés
        "lung",
        "heart",
        "kidney",
        "liver",
        "spleen",
        "colon",
        # Institutions médicales
        "nih",
        "cdc",
        "who",
        "nhlbi",
        # Médicaments courants (marques taguées comme personnes)
        "coversyl",
        "perindopril",
        "levothyrox",
        "doliprane",
        "aspirin",
        # Termes génétiques
        "glut",
        "brca",
        "cftr",
        "mthfr",
        # Abbréviations souvent mal taguées
        "arp",
        "ags",
        # Labels d'urgence du format triage — ne jamais anonymiser
        # Presidio/spaCy anglais détecte "URGENCE MODÉRÉE" comme entité PERSON (mot français = nom étranger)
        "urgence",
        "maximale",
        "modérée",
        "différée",
    }
)


def filter_presidio_false_positives(
    results: list,
    text: str,
    person_threshold: float = PERSON_CONFIDENCE_THRESHOLD,
    allowlist: frozenset[str] = MEDICAL_TERMS_ALLOWLIST,
) -> list:
    """Filtre les faux positifs Presidio sur l'entité PERSON.

    Deux règles de filtrage pour PERSON :
    1. Score de confiance < person_threshold → exclu (détection trop incertaine).
    2. Texte détecté dans l'allowlist médicale → exclu (éponyme ou terme médical).

    Les autres entités (LOCATION, DATE_TIME, etc.) ne sont pas filtrées.

    Args:
        results: Liste des RecognizerResult retournés par AnalyzerEngine.analyze().
        text: Texte original analysé (pour extraire le span détecté).
        person_threshold: Seuil de confiance minimum pour PERSON.
        allowlist: Ensemble de termes médicaux à ne pas masquer.

    Returns:
        Liste filtrée de RecognizerResult.
    """
    filtered = []
    for result in results:
        if result.entity_type == "PERSON":
            if result.score < person_threshold:
                continue
            detected = text[result.start : result.end].lower().strip()
            if detected in allowlist:
                continue
            # Vérifier si le terme contient un mot de l'allowlist
            if any(term in detected for term in allowlist):
                continue
        filtered.append(result)
    return filtered


# ── DPO formatting (S3) ──────────────────────────────────────────────────────


def format_dpo_prompt(prompt: str) -> str:
    """Formate un prompt DPO au format ChatML Qwen3 (sans le tour assistant).

    Utilisé pour la colonne 'prompt' du DPOTrainer. Le trainer concatène
    ce préfixe avec chosen/rejected pour former les séquences d'entraînement.

    Args:
        prompt: L'instruction ou question médicale brute.

    Returns:
        Prompt formaté en ChatML, terminant par ``<|im_start|>assistant\\n``.
    """
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n{prompt}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def format_dpo_response(response: str) -> str:
    """Formate une réponse DPO (chosen ou rejected) avec le token de fin de tour.

    Args:
        response: Texte de la réponse de l'assistant.

    Returns:
        Réponse avec ``<|im_end|>`` en suffixe (EOS du tour assistant en ChatML).
    """
    return f"{response}<|im_end|>"


# ── Demo environment guard ───────────────────────────────────────────────────


def check_demo_env() -> None:
    """Validate required environment variables before running against the GCP MLflow backend.

    Called at the start of each training/evaluation script. Detects demo mode
    when ``MLFLOW_TRACKING_URI`` starts with ``https://`` (Cloud Run endpoint).

    Checks:
    - ``MLFLOW_TRACKING_URI``: must be set and start with ``https://``.
    - ``MLFLOW_TRACKING_USERNAME``: must be set and non-empty (MLflow basic-auth username).
    - ``MLFLOW_TRACKING_PASSWORD``: must be set and non-empty (MLflow basic-auth password).
    - ``google-cloud-storage``: must be installed for GCS artifact storage.

    Exits with a clear error message if a required variable is missing, so
    the problem is caught immediately rather than after hours of training.

    No-op when ``MLFLOW_TRACKING_URI`` is a local SQLite URI (dev mode).
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not tracking_uri.startswith("https://"):
        return  # dev mode — no GCP auth required

    errors: list[str] = []
    warnings: list[str] = []

    username = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
    if not username:
        errors.append(
            "  • MLFLOW_TRACKING_USERNAME is not set.\n"
            "    Add MLFLOW_TRACKING_USERNAME=admin to your .env.demo file."
        )

    password = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")
    if not password:
        errors.append(
            "  • MLFLOW_TRACKING_PASSWORD is not set.\n"
            "    Add MLFLOW_TRACKING_PASSWORD=<password> to your .env.demo file."
        )

    # google-cloud-storage is required by MLflow to read/write artifacts on GCS.
    # It is an optional MLflow dependency installed via: uv sync --extra gcp
    try:
        import importlib

        importlib.import_module("google.cloud.storage")
    except ImportError:
        errors.append(
            "  • google-cloud-storage is not installed.\n"
            "    Run: uv sync --extra gcp\n"
            "    Or via make: make train-sft ENV=demo  (setup-gcp runs automatically)"
        )

    if errors:
        lines = "\n".join(errors)
        print(
            f"\n[check_demo_env] ✗ Missing required variables for demo (GCP) mode:\n{lines}\n",
            file=sys.stderr,
        )
        sys.exit(1)

    if warnings:
        lines = "\n".join(warnings)
        print(
            f"\n[check_demo_env] ⚠ Warnings for demo (GCP) mode:\n{lines}\n",
            file=sys.stderr,
        )


# ── Checkpoint helpers (S2 — entraînement) ───────────────────────────────────


def get_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Retourne le chemin du dernier checkpoint dans checkpoint_dir.

    Cherche les dossiers nommés "checkpoint-<step>" et retourne
    celui avec le numéro de step le plus élevé.

    Args:
        checkpoint_dir: Répertoire contenant les checkpoints.

    Returns:
        Path vers le dernier checkpoint, ou None si aucun trouvé.
    """
    if not checkpoint_dir.exists():
        return None
    checkpoints = sorted(
        [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    return checkpoints[-1] if checkpoints else None


# ── Dataset config loader ─────────────────────────────────────────────────────


def load_datasets_config(config_path: Path, project_root: Path) -> dict:
    """Load and resolve dataset configurations from a YAML file.

    Resolves ``cache_dir`` entries to absolute ``pathlib.Path`` objects
    relative to ``project_root``.

    Args:
        config_path: Path to the YAML configuration file.
        project_root: Root of the project, used to resolve relative ``cache_dir`` paths.

    Returns:
        Dictionary mapping dataset names to their resolved configurations.
        Each entry has keys: ``hf_id`` (str), ``hf_config`` (str | None),
        ``cache_dir`` (Path), ``usage`` (str).

    Raises:
        FileNotFoundError: If ``config_path`` does not exist.
        KeyError: If the YAML file is missing the top-level ``datasets`` key.
    """
    with config_path.open() as f:
        raw = yaml.safe_load(f)

    return {
        name: {
            "hf_id": entry["hf_id"],
            "hf_config": entry.get("hf_config"),
            "cache_dir": project_root / entry["cache_dir"],
            "usage": entry["usage"],
        }
        for name, entry in raw["datasets"].items()
    }


# ── Logging ───────────────────────────────────────────────────────────────────

_HANDLER_ID: int | None = None


def get_logger(name: str, verbose: bool = False) -> _LoguruLogger:
    """Configure loguru et retourne un logger contextualisé avec le nom du script.

    Supprime le handler précédent pour éviter les doublons en cas d'appels multiples.
    Format : HH:MM:SS [script] LEVEL — message (colorisé sur TTY).

    Args:
        name: Nom du script (ex: "11_train_sft"), affiché entre crochets.
        verbose: Si True, active le niveau DEBUG (défaut : INFO).

    Returns:
        Logger loguru contextualisé prêt à l'emploi.
    """
    global _HANDLER_ID

    level = "DEBUG" if verbose else "INFO"

    # Supprimer le handler précédent (évite les lignes en double)
    if _HANDLER_ID is not None:
        try:
            _loguru_logger.remove(_HANDLER_ID)
        except ValueError:
            pass
    else:
        # Supprimer le handler stderr par défaut de loguru (id=0)
        try:
            _loguru_logger.remove(0)
        except ValueError:
            pass

    _HANDLER_ID = _loguru_logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> [<cyan>{extra[script]}</cyan>] <level>{level: <8}</level> — {message}",
        level=level,
        colorize=True,
    )

    return _loguru_logger.bind(script=name)
