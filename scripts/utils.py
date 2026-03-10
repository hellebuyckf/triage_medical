"""Module partagé — constantes, filtres qualité, inférence d'urgence, logging."""

import hashlib
import logging
import re
from pathlib import Path

# ── Schémas de données ───────────────────────────────────────────────────────

SFT_COLUMNS = ["instruction", "response", "source", "language", "urgency_level", "confidence"]
DPO_COLUMNS = ["prompt", "chosen", "rejected", "source", "language"]
URGENCY_LEVELS = ["max", "moderate", "deferred"]

# ── Mots-clés d'urgence (bilingues) ──────────────────────────────────────────

URGENCY_MAX_KEYWORDS = [
    # FR
    "douleur thoracique", "difficultés respiratoires", "perte de conscience",
    "arrêt cardiaque", "hémorragie", "anaphylaxie", "AVC", "urgence vitale",
    "danger vital", "convulsions", "coma", "détresse respiratoire",
    # EN
    "chest pain", "difficulty breathing", "loss of consciousness",
    "cardiac arrest", "hemorrhage", "anaphylaxis", "stroke",
    "life-threatening", "emergency", "seizure", "unconscious",
]

URGENCY_DEFERRED_KEYWORDS = [
    # FR
    "rhume", "légère douleur", "fatigue chronique", "médecin traitant",
    "rendez-vous", "peut attendre", "suivi régulier", "vaccin",
    "prévention", "dépistage", "bilan", "contrôle", "consultation",
    "vitamines", "nutrition", "hygiène", "allergie saisonnière",
    "eczéma", "acné", "insomnie", "constipation", "régime",
    # EN
    "cold", "mild pain", "chronic fatigue", "general practitioner",
    "appointment", "can wait", "routine", "follow-up", "vaccine",
    "prevention", "screening", "check-up", "annual", "wellness",
    "vitamins", "nutrition", "hygiene", "seasonal allergy",
    "eczema", "acne", "insomnia", "constipation", "diet",
    "supplement", "lifestyle", "rehabilitation", "physical therapy",
    "counseling", "medication management", "refill",
]

# Pré-compilation des patterns pour la performance
_MAX_PATTERN = re.compile(
    "|".join(re.escape(kw) for kw in URGENCY_MAX_KEYWORDS), re.IGNORECASE
)
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


# ── Prompt formatting (S2 — SFT) ─────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un agent de triage médical pour le Centre Hospitalier Saint-Aurélien.
Analyse les symptômes décrits et fournis :
1. Le niveau d'urgence : URGENCE MAXIMALE / URGENCE MODÉRÉE / URGENCE DIFFÉRÉE
2. Une évaluation clinique brève
3. Des recommandations concrètes

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

    Cherche les patterns :
    - "URGENCE MAXIMALE" / "maximale" / "urgence max" → "max"
    - "URGENCE MODÉRÉE" / "modérée" / "urgence mod" → "moderate"
    - "URGENCE DIFFÉRÉE" / "différée" / "deferred" → "deferred"

    Args:
        text: Le texte de la réponse générée.

    Returns:
        "max", "moderate", "deferred", ou None si non trouvé.
    """
    for level, pattern in _URGENCY_PATTERNS.items():
        if pattern.search(text):
            return level
    return None


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


# ── Logging ───────────────────────────────────────────────────────────────────

def get_logger(name: str, verbose: bool = False) -> logging.Logger:
    """Logger formaté avec timestamp et nom du script."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s — %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    return logger
