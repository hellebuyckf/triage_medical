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
        eval_text = eval_text[:_MAX_EVAL_CHARS - 3] + "..."

    return f"{label}\n\nÉvaluation clinique : {eval_text}\n\nRecommandations : {reco}"


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
MEDICAL_TERMS_ALLOWLIST: frozenset[str] = frozenset({
    # Éponymes — maladies et syndromes (EN)
    "alzheimer", "parkinson", "huntington", "crohn", "hodgkin", "addison",
    "cushing", "graves", "hashimoto", "wilson", "marfan", "turner", "down",
    "raynaud", "behcet", "sjogren", "sjögren", "brugada", "wolff",
    "klinefelter", "noonan", "prader", "willi", "angelman", "rett",
    "duchenne", "becker", "charcot", "marie", "tooth", "gaucher", "fabry",
    "niemann", "pick", "pompe", "hurler", "hunter", "sanfilippo", "morquio",
    "tay", "sachs", "canavan", "krabbe", "batten", "spielmeyer", "vogt",
    "aicardi", "goutières", "aicardi-goutières", "lennox", "gastaut",
    "dravet", "landau", "kleffner", "sturge", "weber", "von hippel",
    "lindau", "neurofibromatosis", "tuberous", "sézary", "szary",
    "paget", "bowen", "kaposi", "burkitt", "wilms", "ewing", "pott",
    "bright", "berger", "henoch", "schönlein", "wegener", "goodpasture",
    "buerger", "takayasu", "horton", "kawasaki", "still", "felty",
    "reiter", "behçet", "whipple", "menetrier", "zollinger", "ellison",
    "sipple", "wermer", "verner", "morrison", "ogilvie", "hirschsprung",
    "meckel", "peutz", "jeghers", "lynch", "cowden", "bannayan",
    "riley", "day", "fanconi", "blackfan", "diamond", "shwachman",
    "kostmann", "chediak", "higashi", "wiskott", "aldrich", "bruton",
    "digeorge", "treacher", "collins", "pierre", "robin", "goldenhar",
    "charge", "vacter", "vacterl",
    # Éponymes — maladies et syndromes (FR)
    "alzheimer", "parkinson", "huntington", "crohn", "hodgkin",
    "basedow", "quincke", "biermer", "leriche", "osler", "rendu",
    "weber", "barre", "guillain", "millard", "gubler",
    # Procédures / examens
    "x-ray", "x ray", "mri", "ct scan", "ercp",
    # Organes / termes anatomiques mal détectés
    "lung", "heart", "kidney", "liver", "spleen", "colon",
    # Institutions médicales
    "nih", "cdc", "who", "nhlbi",
    # Médicaments courants (marques taguées comme personnes)
    "coversyl", "perindopril", "levothyrox", "doliprane", "aspirin",
    # Termes génétiques
    "glut", "brca", "cftr", "mthfr",
    # Abbréviations souvent mal taguées
    "arp", "ags",
    # Labels d'urgence du format triage — ne jamais anonymiser
    # Presidio/spaCy anglais détecte "URGENCE MODÉRÉE" comme entité PERSON (mot français = nom étranger)
    "urgence", "maximale", "modérée", "différée",
})


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
            detected = text[result.start:result.end].lower().strip()
            if detected in allowlist:
                continue
            # Vérifier si le terme contient un mot de l'allowlist
            if any(term in detected for term in allowlist):
                continue
        filtered.append(result)
    return filtered


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
