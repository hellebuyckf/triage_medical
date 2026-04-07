"""Schémas Pydantic pour l'API de triage médical."""

from pydantic import BaseModel, Field


class TriageRequest(BaseModel):
    """Corps de la requête POST /triage.

    Attributes:
        symptoms: Description libre des symptômes du patient.

    Example:
        >>> req = TriageRequest(symptoms="Douleur thoracique intense depuis 30 min")
    """

    symptoms: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Description des symptômes du patient (10–2000 caractères).",
        examples=["Douleur thoracique intense, sudation, nausées depuis 30 minutes."],
    )


class TriageResponse(BaseModel):
    """Corps de la réponse POST /triage.

    Attributes:
        urgency_level: Niveau d'urgence parsé (``"max"``, ``"moderate"``, ``"deferred"``
            ou ``None`` si non parseable).
        urgency_label: Label humain correspondant (``"URGENCE MAXIMALE"`` etc.).
        raw_response: Réponse brute générée par le modèle.
        disclaimer: Avertissement médical obligatoire.
        model: Identifiant du modèle utilisé pour la génération.
        latency_ms: Temps de génération en millisecondes.

    Example:
        >>> resp = TriageResponse(
        ...     urgency_level="max",
        ...     urgency_label="URGENCE MAXIMALE",
        ...     raw_response="URGENCE MAXIMALE\\n\\nÉvaluation...",
        ...     disclaimer="⚠️ ...",
        ...     model="checkpoints/dpo_merged",
        ...     latency_ms=850.3,
        ... )
    """

    urgency_level: str | None
    urgency_label: str
    raw_response: str
    disclaimer: str
    model: str
    latency_ms: float
