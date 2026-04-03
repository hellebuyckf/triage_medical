"""Configuration du serveur — lue depuis les variables d'environnement."""

from pydantic_settings import BaseSettings


class ServerConfig(BaseSettings):
    """Paramètres du serveur FastAPI + vLLM.

    Toutes les valeurs sont surchargeable via variables d'environnement
    (ex. ``MODEL_PATH=/model``, ``MAX_NEW_TOKENS=256``).

    Attributes:
        model_path: Chemin local ou identifiant HuggingFace Hub du modèle fusionné.
        max_model_len: Longueur maximale de séquence (prompt + réponse).
        dtype: Type de données pour les poids du modèle (``bfloat16`` recommandé).
        max_new_tokens: Nombre maximal de tokens à générer par requête.
        temperature: Température de génération. 0.1 = quasi-greedy (recommandé pour
            le triage médical — réduit les hallucinations). 1.0 = échantillonnage libre.
        port: Port d'écoute du serveur uvicorn.
        log_level: Niveau de log uvicorn (``info``, ``debug``, ``warning``).

    Example:
        >>> config = ServerConfig()
        >>> config.model_path
        'checkpoints/dpo_merged'
    """

    model_path: str = "checkpoints/dpo_merged"
    max_model_len: int = 1024
    dtype: str = "bfloat16"
    max_new_tokens: int = 512
    temperature: float = 0.1
    port: int = 8080
    log_level: str = "info"

    model_config = {"env_file": ".env", "extra": "ignore"}
