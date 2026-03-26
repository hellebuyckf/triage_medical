"""FastAPI application — agent de triage médical avec inférence vLLM."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer
from utils import SYSTEM_PROMPT, extract_urgency_from_response, get_logger

from serving.config import ServerConfig
from serving.models import TriageRequest, TriageResponse

if TYPE_CHECKING:
    from vllm import AsyncLLMEngine

# ── Constantes ────────────────────────────────────────────────────────────────

DISCLAIMER = "⚠️ Cet agent est un outil d'aide au triage, pas un diagnostic médical."

URGENCY_LABELS: dict[str, str] = {
    "max": "URGENCE MAXIMALE",
    "moderate": "URGENCE MODÉRÉE",
    "deferred": "URGENCE DIFFÉRÉE",
}

# Chat template ChatML standard pour Qwen3-Base (pas de template natif sur le modèle de base).
# Même pattern que scripts/training/12_evaluate_sft.py:126-133.
_QWEN3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)

# ── État global de l'application ──────────────────────────────────────────────

config = ServerConfig()
logger = get_logger("serving")

engine: AsyncLLMEngine | None = None
tokenizer: AutoTokenizer | None = None  # type: ignore[type-arg]


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Charge le moteur vLLM et le tokenizer au démarrage, libère à l'arrêt.

    Args:
        app: Instance FastAPI (injectée par le framework).

    Yields:
        Rien — cède la main à l'application pendant sa durée de vie.
    """
    from vllm import AsyncEngineArgs, AsyncLLMEngine  # import tardif — vLLM lourd à charger

    global engine, tokenizer

    logger.info("Chargement du tokenizer depuis {}...", config.model_path)
    tok = AutoTokenizer.from_pretrained(config.model_path)
    if not tok.chat_template:
        tok.chat_template = _QWEN3_CHAT_TEMPLATE
    tokenizer = tok

    logger.info(
        "Chargement du moteur vLLM (dtype={}, max_model_len={})...",
        config.dtype,
        config.max_model_len,
    )
    engine_args = AsyncEngineArgs(
        model=config.model_path,
        dtype=config.dtype,
        max_model_len=config.max_model_len,
        trust_remote_code=False,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    logger.info("Moteur vLLM prêt.")

    yield

    logger.info("Arrêt du moteur vLLM.")
    engine = None
    tokenizer = None


# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="CHSA Triage API",
    description=(
        "Agent de triage médical — Centre Hospitalier Saint-Aurélien.\n\n"
        "**⚠️ Outil d'aide au triage uniquement — pas un diagnostic médical.**"
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health", summary="Health check")
async def health() -> dict[str, str]:
    """Vérifie que le serveur est opérationnel.

    Returns:
        Dict avec ``status`` (``"ok"`` ou ``"loading"``) et le chemin du modèle.
    """
    status = "ok" if engine is not None else "loading"
    return {"status": status, "model": config.model_path}


@app.post("/triage", response_model=TriageResponse, summary="Triage médical")
async def triage(request: TriageRequest) -> TriageResponse:
    """Analyse les symptômes et retourne un niveau d'urgence médical.

    Le modèle génère une réponse structurée avec :
    - Le niveau d'urgence (MAXIMALE / MODÉRÉE / DIFFÉRÉE)
    - Une évaluation clinique brève
    - Des recommandations concrètes

    Args:
        request: Corps JSON avec le champ ``symptoms``.

    Returns:
        TriageResponse avec le niveau d'urgence parsé et la réponse brute.

    Raises:
        HTTPException: 503 si le moteur vLLM n'est pas encore chargé.
    """
    if engine is None or tokenizer is None:
        raise HTTPException(
            status_code=503, detail="Moteur vLLM non chargé — réessayer dans quelques secondes."
        )

    from vllm import SamplingParams

    t0 = time.monotonic()

    # Formatage du prompt ChatML + suppression du <think> Qwen3.
    # Pattern identique à scripts/training/12_evaluate_sft.py:187-199.
    prompt = (
        str(
            tokenizer.apply_chat_template(  # type: ignore[union-attr]
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": request.symptoms},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        + "<think>\n\n</think>\n"
    )

    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_new_tokens,
        stop=["<|im_end|>"],
    )

    # Génération asynchrone — itère jusqu'au dernier output (finished=True)
    request_id = str(uuid4())
    raw_response = ""
    async for output in engine.generate(prompt, sampling_params, request_id):  # type: ignore[union-attr]
        if output.finished:
            raw_response = output.outputs[0].text

    latency_ms = (time.monotonic() - t0) * 1000

    urgency_level = extract_urgency_from_response(raw_response)
    urgency_label = URGENCY_LABELS.get(urgency_level or "", "URGENCE INDÉTERMINÉE")

    logger.info(
        "triage | urgency={} | latency={:.0f}ms | symptoms_len={}",
        urgency_level,
        latency_ms,
        len(request.symptoms),
    )

    return TriageResponse(
        urgency_level=urgency_level,
        urgency_label=urgency_label,
        raw_response=raw_response,
        disclaimer=DISCLAIMER,
        model=config.model_path,
        latency_ms=round(latency_ms, 1),
    )
