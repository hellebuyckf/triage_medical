"""FastAPI application — agent de triage médical avec inférence vLLM."""

from __future__ import annotations

import re
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from utils import SYSTEM_PROMPT, extract_urgency_from_response, get_logger

from serving.config import ServerConfig
from serving.models import TriageRequest, TriageResponse

if TYPE_CHECKING:
    from transformers import AutoTokenizer  # type: ignore[type-arg]
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

# Tokens d'anonymisation Presidio — le modèle les a appris sur le dataset SFT
# et peut les reproduire à l'inférence. On les supprime proprement en post-processing.
_PRESIDIO_TOKEN_RE = re.compile(r"<[A-Z_]{2,}>")


def _clean_response(text: str) -> str:
    """Supprime les tokens d'anonymisation Presidio résiduels de la réponse.

    Le modèle a été entraîné sur des données anonymisées contenant des tokens
    comme ``<PERSON>``, ``<LOCATION>``, ``<DATE_TIME>``, etc. Il peut les
    reproduire à l'inférence. Cette fonction les retire pour éviter de les
    exposer à l'utilisateur final.

    Args:
        text: Réponse brute générée par le modèle.

    Returns:
        Réponse nettoyée, sans tokens Presidio.
    """
    return _PRESIDIO_TOKEN_RE.sub("", text).strip()


# ── État global de l'application ──────────────────────────────────────────────

config = ServerConfig()
logger = get_logger("serving")

engine: AsyncLLMEngine | None = None
tokenizer: AutoTokenizer | None = None  # type: ignore[type-arg]
http_client: httpx.AsyncClient | None = None


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Charge le moteur vLLM (local) ou le client HTTP (gateway) au démarrage.

    Args:
        app: Instance FastAPI (injectée par le framework).

    Yields:
        Rien — cède la main à l'application pendant sa durée de vie.
    """
    global engine, tokenizer, http_client

    if config.vllm_api_base_url:
        # ── Mode GATEWAY (Cloud Run -> Compute Engine vLLM) ──
        base_url = config.vllm_api_base_url
        if not base_url.endswith("/"):
            base_url += "/"

        logger.info("Mode GATEWAY activé (vLLM distant: {})", base_url)
        http_client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {config.vllm_api_key}"}
            if config.vllm_api_key
            else {},
            timeout=60.0,
        )
        # Tokenizer optionnel pour le gateway (utilisé pour apply_chat_template si besoin)
        # mais on peut s'en passer si on utilise l'API /chat/completions de vLLM.
    else:
        # ── Mode LOCAL (vLLM embarqué) ──
        from transformers import AutoTokenizer  # import tardif
        from vllm import AsyncEngineArgs, AsyncLLMEngine  # import tardif — vLLM lourd

        logger.info("Mode LOCAL activé (vLLM embarqué)")
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

    if http_client:
        await http_client.aclose()
        http_client = None

    logger.info("Arrêt du serveur.")
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
        Dict avec ``status`` (``"ok"`` ou ``"loading"``) et le mode (local/gateway).
    """
    if config.vllm_api_base_url:
        status = "ok" if http_client is not None else "error"
        mode = "gateway"
        target = config.vllm_api_base_url
    else:
        status = "ok" if engine is not None else "loading"
        mode = "local"
        target = config.model_path

    return {"status": status, "mode": mode, "target": target}


@app.post("/triage", response_model=TriageResponse, summary="Triage médical")
async def triage(request: TriageRequest) -> TriageResponse:
    """Analyse les symptômes et retourne un niveau d'urgence médical.

    Args:
        request: Corps JSON avec le champ ``symptoms``.

    Returns:
        TriageResponse avec le niveau d'urgence parsé et la réponse brute.
    """
    t0 = time.monotonic()
    raw_response = ""

    if config.vllm_api_base_url:
        # ── Inférence via GATEWAY (vLLM OpenAI API) ──
        if http_client is None:
            raise HTTPException(status_code=503, detail="Client Gateway non initialisé.")

        try:
            # vLLM-OpenAI supporte /v1/chat/completions
            resp = await http_client.post(
                "chat/completions",  # Chemin relatif
                json={
                    "model": config.model_path,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": request.symptoms},
                    ],
                    "temperature": config.temperature,
                    "max_tokens": config.max_new_tokens,
                    "stop": ["<|im_end|>"],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            raw_response = data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error("Erreur Gateway: {}", e)
            raise HTTPException(
                status_code=502, detail=f"Erreur de communication avec vLLM: {e!s}"
            ) from e

    else:
        # ── Inférence via LOCAL vLLM ──
        if engine is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Moteur vLLM non chargé.")

        from vllm import SamplingParams

        # Formatage du prompt ChatML + suppression du <think> Qwen3.
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

        request_id = str(uuid4())
        async for output in engine.generate(prompt, sampling_params, request_id):  # type: ignore[union-attr]
            if output.finished:
                raw_response = output.outputs[0].text

    latency_ms = (time.monotonic() - t0) * 1000

    # Post-processing commun
    raw_response = _clean_response(raw_response)
    urgency_level = extract_urgency_from_response(raw_response)
    urgency_label = URGENCY_LABELS.get(urgency_level or "", "URGENCE INDÉTERMINÉE")

    logger.info(
        "triage | mode={} | urgency={} | latency={:.0f}ms",
        "gateway" if config.vllm_api_base_url else "local",
        urgency_level,
        latency_ms,
    )

    return TriageResponse(
        urgency_level=urgency_level,
        urgency_label=urgency_label,
        raw_response=raw_response,
        disclaimer=DISCLAIMER,
        model=config.model_path,
        latency_ms=round(latency_ms, 1),
    )
