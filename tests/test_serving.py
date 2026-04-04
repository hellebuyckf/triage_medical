"""Tests unitaires pour l'API de triage médical — sans GPU (moteur vLLM mocké)."""

from __future__ import annotations

import sys
from pathlib import Path

# Ajouter le répertoire scripts au path pour résoudre 'from utils import ...'
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Mock vllm avant d'importer l'app
from unittest.mock import MagicMock, patch

mock_vllm = MagicMock()
sys.modules["vllm"] = mock_vllm

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# ── Fixtures ──────────────────────────────────────────────────────────────────

VALID_RESPONSE_TEXT = (
    "URGENCE MAXIMALE\n\n"
    "Évaluation clinique : Douleur thoracique avec sudation évoque un syndrome coronarien aigu.\n\n"
    "Recommandations : Appelez le 15 (SAMU) immédiatement. Ne restez pas seul."
)


def _make_mock_engine(response_text: str = VALID_RESPONSE_TEXT) -> MagicMock:
    """Crée un AsyncLLMEngine mocké qui retourne response_text en une itération."""
    mock_output = MagicMock()
    mock_output.finished = True
    mock_output.outputs = [MagicMock(text=response_text)]

    async def _fake_generate(prompt: str, sampling_params, request_id: str):
        yield mock_output

    mock_engine = MagicMock()
    mock_engine.generate = _fake_generate
    return mock_engine


def _make_mock_tokenizer() -> MagicMock:
    """Crée un AutoTokenizer mocké avec apply_chat_template fonctionnel."""
    mock_tok = MagicMock()
    mock_tok.chat_template = "fake_template"
    mock_tok.apply_chat_template.return_value = "<|im_start|>system\nSYSTEM\n<|im_end|>\n<|im_start|>user\nSYMPTOMS\n<|im_end|>\n<|im_start|>assistant\n"
    return mock_tok


@pytest_asyncio.fixture
async def client():
    """Client HTTP de test avec engine et tokenizer mockés (pas de GPU requis)."""
    import scripts.serving.app as app_module

    mock_engine = _make_mock_engine()
    mock_tokenizer = _make_mock_tokenizer()

    with (
        patch.object(app_module, "engine", mock_engine),
        patch.object(app_module, "tokenizer", mock_tokenizer),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app_module.app),
            base_url="http://test",
        ) as c:
            yield c


@pytest_asyncio.fixture
async def client_unparseable():
    """Client mocké dont le modèle retourne une réponse sans label d'urgence."""
    import scripts.serving.app as app_module

    mock_engine = _make_mock_engine(response_text="Je ne sais pas quoi dire.")
    mock_tokenizer = _make_mock_tokenizer()

    with (
        patch.object(app_module, "engine", mock_engine),
        patch.object(app_module, "tokenizer", mock_tokenizer),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app_module.app),
            base_url="http://test",
        ) as c:
            yield c


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient) -> None:
    """GET /health doit retourner 200 avec status='ok'."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model" in data


@pytest.mark.asyncio
async def test_triage_valid_request(client: AsyncClient) -> None:
    """POST /triage avec des symptômes valides doit retourner une TriageResponse complète."""
    response = await client.post(
        "/triage",
        json={"symptoms": "Douleur thoracique intense, sudation, nausées depuis 30 minutes."},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["urgency_level"] in ["max", "moderate", "deferred", None]
    assert data["urgency_label"] != ""
    assert data["raw_response"] != ""
    assert data["disclaimer"] != ""
    assert data["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_triage_urgency_max_detected(client: AsyncClient) -> None:
    """Le modèle mocké retourne 'URGENCE MAXIMALE' → urgency_level doit être 'max'."""
    response = await client.post(
        "/triage",
        json={"symptoms": "Douleur thoracique intense avec sudation profuse."},
    )
    assert response.status_code == 200
    assert response.json()["urgency_level"] == "max"
    assert response.json()["urgency_label"] == "URGENCE MAXIMALE"


@pytest.mark.asyncio
async def test_triage_symptoms_too_short(client: AsyncClient) -> None:
    """POST /triage avec symptoms < 10 caractères doit retourner 422."""
    response = await client.post("/triage", json={"symptoms": "mal"})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_triage_symptoms_too_long(client: AsyncClient) -> None:
    """POST /triage avec symptoms > 2000 caractères doit retourner 422."""
    response = await client.post("/triage", json={"symptoms": "x" * 2001})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_disclaimer_always_present(client: AsyncClient) -> None:
    """Le disclaimer médical doit être présent dans toute réponse."""
    response = await client.post(
        "/triage",
        json={"symptoms": "J'ai de la fièvre depuis deux jours et des maux de tête."},
    )
    assert response.status_code == 200
    assert response.json()["disclaimer"] != ""
    assert "triage" in response.json()["disclaimer"].lower()


@pytest.mark.asyncio
async def test_triage_unparseable_response(client_unparseable: AsyncClient) -> None:
    """Si le modèle ne génère pas de label d'urgence, urgency_level doit être None."""
    response = await client_unparseable.post(
        "/triage",
        json={"symptoms": "Symptômes difficiles à interpréter pour ce test."},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["urgency_level"] is None
    assert data["urgency_label"] == "URGENCE INDÉTERMINÉE"


@pytest.mark.asyncio
async def test_engine_not_loaded_returns_503() -> None:
    """Si le moteur vLLM n'est pas chargé, POST /triage doit retourner 503."""
    import scripts.serving.app as app_module

    with patch.object(app_module, "engine", None), patch.object(app_module, "tokenizer", None):
        async with AsyncClient(
            transport=ASGITransport(app=app_module.app),
            base_url="http://test",
        ) as c:
            response = await c.post(
                "/triage",
                json={"symptoms": "Douleur thoracique intense depuis 30 minutes."},
            )
    assert response.status_code == 503
