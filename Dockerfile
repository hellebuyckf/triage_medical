# Image API — Agent de triage médical (FastAPI + vLLM)
#
# Base : vllm/vllm-openai:latest — contient CUDA, PyTorch et vLLM pré-installés.
# On y ajoute uniquement les dépendances FastAPI légères.
#
# Build  : docker build -t triage-api:latest .
# Run    : docker compose up   (voir docker-compose.yml)
# Local  : MODEL_PATH=checkpoints/dpo_merged uvicorn scripts.serving.app:app --port 8080

FROM vllm/vllm-openai:v0.4.2

# Dépendances FastAPI — vLLM + transformers déjà présents dans l'image de base
RUN pip install --no-cache-dir \
    fastapi>=0.111 \
    "uvicorn[standard]>=0.29" \
    "pydantic-settings>=2.2" \
    httpx>=0.27 \
    loguru>=0.7 \
    pyyaml>=6.0

WORKDIR /app

# Code applicatif — utils.py partagé + module serving
COPY scripts/utils.py /app/scripts/utils.py
COPY scripts/serving/ /app/scripts/serving/

# PYTHONPATH expose scripts/ pour que `from utils import ...` fonctionne
ENV PYTHONPATH=/app/scripts
ENV PORT=8080

EXPOSE 8080

# L'image vllm/vllm-openai définit ENTRYPOINT ["vllm"] — on le réinitialise
# pour pouvoir lancer uvicorn directement.
ENTRYPOINT []
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8080"]
