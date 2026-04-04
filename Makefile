.PHONY: all setup lint test test-serving download build-sft build-dpo anonymize split \
        prepare-tokenizer train-sft evaluate-sft sft-pipeline \
        sft-errors rebuild-dpo \
        dpo-pipeline dpo-pipeline-hard train-dpo evaluate-dpo export-model push-model \
        push-datasets push-datasets-all \
        build-api serve-local serve-down serve-restart api-health api-triage \
        alpha-health alpha-triage alpha-url benchmark \
        clean clean-sft clean-dpo clean-all retrain help

# Variables
-include .env
export

PYTHON          = uv run python
DATA_PREP       = scripts/data_prep
TRAINING        = scripts/training

# Environnement cible : dev (défaut) ou demo
# Usage : make train-sft ENV=demo   → logs vers MLflow GCP (Cloud Run)
#         make train-sft            → logs vers MLflow local (SQLite)
# Prérequis pour ENV=demo : gcloud auth login (une seule fois)
ENV             ?= dev
MLFLOW_TRACKING_URI := $(shell \
    grep '^MLFLOW_TRACKING_URI=' .env.$(ENV) 2>/dev/null | head -1 | cut -d'=' -f2-)

# Variables d'env injectées dans les cibles d'entraînement.
# demo : ajoute un Identity Token Google Cloud (expire après 1h, régénéré automatiquement).
ifeq ($(ENV),demo)
_MLFLOW_ENVVARS = MLFLOW_TRACKING_URI="$(MLFLOW_TRACKING_URI)" MLFLOW_TRACKING_TOKEN="$$(gcloud auth print-identity-token)"
# En mode demo, google-cloud-storage est requis pour les artefacts GCS.
# setup-gcp est ajouté comme prérequis automatique des cibles d'entraînement/évaluation.
_GCP_PREREQ     = setup-gcp
else
_MLFLOW_ENVVARS = MLFLOW_TRACKING_URI="$(MLFLOW_TRACKING_URI)"
_GCP_PREREQ     =
endif

# Training config YAML
# Usage : make train-sft SFT_CONFIG=configs/sft_fast.yaml
SFT_CONFIG      ?= configs/sft.yaml
DPO_CONFIG      ?= configs/dpo.yaml

# HuggingFace Hub
# Usage : make push-datasets HF_USERNAME=johndoe
# Ou    : export HF_USERNAME=johndoe && make push-datasets
HF_USERNAME     ?= $(shell echo $$HF_USERNAME)
HF_PRIVATE      ?= 0
_HF_PRIVATE_FLAG = $(if $(filter 1,$(HF_PRIVATE)),--private,)

# Evaluation options
# Set EVAL_VAL=1 to also evaluate on the val set (biased — model was selected on val loss).
# Example: make evaluate-sft EVAL_VAL=1
EVAL_VAL        ?= 0
_EVAL_VAL_FLAG  = $(if $(filter 1,$(EVAL_VAL)),--eval-val,)

# IP Tailscale du serveur de calcul — accessible directement depuis le Mac M3.
# Pour trouver l'IP : tailscale ip -4
ALPHA_HOST      ?= 100.115.15.123

# Benchmark
BENCH_URL       ?= http://localhost:8080
BENCH_N         ?= 20
BENCH_C         ?= 5
BENCH_P95       ?= 5000

# Cible par défaut
.DEFAULT_GOAL := help

# ── Setup ─────────────────────────────────────────────────────────────────────

setup:
	uv sync --extra dev
	uv pip install https://github.com/explosion/spacy-models/releases/download/fr_core_news_md-3.8.0/fr_core_news_md-3.8.0-py3-none-any.whl
	uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl
	uv run pre-commit install

# ── Qualité du code ───────────────────────────────────────────────────────────

lint:
	uv run ruff check scripts/
	uv run ruff format --check scripts/
	uv run pyright scripts/

test:
	$(PYTHON) -m pytest tests/

test-serving:
	$(PYTHON) -m pytest tests/test_serving.py

# ── Data Engineering ──────────────────────────────────────────────────────────
data-pipeline: download build-sft build-dpo anonymize split

download:
	$(PYTHON) $(DATA_PREP)/01_download.py

build-sft: download
	$(PYTHON) $(DATA_PREP)/02_build_sft.py

build-dpo: download
	$(PYTHON) $(DATA_PREP)/03_build_dpo.py

anonymize: build-sft build-dpo
	$(PYTHON) $(DATA_PREP)/04_anonymize.py

split: anonymize
	$(PYTHON) $(DATA_PREP)/05_split_and_validate.py

# ── GCP extras (requis uniquement pour ENV=demo) ──────────────────────────────

# Installe google-cloud-storage pour les artefacts GCS de MLflow.
# Utilise uv pip install (additif) et non uv sync, pour ne pas désinstaller
# les packages hors-lockfile comme unsloth.
setup-gcp:
	uv pip install "google-cloud-storage>=2.16"

# ── SFT ───────────────────────────────────────────────────────────────────────

sft-pipeline: prepare-tokenizer train-sft evaluate-sft

prepare-tokenizer:
	$(PYTHON) $(TRAINING)/10_prepare_tokenizer.py

train-sft: $(_GCP_PREREQ) prepare-tokenizer
	$(_MLFLOW_ENVVARS) $(PYTHON) $(TRAINING)/11_train_sft.py --config $(SFT_CONFIG)

evaluate-sft: $(_GCP_PREREQ) train-sft
	$(_MLFLOW_ENVVARS) $(PYTHON) $(TRAINING)/12_evaluate_sft.py $(_EVAL_VAL_FLAG)

# ── DPO ───────────────────────────────────────────────────────────────────────

# Generate hard-negative DPO pairs from SFT misclassifications.
# Requires: checkpoints/sft (run sft-pipeline first).
sft-errors: train-sft
	$(PYTHON) $(DATA_PREP)/03b_sft_errors.py

# Rebuild the DPO raw dataset (picks up hard negatives if present) and
# re-run anonymize + split to propagate changes to data/final/dpo.
rebuild-dpo: sft-errors
	rm -rf data/processed/dpo_raw data/processed/sft_anonymized data/processed/dpo_anonymized data/final/dpo
	$(PYTHON) $(DATA_PREP)/03_build_dpo.py
	$(PYTHON) $(DATA_PREP)/04_anonymize.py
	$(PYTHON) $(DATA_PREP)/05_split_and_validate.py

# Full DPO pipeline with hard negatives: generate errors → rebuild data → train → eval → export.
dpo-pipeline-hard: rebuild-dpo clean-dpo dpo-pipeline

dpo-pipeline: train-dpo evaluate-dpo export-model

train-dpo: $(_GCP_PREREQ)
	$(_MLFLOW_ENVVARS) $(PYTHON) $(TRAINING)/20_train_dpo.py --config $(DPO_CONFIG)

evaluate-dpo: $(_GCP_PREREQ) train-dpo
	$(_MLFLOW_ENVVARS) $(PYTHON) $(TRAINING)/21_evaluate_dpo.py $(_EVAL_VAL_FLAG)

export-model: evaluate-dpo
	$(PYTHON) $(TRAINING)/22_export_model.py --skip-verify

push-model: export-model
	@if [ -z "$(HF_USERNAME)" ]; then \
		echo "Erreur : HF_USERNAME non défini."; \
		echo "Usage  : make push-model HF_USERNAME=<votre_username>"; \
		exit 1; \
	fi
	$(PYTHON) $(TRAINING)/22_export_model.py \
		--push-to-hub \
		--repo-id $(HF_USERNAME)/qwen3-triage-dpo \
		--skip-verify

upload-model:
	@if [ -z "$(HF_USERNAME)" ]; then \
		echo "Erreur : HF_USERNAME non défini."; \
		echo "Usage  : make upload-model HF_USERNAME=<votre_username>"; \
		exit 1; \
	fi
	uv run huggingface-cli upload $(HF_USERNAME)/qwen3-triage-dpo ./checkpoints/dpo_merged/

# ── HuggingFace Hub ───────────────────────────────────────────────────────────

push-datasets: split
	@if [ -z "$(HF_USERNAME)" ]; then \
		echo "Erreur : HF_USERNAME non défini."; \
		echo "Usage  : make push-datasets HF_USERNAME=<votre_username>"; \
		exit 1; \
	fi
	$(PYTHON) $(DATA_PREP)/06_push_to_hub.py \
		--username $(HF_USERNAME) \
		$(_HF_PRIVATE_FLAG)

push-datasets-all: split
	@if [ -z "$(HF_USERNAME)" ]; then \
		echo "Erreur : HF_USERNAME non défini."; \
		echo "Usage  : make push-datasets-all HF_USERNAME=<votre_username>"; \
		exit 1; \
	fi
	$(PYTHON) $(DATA_PREP)/06_push_to_hub.py \
		--username $(HF_USERNAME) \
		--include-processed \
		$(_HF_PRIVATE_FLAG)

# ── API (FastAPI + vLLM) ──────────────────────────────────────────────────────
#
# Prérequis : checkpoints/dpo_merged/ doit exister (make export-model)
#
# Accès local  : http://localhost:8080/docs
# Accès réseau : http://$(ALPHA_HOST):8080/docs  (via Tailscale, sans tunnel SSH)

build-api:
	docker build -t triage-api:latest .

serve-local:
	docker compose up --build

serve-down:
	docker compose down

serve-restart:
	docker compose down && docker compose up --build

api-health:
	curl -s http://localhost:8080/health | python3 -m json.tool

api-triage:
	curl -s -X POST http://localhost:8080/triage \
	  -H "Content-Type: application/json" \
	  -d '{"symptoms": "Douleur thoracique intense, sudation, nausées depuis 30 minutes."}' \
	  | python3 -m json.tool

# Cibles alpha — interroge le serveur directement via Tailscale (sans tunnel SSH)
alpha-health:
	curl -s http://$(ALPHA_HOST):8080/health | python3 -m json.tool

alpha-triage:
	curl -s -X POST http://$(ALPHA_HOST):8080/triage \
	  -H "Content-Type: application/json" \
	  -d '{"symptoms": "Douleur thoracique intense, sudation, nausées depuis 30 minutes."}' \
	  | python3 -m json.tool

alpha-url:
	@echo "API Triage CHSA (alpha) :"
	@echo "  Docs    → http://$(ALPHA_HOST):8080/docs"
	@echo "  Health  → http://$(ALPHA_HOST):8080/health"
	@echo "  Triage  → POST http://$(ALPHA_HOST):8080/triage"

benchmark:
	$(PYTHON) scripts/serving/benchmark.py \
	  --url $(BENCH_URL) \
	  --n-requests $(BENCH_N) \
	  --concurrency $(BENCH_C) \
	  --p95-max-ms $(BENCH_P95)

# ── Nettoyage ─────────────────────────────────────────────────────────────────

clean:
	rm -rf data/raw data/processed

clean-sft:
	rm -rf checkpoints/sft data/processed/sft_tokenized

clean-dpo:
	rm -rf checkpoints/dpo checkpoints/dpo_merged

clean-all:
	rm -rf data/raw data/processed data/final checkpoints

retrain:
	@echo "=== Nettoyage des checkpoints SFT et DPO ==="
	$(MAKE) clean-sft clean-dpo
	@echo "=== Pipeline SFT (tokenize → train → eval) ==="
	$(MAKE) sft-pipeline
	@echo "=== Pipeline DPO (train → eval → export) ==="
	$(MAKE) dpo-pipeline

# ── Aide ──────────────────────────────────────────────────────────────────────

help:
	@echo "Cibles disponibles :"
	@echo ""
	@echo "  Setup"
	@echo "  make setup             — installe les dépendances et modèles spaCy"
	@echo ""
	@echo "  Qualité du code"
	@echo "  make lint              — ruff (linter + format) + pyright (typage)"
	@echo "  make test              — lance tous les tests unitaires (pytest)"
	@echo "  make test-serving      — lance les tests de l'API (vLLM mocké)"
	@echo ""
	@echo "  Data Engineering"
	@echo "  make data-pipeline     — pipeline complet data (download → split)"
	@echo "  make download          — télécharge les datasets HuggingFace"
	@echo "  make build-sft         — construit le dataset SFT"
	@echo "  make build-dpo         — construit le dataset DPO"
	@echo "  make anonymize         — anonymisation RGPD + rapport"
	@echo "  make split             — split train/val/test + validation"
	@echo ""
	@echo "  Environnements"
	@echo "  ENV=dev  (défaut)  — MLflow local SQLite (alpha-server)"
	@echo "  ENV=demo           — MLflow GCP Cloud Run (lit .env.demo)"
	@echo "  Exemple : make train-sft ENV=demo"
	@echo ""
	@echo "  SFT"
	@echo "  make sft-pipeline      — pipeline complet SFT (tokenize → train → eval)"
	@echo "  make prepare-tokenizer — tokenisation + formatage ChatML"
	@echo "  make train-sft         — entraînement SFT LoRA (config: configs/sft.yaml)"
	@echo "  make train-sft SFT_CONFIG=configs/sft_fast.yaml  — config alternative"
	@echo "  make evaluate-sft      — évaluation sur test set (honnête)"
	@echo "  make evaluate-sft EVAL_VAL=1  — idem + val set (biaisé, désactivé par défaut)"
	@echo ""
	@echo "  DPO"
	@echo "  make dpo-pipeline      — pipeline complet DPO (train → eval → export)"
	@echo "  make train-dpo         — alignement DPO LoRA (config: configs/dpo.yaml)"
	@echo "  make train-dpo DPO_CONFIG=configs/dpo_fast.yaml  — config alternative"
	@echo "  make evaluate-dpo      — évaluation SFT vs DPO sur test set (honnête)"
	@echo "  make evaluate-dpo EVAL_VAL=1  — idem + val set (biaisé, désactivé par défaut)"
	@echo "  make export-model      — fusion LoRA SFT+DPO → checkpoints/dpo_merged/"
	@echo "  make push-model HF_USERNAME=<user>   — export + push modèle fusionné vers HF Hub"
	@echo ""
	@echo "  HuggingFace Hub"
	@echo "  make push-model HF_USERNAME=<user>         — push modèle fusionné (username/qwen3-triage-dpo)"
	@echo "  make upload-model HF_USERNAME=<user>       — upload le modèle déjà fusionné (compatible Mac)"
	@echo "  make push-datasets HF_USERNAME=<user>      — publie sft + dpo finaux (DatasetDict)"
	@echo "  make push-datasets-all HF_USERNAME=<user>  — idem + datasets intermédiaires"
	@echo "  make push-datasets HF_USERNAME=<user> HF_PRIVATE=1  — dépôts privés"
	@echo ""
	@echo "  API (FastAPI + vLLM)"
	@echo "  make build-api         — construit l'image Docker API"
	@echo "  make serve-local       — démarre l'API en local (docker compose, port 8080)"
	@echo "  make serve-down        — arrête l'API (docker compose down)"
	@echo "  make serve-restart     — redémarre l'API (down + build + up)"
	@echo "  make api-health        — vérifie que l'API répond (GET /health)"
	@echo "  make api-triage        — test rapide de l'endpoint POST /triage"
	@echo "  make alpha-health      — health check via Tailscale ($(ALPHA_HOST))"
	@echo "  make alpha-triage      — test /triage via Tailscale"
	@echo "  make alpha-url         — affiche les URLs alpha (Tailscale)"
	@echo "  make benchmark         — benchmark latence (20 req séq. + 5 conc., SLA P95 ≤ 5 s)"
	@echo "  make benchmark BENCH_N=30 BENCH_C=8 BENCH_P95=3000  — paramètres personnalisés"
	@echo "  make benchmark BENCH_URL=http://$(ALPHA_HOST):8080   — benchmark via Tailscale"
	@echo ""
	@echo "  Nettoyage"
	@echo "  make clean             — supprime raw/ et processed/"
	@echo "  make clean-sft         — supprime checkpoints/sft et sft_tokenized/"
	@echo "  make clean-dpo         — supprime checkpoints/dpo et dpo_merged/"
	@echo "  make clean-all         — supprime tout data/ et checkpoints/"
	@echo "  make retrain           — clean SFT+DPO puis relance le pipeline complet (sans data)"
	@echo ""
	@echo "  Infrastructure → cd infra && make help"
