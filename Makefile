.PHONY: all setup download build-sft build-dpo anonymize split \
        prepare-tokenizer train-sft evaluate-sft sft-pipeline \
        dpo-pipeline train-dpo evaluate-dpo export-model \
        mlflow mlflow-build mlflow-up mlflow-down mlflow-logs clean-mlflow \
        clean clean-sft clean-dpo clean-all help

# Variables
PYTHON          = uv run python
DATA_PREP       = scripts/data_prep
TRAINING        = scripts/training
MLFLOW_IMAGE    = project14-mlflow
MLFLOW_CONTAINER = project14-mlflow

# Evaluation options
# Set EVAL_VAL=1 to also evaluate on the val set (biased — model was selected on val loss).
# Example: make evaluate-sft EVAL_VAL=1
EVAL_VAL        ?= 0
_EVAL_VAL_FLAG  = $(if $(filter 1,$(EVAL_VAL)),--eval-val,)

# Cible par défaut
.DEFAULT_GOAL := help

# ── Setup ─────────────────────────────────────────────────────────────────────

setup:
	uv sync
	uv pip install https://github.com/explosion/spacy-models/releases/download/fr_core_news_md-3.8.0/fr_core_news_md-3.8.0-py3-none-any.whl
	uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl

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

# ── SFT ───────────────────────────────────────────────────────────────────────

sft-pipeline: prepare-tokenizer train-sft evaluate-sft

prepare-tokenizer:
	$(PYTHON) $(TRAINING)/10_prepare_tokenizer.py

train-sft: prepare-tokenizer
	$(PYTHON) $(TRAINING)/11_train_sft.py

evaluate-sft: train-sft
	$(PYTHON) $(TRAINING)/12_evaluate_sft.py $(_EVAL_VAL_FLAG)

# ── DPO ───────────────────────────────────────────────────────────────────────

dpo-pipeline: train-dpo evaluate-dpo export-model

train-dpo:
	$(PYTHON) $(TRAINING)/20_train_dpo.py

evaluate-dpo: train-dpo
	$(PYTHON) $(TRAINING)/21_evaluate_dpo.py $(_EVAL_VAL_FLAG)

export-model: evaluate-dpo
	$(PYTHON) $(TRAINING)/22_export_model.py

# ── MLflow ────────────────────────────────────────────────────────────────────
#
# Accès depuis le Mac M3 via tunnel SSH :
#   ssh -L 5000:localhost:5000 <user>@<ip_serveur>
# puis ouvrir http://localhost:5000 dans le navigateur.

mlflow: mlflow-build mlflow-up

mlflow-build:
	docker build -t $(MLFLOW_IMAGE) -f docker/mlflow/Dockerfile .

mlflow-up:
	docker run -d \
		--name $(MLFLOW_CONTAINER) \
		-p 127.0.0.1:5000:5000 \
		-v $(PWD)/mlflow.db:/mlflow.db \
		-v $(PWD)/mlruns:/mlruns \
		--restart unless-stopped \
		$(MLFLOW_IMAGE)
	@echo "MLflow UI démarré → tunnel SSH : ssh -L 5000:localhost:5000 <user>@<ip_serveur>"

mlflow-down:
	docker stop $(MLFLOW_CONTAINER) && docker rm $(MLFLOW_CONTAINER)

mlflow-logs:
	docker logs -f $(MLFLOW_CONTAINER)

clean-mlflow:
	rm -rf mlruns/ mlflow.db

# ── Nettoyage ─────────────────────────────────────────────────────────────────

clean:
	rm -rf data/raw data/processed

clean-sft:
	rm -rf checkpoints/sft data/processed/sft_tokenized

clean-dpo:
	rm -rf checkpoints/dpo checkpoints/dpo_merged

clean-all:
	rm -rf data/raw data/processed data/final checkpoints

# ── Aide ──────────────────────────────────────────────────────────────────────

help:
	@echo "Cibles disponibles :"
	@echo ""
	@echo "  Setup"
	@echo "  make setup             — installe les dépendances et modèles spaCy"
	@echo ""
	@echo "  Data Engineering"
	@echo "  make data-pipeline     — pipeline complet data (download → split)"
	@echo "  make download          — télécharge les datasets HuggingFace"
	@echo "  make build-sft         — construit le dataset SFT"
	@echo "  make build-dpo         — construit le dataset DPO"
	@echo "  make anonymize         — anonymisation RGPD + rapport"
	@echo "  make split             — split train/val/test + validation"
	@echo ""
	@echo "  SFT"
	@echo "  make sft-pipeline      — pipeline complet SFT (tokenize → train → eval)"
	@echo "  make prepare-tokenizer — tokenisation + formatage ChatML"
	@echo "  make train-sft         — entraînement SFT LoRA"
	@echo "  make evaluate-sft      — évaluation sur test set (honnête)"
	@echo "  make evaluate-sft EVAL_VAL=1  — idem + val set (biaisé, désactivé par défaut)"
	@echo ""
	@echo "  DPO"
	@echo "  make dpo-pipeline      — pipeline complet DPO (train → eval → export)"
	@echo "  make train-dpo         — alignement DPO LoRA"
	@echo "  make evaluate-dpo      — évaluation SFT vs DPO sur test set (honnête)"
	@echo "  make evaluate-dpo EVAL_VAL=1  — idem + val set (biaisé, désactivé par défaut)"
	@echo "  make export-model      — fusion LoRA + export format HuggingFace"
	@echo ""
	@echo "  MLflow"
	@echo "  make mlflow            — build + démarre le conteneur MLflow"
	@echo "  make mlflow-build      — construit l'image Docker MLflow"
	@echo "  make mlflow-up         — démarre le conteneur (port 127.0.0.1:5000)"
	@echo "  make mlflow-down       — arrête et supprime le conteneur"
	@echo "  make mlflow-logs       — affiche les logs du conteneur"
	@echo "  make clean-mlflow      — supprime tous les runs MLflow (mlruns/)"
	@echo ""
	@echo "  Nettoyage"
	@echo "  make clean             — supprime raw/ et processed/"
	@echo "  make clean-sft         — supprime checkpoints/sft et sft_tokenized/"
	@echo "  make clean-dpo         — supprime checkpoints/dpo et dpo_merged/"
	@echo "  make clean-all         — supprime tout data/ et checkpoints/"
