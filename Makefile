.PHONY: all setup download build-sft build-dpo anonymize split \
        prepare-tokenizer train-sft evaluate-sft sft-pipeline \
        clean clean-all help

# Variables
PYTHON   = uv run python
DATA_PREP = scripts/data_prep
TRAINING  = scripts/training

# Cible par défaut : pipeline complet S1
all: download build-sft build-dpo anonymize split

# Installation des dépendances et modèles spaCy
setup:
	uv sync
	uv pip install https://github.com/explosion/spacy-models/releases/download/fr_core_news_md-3.8.0/fr_core_news_md-3.8.0-py3-none-any.whl
	uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl

# ── Semaine 1 — Data Engineering ─────────────────────────────────────────────

# Étape 1 : téléchargement des datasets
download:
	$(PYTHON) $(DATA_PREP)/01_download.py

# Étape 2 : construction du dataset SFT
build-sft: download
	$(PYTHON) $(DATA_PREP)/02_build_sft.py

# Étape 3 : construction du dataset DPO
build-dpo: download
	$(PYTHON) $(DATA_PREP)/03_build_dpo.py

# Étape 4 : anonymisation RGPD
anonymize: build-sft build-dpo
	$(PYTHON) $(DATA_PREP)/04_anonymize.py

# Étape 5 : split et validation finale
split: anonymize
	$(PYTHON) $(DATA_PREP)/05_split_and_validate.py

# ── Semaine 2 — SFT ──────────────────────────────────────────────────────────

# Étape 10 : tokenisation + formatage ChatML
prepare-tokenizer:
	$(PYTHON) $(TRAINING)/10_prepare_tokenizer.py

# Étape 11 : entraînement SFT LoRA
train-sft: prepare-tokenizer
	$(PYTHON) $(TRAINING)/11_train_sft.py

# Étape 12 : évaluation du modèle fine-tuné
evaluate-sft: train-sft
	$(PYTHON) $(TRAINING)/12_evaluate_sft.py

# Pipeline complet S2
sft-pipeline: prepare-tokenizer train-sft evaluate-sft

# ── Nettoyage ─────────────────────────────────────────────────────────────────

# Nettoyage des fichiers intermédiaires (raw et processed)
# Ne supprime PAS data/final/ ni checkpoints/
clean:
	rm -rf data/raw data/processed

# Nettoyage complet incluant les livrables et checkpoints
clean-all:
	rm -rf data/raw data/processed data/final checkpoints

# Aide
help:
	@echo "Cibles disponibles :"
	@echo ""
	@echo "  Setup"
	@echo "  make setup             — installe les dépendances et modèles spaCy"
	@echo ""
	@echo "  Semaine 1 — Data Engineering"
	@echo "  make all               — pipeline complet S1 (download → split)"
	@echo "  make download          — télécharge les datasets HuggingFace"
	@echo "  make build-sft         — construit le dataset SFT"
	@echo "  make build-dpo         — construit le dataset DPO"
	@echo "  make anonymize         — anonymisation RGPD + rapport"
	@echo "  make split             — split train/val/test + validation"
	@echo ""
	@echo "  Semaine 2 — SFT"
	@echo "  make sft-pipeline      — pipeline complet S2 (tokenize → train → eval)"
	@echo "  make prepare-tokenizer — tokenisation + formatage ChatML"
	@echo "  make train-sft         — entraînement SFT LoRA"
	@echo "  make evaluate-sft      — évaluation du modèle fine-tuné"
	@echo ""
	@echo "  Nettoyage"
	@echo "  make clean             — supprime raw/ et processed/"
	@echo "  make clean-all         — supprime tout data/ et checkpoints/"
