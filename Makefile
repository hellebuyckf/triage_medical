.PHONY: all setup download build-sft build-dpo anonymize split clean clean-all help

# Variables
PYTHON = uv run python
SCRIPTS = scripts

# Cible par défaut : pipeline complet
all: download build-sft build-dpo anonymize split

# Installation des dépendances et modèles spaCy
setup:
	uv sync
	uv pip install https://github.com/explosion/spacy-models/releases/download/fr_core_news_md-3.8.0/fr_core_news_md-3.8.0-py3-none-any.whl
	uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl

# Étape 1 : téléchargement des datasets
download:
	$(PYTHON) $(SCRIPTS)/01_download.py

# Étape 2 : construction du dataset SFT
build-sft: download
	$(PYTHON) $(SCRIPTS)/02_build_sft.py

# Étape 3 : construction du dataset DPO
build-dpo: download
	$(PYTHON) $(SCRIPTS)/03_build_dpo.py

# Étape 4 : anonymisation RGPD
anonymize: build-sft build-dpo
	$(PYTHON) $(SCRIPTS)/04_anonymize.py

# Étape 5 : split et validation finale
split: anonymize
	$(PYTHON) $(SCRIPTS)/05_split_and_validate.py

# Nettoyage des fichiers intermédiaires (raw et processed)
# Ne supprime PAS data/final/
clean:
	rm -rf data/raw data/processed

# Nettoyage complet incluant les livrables
clean-all:
	rm -rf data/raw data/processed data/final

# Aide
help:
	@echo "Cibles disponibles :"
	@echo "  make setup      — installe les dépendances et modèles spaCy"
	@echo "  make all        — pipeline complet (download → split)"
	@echo "  make download   — télécharge les datasets HuggingFace"
	@echo "  make build-sft  — construit le dataset SFT"
	@echo "  make build-dpo  — construit le dataset DPO"
	@echo "  make anonymize  — anonymisation RGPD + rapport"
	@echo "  make split      — split train/val/test + validation"
	@echo "  make clean      — supprime raw/ et processed/"
	@echo "  make clean-all  — supprime tout data/"
