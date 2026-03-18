.PHONY: all setup download build-sft build-dpo anonymize split \
        prepare-tokenizer train-sft evaluate-sft sft-pipeline \
        dpo-pipeline train-dpo evaluate-dpo export-model \
        mlflow mlflow-build mlflow-up mlflow-down mlflow-logs clean-mlflow \
        clean clean-all help

# Variables
PYTHON          = uv run python
DATA_PREP       = scripts/data_prep
TRAINING        = scripts/training
MLFLOW_IMAGE    = project14-mlflow
MLFLOW_CONTAINER = project14-mlflow

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

# ── Semaine 3 — Pipeline DPO ──────────────────────────────────────────────────

# Pipeline complet S3
dpo-pipeline: train-dpo evaluate-dpo export-model

# Étape 20 : alignement DPO LoRA
train-dpo:
	$(PYTHON) $(TRAINING)/20_train_dpo.py

# Étape 21 : évaluation SFT vs DPO
evaluate-dpo: train-dpo
	$(PYTHON) $(TRAINING)/21_evaluate_dpo.py

# Étape 22 : fusion des poids LoRA et export HuggingFace
export-model: evaluate-dpo
	$(PYTHON) $(TRAINING)/22_export_model.py

# ── MLflow (Docker) ───────────────────────────────────────────────────────────
#
# Accès depuis le Mac M3 via tunnel SSH :
#   ssh -L 5000:localhost:5000 <user>@<ip_serveur>
# puis ouvrir http://localhost:5000 dans le navigateur.

# Build de l'image Docker MLflow
mlflow-build:
	docker build -t $(MLFLOW_IMAGE) -f docker/mlflow/Dockerfile .

# Démarrage du conteneur (détaché, redémarre automatiquement)
mlflow-up:
	docker run -d \
		--name $(MLFLOW_CONTAINER) \
		-p 127.0.0.1:5000:5000 \
		-v $(PWD)/mlruns:/mlruns:ro \
		--restart unless-stopped \
		$(MLFLOW_IMAGE)
	@echo "MLflow UI démarré → tunnel SSH : ssh -L 5000:localhost:5000 <user>@<ip_serveur>"

# Arrêt et suppression du conteneur
mlflow-down:
	docker stop $(MLFLOW_CONTAINER) && docker rm $(MLFLOW_CONTAINER)

# Logs du conteneur en temps réel
mlflow-logs:
	docker logs -f $(MLFLOW_CONTAINER)

# Alias pratique : build + up en une commande
mlflow: mlflow-build mlflow-up

# Supprime tous les runs MLflow (mlruns/)
clean-mlflow:
	rm -rf mlruns/

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
	@echo "  Semaine 3 — DPO"
	@echo "  make dpo-pipeline      — pipeline complet S3 (train → eval → export)"
	@echo "  make train-dpo         — alignement DPO LoRA"
	@echo "  make evaluate-dpo      — évaluation SFT vs DPO + rapport"
	@echo "  make export-model      — fusion LoRA + export format HuggingFace"
	@echo ""
	@echo "  MLflow (Docker)"
	@echo "  make mlflow            — build + démarre le conteneur MLflow"
	@echo "  make mlflow-build      — construit l'image Docker MLflow"
	@echo "  make mlflow-up         — démarre le conteneur (port 127.0.0.1:5000)"
	@echo "  make mlflow-down       — arrête et supprime le conteneur"
	@echo "  make mlflow-logs       — affiche les logs du conteneur"
	@echo "  make clean-mlflow      — supprime tous les runs MLflow (mlruns/)"
	@echo ""
	@echo "  Nettoyage"
	@echo "  make clean             — supprime raw/ et processed/"
	@echo "  make clean-all         — supprime tout data/ et checkpoints/"
