PROJECT_ID        := oc-p14
REGION            := europe-west1
CLOUDRUN_SERVICE  := mlflow
CLOUDSQL_INSTANCE := mlflow-poc-db

# Racine du projet (parent de infra/)
PROJECT_ROOT      := $(realpath $(dir $(lastword $(MAKEFILE_LIST)))..)

# MLflow local Docker (ENV=dev)
MLFLOW_IMAGE      := project14-mlflow
MLFLOW_CONTAINER  := project14-mlflow

# SSH — tunnel vers le serveur de calcul (accès MLflow local depuis le Mac)
# SSH_USER : utilisateur Linux sur le serveur distant
# SSH_KEY  : chemin vers la clé privée SSH (optionnel — omis si géré par ssh-agent)
# Usage    : make mlflow-tunnel
#            make mlflow-tunnel SSH_USER=francois SSH_KEY=~/.ssh/id_ed25519
ALPHA_HOST        ?= 100.115.15.123
SSH_USER          ?= francois
SSH_KEY           ?=
_SSH_ID           = $(if $(SSH_KEY),-i $(SSH_KEY),)

CLOUDRUN_LINES    ?= 100

.DEFAULT_GOAL := help

# Image Cloud Run — Artifact Registry
AR_IMAGE := $(REGION)-docker.pkg.dev/$(PROJECT_ID)/$(CLOUDRUN_SERVICE)/$(CLOUDRUN_SERVICE):latest

.PHONY: help list start stop \
        cloudrun-status cloudrun-logs cloudrun-url cloudrun-redeploy \
        docker-build docker-push deploy \
        mlflow-proxy mlflow-tunnel \
        mlflow mlflow-build mlflow-up mlflow-down mlflow-logs clean-mlflow

help:
	@echo "Usage: make <target>  (depuis infra/ ou : make -C infra <target>)"
	@echo ""
	@echo "  Services GCP"
	@echo "  list              — état de tous les services (Cloud Run, Cloud SQL, GCS)"
	@echo "  start             — démarrer Cloud SQL + Cloud Run"
	@echo "  stop              — arrêter Cloud SQL + Cloud Run (économise les coûts)"
	@echo ""
	@echo "  Déploiement"
	@echo "  deploy            — docker-build + docker-push + terraform apply"
	@echo "  docker-build      — build l'image Docker MLflow pour Cloud Run"
	@echo "  docker-push       — push l'image vers Artifact Registry"
	@echo ""
	@echo "  Cloud Run"
	@echo "  cloudrun-status   — état complet du service (URL, révision, trafic)"
	@echo "  cloudrun-url      — URL publique du service"
	@echo "  cloudrun-logs       — 100 dernières lignes de logs"
	@echo "  cloudrun-logs CLOUDRUN_LINES=200  — nombre de lignes personnalisé"
	@echo "  cloudrun-redeploy   — force une nouvelle révision (repull image :latest)"
	@echo "  cloudrun-logs-revision REVISION=<nom>  — logs d'une révision spécifique"
	@echo ""
	@echo "  Accès MLflow Cloud Run"
	@echo "  cloudrun-url      — affiche l'URL publique → ouvrir dans le navigateur"
	@echo "  (login : admin / mot de passe défini dans terraform.tfvars)"
	@echo ""
	@echo "  Accès MLflow local Docker (tunnel SSH depuis le Mac)"
	@echo "  1. Sur alpha-server : make mlflow-proxy"
	@echo "  2. Sur le Mac       : make mlflow-tunnel  →  http://localhost:8080"
	@echo "  mlflow-tunnel SSH_KEY=~/.ssh/id_ed25519   — clé SSH explicite"
	@echo ""
	@echo "  MLflow local Docker (ENV=dev)"
	@echo "  mlflow            — build + démarre le conteneur MLflow"
	@echo "  mlflow-build      — construit l'image Docker MLflow"
	@echo "  mlflow-up         — démarre le conteneur (port 5000)"
	@echo "  mlflow-down       — arrête et supprime le conteneur"
	@echo "  mlflow-logs       — affiche les logs du conteneur"
	@echo "  clean-mlflow      — supprime mlruns/ et mlflow.db"

# ─────────────────────────────────────────────
# Services GCP — liste, démarrage, arrêt
# ─────────────────────────────────────────────

list:
	@echo "=== Cloud Run ==="
	@gcloud run services list \
		--project=$(PROJECT_ID) \
		--region=$(REGION) \
		--format="table(name,status,traffic[0].latestRevision:label=LATEST,url)"
	@echo ""
	@echo "=== Cloud SQL ==="
	@gcloud sql instances list \
		--project=$(PROJECT_ID) \
		--format="table(name,state,region,databaseVersion,settings.tier)"
	@echo ""
	@echo "=== GCS Bucket ==="
	@gcloud storage buckets list \
		--project=$(PROJECT_ID) \
		--format="table(name,location,storageClass)"

stop:
	@echo ">>> Suspension de Cloud SQL $(CLOUDSQL_INSTANCE)..."
	@gcloud sql instances patch $(CLOUDSQL_INSTANCE) \
		--activation-policy=NEVER \
		--project=$(PROJECT_ID) \
		--quiet
	@echo ">>> Cloud SQL suspendu."
	@echo ""
	@echo ">>> Mise à zéro du trafic Cloud Run $(CLOUDRUN_SERVICE)..."
	@gcloud run services update-traffic $(CLOUDRUN_SERVICE) \
		--region=$(REGION) \
		--project=$(PROJECT_ID) \
		--to-revisions=LATEST=0 \
		--quiet 2>/dev/null || \
	echo "    (Cloud Run scale-to-zero actif — aucun trafic entrant ne démarrera d'instance)"
	@echo ""
	@echo "Tous les services sont arrêtés."

start:
	@echo ">>> Démarrage de Cloud SQL $(CLOUDSQL_INSTANCE)..."
	@gcloud sql instances patch $(CLOUDSQL_INSTANCE) \
		--activation-policy=ALWAYS \
		--project=$(PROJECT_ID) \
		--quiet
	@echo ">>> Cloud SQL en cours de démarrage (peut prendre ~1 minute)."
	@echo ""
	@echo ">>> Restauration du trafic Cloud Run $(CLOUDRUN_SERVICE)..."
	@gcloud run services update-traffic $(CLOUDRUN_SERVICE) \
		--region=$(REGION) \
		--project=$(PROJECT_ID) \
		--to-latest \
		--quiet
	@echo ">>> Cloud Run prêt."
	@echo ""
	@echo "Tous les services sont démarrés."
	@echo ""
	@echo "URL MLflow : $$(gcloud run services describe $(CLOUDRUN_SERVICE) \
		--region=$(REGION) \
		--project=$(PROJECT_ID) \
		--format='value(status.url)')"

# ─────────────────────────────────────────────
# Déploiement — build Docker + push + terraform apply
# ─────────────────────────────────────────────

docker-build:
	docker build -t $(AR_IMAGE) $(PROJECT_ROOT)/docker/mlflow/

docker-push: docker-build
	docker push $(AR_IMAGE)

deploy: docker-push
	terraform apply

# ─────────────────────────────────────────────
# Cloud Run — observabilité
# ─────────────────────────────────────────────

cloudrun-status:
	gcloud run services describe $(CLOUDRUN_SERVICE) \
	    --region=$(REGION) \
	    --project=$(PROJECT_ID)

cloudrun-url:
	@gcloud run services describe $(CLOUDRUN_SERVICE) \
	    --region=$(REGION) \
	    --project=$(PROJECT_ID) \
	    --format='value(status.url)'

cloudrun-logs:
	gcloud run services logs read $(CLOUDRUN_SERVICE) \
	    --region=$(REGION) \
	    --project=$(PROJECT_ID) \
	    --limit=$(CLOUDRUN_LINES)

cloudrun-redeploy:
	gcloud run deploy $(CLOUDRUN_SERVICE) \
	    --image=$(AR_IMAGE) \
	    --region=$(REGION) \
	    --project=$(PROJECT_ID)

# Logs d'une révision spécifique — utile pour diagnostiquer un échec de démarrage.
# Usage : make cloudrun-logs-revision REVISION=mlflow-00004-hcg
REVISION ?=
cloudrun-logs-revision:
	@test -n "$(REVISION)" || (echo "Usage: make cloudrun-logs-revision REVISION=<nom>"; exit 1)
	gcloud logging read \
	    'resource.type="cloud_run_revision" AND resource.labels.revision_name="$(REVISION)"' \
	    --project=$(PROJECT_ID) \
	    --limit=100 \
	    --order=asc \
	    --format="value(textPayload)"

# ─────────────────────────────────────────────
# Accès MLflow Cloud Run
# mlflow-proxy : à lancer sur alpha-server (étape 1)
#   Forward Cloud Run → localhost:8080 via gcloud (nécessite gcloud auth).
# mlflow-tunnel : à lancer depuis le Mac (étape 2)
#   Forward alpha-server:8080 → localhost:8080 via SSH.
# ─────────────────────────────────────────────

mlflow-proxy:
	gcloud run services proxy $(CLOUDRUN_SERVICE) \
	    --region=$(REGION) \
	    --project=$(PROJECT_ID) \
	    --port=8080

mlflow-tunnel:
	@echo "Tunnel SSH → $(SSH_USER)@$(ALPHA_HOST):8080 → http://localhost:8080"
	ssh -N $(_SSH_ID) -L 8080:localhost:8080 $(SSH_USER)@$(ALPHA_HOST)

# ─────────────────────────────────────────────
# MLflow local Docker (ENV=dev)
# Utilise les chemins du projet racine (PROJECT_ROOT) pour les volumes.
# ─────────────────────────────────────────────

mlflow: mlflow-build mlflow-up

mlflow-build:
	docker build -t $(MLFLOW_IMAGE) -f $(PROJECT_ROOT)/docker/mlflow/Dockerfile $(PROJECT_ROOT)

mlflow-up:
	docker run -d \
		--name $(MLFLOW_CONTAINER) \
		-p 5000:5000 \
		-e MLFLOW_SERVER_ALLOWED_HOSTS="$(ALPHA_HOST),$(ALPHA_HOST):5000,localhost,127.0.0.1" \
		-e MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE=true \
		-v $(PROJECT_ROOT)/mlflow.db:/mlflow.db \
		-v $(PROJECT_ROOT)/mlruns:/mlruns \
		--restart unless-stopped \
		$(MLFLOW_IMAGE)
	@echo "MLflow UI démarré → http://$(ALPHA_HOST):5000"

mlflow-down:
	docker stop $(MLFLOW_CONTAINER) && docker rm $(MLFLOW_CONTAINER)

mlflow-logs:
	docker logs -f $(MLFLOW_CONTAINER)

clean-mlflow:
	rm -rf $(PROJECT_ROOT)/mlruns/ $(PROJECT_ROOT)/mlflow.db
