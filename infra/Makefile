PROJECT_ID        := oc-p14
REGION            := europe-west1
CLOUDRUN_SERVICE  := mlflow
CLOUDRUN_API      := triage-api
CLOUDSQL_INSTANCE := mlflow-poc-db
VLLM_VM           := vllm-server

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

# Images Artifact Registry
AR_MLFLOW := $(REGION)-docker.pkg.dev/$(PROJECT_ID)/$(CLOUDRUN_SERVICE)/$(CLOUDRUN_SERVICE):latest
AR_API    := $(REGION)-docker.pkg.dev/$(PROJECT_ID)/$(CLOUDRUN_API)/$(CLOUDRUN_API):latest

.PHONY: help list start stop \
        cloudrun-status cloudrun-logs cloudrun-url cloudrun-redeploy \
        docker-build docker-push deploy \
        api-build api-push prod-deploy \
        vllm-status vllm-logs vllm-ssh \
        mlflow-proxy mlflow-tunnel \
        mlflow mlflow-build mlflow-up mlflow-down mlflow-logs clean-mlflow

help:
	@echo "Usage: make <target>  (depuis infra/ ou : make -C infra <target>)"
	@echo ""
	@echo "  Services GCP"
	@echo "  list              — état de tous les services (Cloud Run, Cloud SQL, GCS, GCE)"
	@echo "  start             — démarrer Cloud SQL + Cloud Run + vLLM VM"
	@echo "  stop              — arrêter Cloud SQL + Cloud Run + vLLM VM (économie)"
	@echo ""
	@echo "  Déploiement (MLflow)"
	@echo "  deploy            — docker-build + docker-push + terraform apply"
	@echo "  docker-build      — build l'image Docker MLflow pour Cloud Run"
	@echo "  docker-push       — push l'image MLflow vers Artifact Registry"
	@echo ""
	@echo "  Déploiement (Prod - FastAPI + vLLM)"
	@echo "  prod-deploy       — api-build + api-push + terraform apply"
	@echo "  api-build         — build l'image Docker FastAPI (triage-api)"
	@echo "  api-push          — push l'image FastAPI vers Artifact Registry"
	@echo ""
	@echo "  Inférence (vLLM VM)"
	@echo "  vllm-status       — état de la VM GPU et du conteneur vLLM"
	@echo "  vllm-logs         — logs du conteneur vLLM sur la VM"
	@echo "  vllm-ssh          — connexion SSH à la VM vLLM (via IAP)"
	@echo ""
	@echo "  Cloud Run"
	@echo "  cloudrun-status   — état complet du service (URL, révision, trafic)"
	@echo "  cloudrun-url      — URL publique du service MLflow"
	@echo "  cloudrun-logs     — 100 dernières lignes de logs"
	@echo "  cloudrun-redeploy — force une nouvelle révision (repull image :latest)"
	@echo ""
	@echo "  Accès MLflow Cloud Run"
	@echo "  cloudrun-url      — affiche l'URL publique → ouvrir dans le navigateur"
	@echo ""
	@echo "  Accès MLflow local Docker (tunnel SSH depuis le Mac)"
	@echo "  1. Sur alpha-server : make mlflow-proxy"
	@echo "  2. Sur le Mac       : make mlflow-tunnel  →  http://localhost:8080"
	@echo ""
	@echo "  MLflow local Docker (ENV=dev)"
	@echo "  mlflow            — build + démarre le conteneur MLflow"
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
	@echo "=== Compute Engine (vLLM) ==="
	@gcloud compute instances list \
		--project=$(PROJECT_ID) \
		--format="table(name,zone,status,machineType,networkInterfaces[0].networkIP:label=INTERNAL_IP)"
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
	@echo ">>> Arrêt de la VM vLLM $(VLLM_VM)..."
	@gcloud compute instances stop $(VLLM_VM) \
		--project=$(PROJECT_ID) \
		--zone=europe-west4-c \
		--quiet
	@echo ">>> Mise à zéro du trafic Cloud Run..."
	@gcloud run services update-traffic $(CLOUDRUN_SERVICE) --region=$(REGION) --project=$(PROJECT_ID) --to-revisions=LATEST=0 --quiet 2>/dev/null || true
	@gcloud run services update-traffic $(CLOUDRUN_API) --region=$(REGION) --project=$(PROJECT_ID) --to-revisions=LATEST=0 --quiet 2>/dev/null || true
	@echo "Tous les services sont arrêtés (économie active)."

start:
	@echo ">>> Démarrage de Cloud SQL $(CLOUDSQL_INSTANCE)..."
	@gcloud sql instances patch $(CLOUDSQL_INSTANCE) \
		--activation-policy=ALWAYS \
		--project=$(PROJECT_ID) \
		--quiet
	@echo ">>> Démarrage de la VM vLLM $(VLLM_VM)..."
	@gcloud compute instances start $(VLLM_VM) \
		--project=$(PROJECT_ID) \
		--zone=europe-west4-c \
		--quiet
	@echo ">>> Restauration du trafic Cloud Run..."
	@gcloud run services update-traffic $(CLOUDRUN_SERVICE) --region=$(REGION) --project=$(PROJECT_ID) --to-latest --quiet
	@gcloud run services update-traffic $(CLOUDRUN_API) --region=$(REGION) --project=$(PROJECT_ID) --to-latest --quiet
	@echo "Tous les services sont démarrés."

# ─────────────────────────────────────────────
# Déploiement — MLflow
# ─────────────────────────────────────────────

docker-build:
	docker build -t $(AR_MLFLOW) $(PROJECT_ROOT)/docker/mlflow/

docker-push: docker-build
	docker push $(AR_MLFLOW)

deploy: docker-push
	terraform apply

# ─────────────────────────────────────────────
# Déploiement — Prod (API FastAPI)
# ─────────────────────────────────────────────

api-build:
	docker build -t $(AR_API) $(PROJECT_ROOT)/

api-push: api-build
	docker push $(AR_API)

prod-deploy: api-push
	terraform apply

# ─────────────────────────────────────────────
# Inférence — Observabilité VM vLLM
# ─────────────────────────────────────────────

vllm-status:
	@echo ">>> Status de l'instance GCE :"
	@gcloud compute instances describe $(VLLM_VM) --zone=europe-west4-c --project=$(PROJECT_ID) --format="value(status)"
	@echo ">>> Status des conteneurs Docker (via SSH IAP) :"
	@gcloud compute ssh $(VLLM_VM) --zone=europe-west4-c --project=$(PROJECT_ID) --tunnel-through-iap --command="docker ps"

vllm-logs:
	@gcloud compute ssh $(VLLM_VM) --zone=europe-west4-c --project=$(PROJECT_ID) --tunnel-through-iap --command="docker logs -f vllm"

vllm-ssh:
	gcloud compute ssh $(VLLM_VM) --zone=europe-west4-c --project=$(PROJECT_ID) --tunnel-through-iap

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
	    --image=$(AR_MLFLOW) \
	    --region=$(REGION) \
	    --project=$(PROJECT_ID)

# ─────────────────────────────────────────────
# Accès MLflow Cloud Run
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
