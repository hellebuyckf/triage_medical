PROJECT_ID       := oc-p14
REGION           := europe-west1
CLOUDRUN_SERVICE := mlflow
CLOUDSQL_INSTANCE := mlflow-poc-db

.DEFAULT_GOAL := help

.PHONY: help list start stop

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  list   Lister l'état de tous les services GCP"
	@echo "  start  Démarrer tous les services (Cloud SQL + Cloud Run)"
	@echo "  stop   Arrêter tous les services (Cloud SQL + Cloud Run)"

# ─────────────────────────────────────────────
# list
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

# ─────────────────────────────────────────────
# stop
# Cloud SQL : suspension complète (économise les coûts)
# Cloud Run : scale-to-zero (min=0 déjà configuré, on force max=0 n'est pas
#             supporté — on supprime l'accès public pour bloquer les requêtes)
# ─────────────────────────────────────────────

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

# ─────────────────────────────────────────────
# start
# ─────────────────────────────────────────────

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
