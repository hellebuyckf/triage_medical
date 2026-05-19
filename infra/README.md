# Infrastructure Terraform — GCP

Déploie l'ensemble de l'infrastructure du POC triage médical sur GCP via Terraform.

## Architecture

```
Internet
  └── Cloud Run (FastAPI triage-api)    ← endpoint public /triage
        └── Compute Engine (vLLM GPU L4) ← inférence modèle (europe-west4-c)
              └── HuggingFace Hub        ← téléchargement du modèle au démarrage

Cloud Run (MLflow)                       ← experiment tracking (authentification basic auth)
  ├── Cloud SQL PostgreSQL (VPC privé)   ← backend runs / métriques
  └── GCS Bucket                         ← artefacts (checkpoints, eval reports)

Artifact Registry                        ← images Docker (mlflow, triage-api)
GCS Bucket                               ← état Terraform (oc-p14-terraform-state)
```

## Modules Terraform

| Module | Service GCP | Rôle |
|---|---|---|
| `modules/cloudsql` | Cloud SQL PostgreSQL | Base de données MLflow (accès VPC privé uniquement) |
| `modules/cloudrun` | Cloud Run | Service MLflow (basic auth) + VPC Connector |
| `modules/cloudrun_api` | Cloud Run | Gateway FastAPI `/triage` (public) + Artifact Registry |
| `modules/vllm_gce` | Compute Engine (GPU L4) | Inférence vLLM — zone `europe-west4-c` |

## Prérequis

- [Terraform](https://developer.hashicorp.com/terraform/install) >= 1.3
- [gcloud CLI](https://cloud.google.com/sdk/docs/install) authentifié (`gcloud auth application-default login`)
- Docker (pour builder et pusher les images)
- Accès au projet GCP `oc-p14`

## Structure

```
infra/
├── main.tf                  # Ressources racine + appel des modules + GCS bucket
├── variables.tf             # Variables d'entrée
├── outputs.tf               # Outputs (URLs, IPs, etc.)
├── Makefile                 # Cibles opérationnelles (start/stop/deploy/logs/ssh)
├── terraform.tfvars.example # Template de configuration (ne pas commiter le .tfvars)
└── modules/
    ├── cloudsql/            # Instance PostgreSQL + VPC peering
    ├── cloudrun/            # Cloud Run MLflow + Artifact Registry + VPC Connector
    ├── cloudrun_api/        # Cloud Run FastAPI + Artifact Registry dédié
    └── vllm_gce/            # VM GPU L4 + startup script Docker vLLM
```

## Déploiement

### 1. Configuration

```bash
cd infra/
cp terraform.tfvars.example terraform.tfvars
# Éditer terraform.tfvars avec les vraies valeurs
```

Variables requises dans `terraform.tfvars` :

| Variable | Description | Exemple |
|---|---|---|
| `project_id` | GCP Project ID | `oc-p14` |
| `db_password` | Mot de passe PostgreSQL | `motdepasse-fort` |
| `mlflow_admin_password` | Mot de passe basic auth MLflow | `motdepasse-fort` |
| `mlflow_flask_secret_key` | Clé secrète Flask CSRF | `clé-aléatoire` |
| `hf_token` | Token HuggingFace (téléchargement modèle sur VM) | `hf_xxx` |
| `model_id` | ID du modèle à servir | `FrancoisFormation/qwen3-triage-dpo` |

### 2. Init

```bash
terraform init
```

Le state est stocké dans le bucket GCS `oc-p14-terraform-state`.

### 3. Première fois — Bootstrap Artifact Registry

L'image Docker doit exister avant le déploiement du service Cloud Run.

```bash
# Créer les repositories Artifact Registry
terraform apply \
  -target=module.cloudrun.google_artifact_registry_repository.mlflow \
  -target=module.cloudrun_api.google_artifact_registry_repository.api_repo

# Authentifier Docker
gcloud auth configure-docker europe-west1-docker.pkg.dev

# Builder et pusher les images
make docker-push   # image MLflow
make api-push      # image FastAPI
```

### 4. Déploiement complet

```bash
terraform apply
```

## Makefile — Cibles opérationnelles

Toutes les cibles sont exécutables depuis la racine du projet avec `make -C infra <cible>`.

### Gestion des services

| Cible | Action |
|---|---|
| `make list` | État de tous les services (Cloud Run, Cloud SQL, GCE, GCS) |
| `make start` | Démarrer Cloud SQL + VM vLLM + restaurer trafic Cloud Run |
| `make stop` | Suspendre Cloud SQL + arrêter VM vLLM + mettre Cloud Run à 0 trafic |

### Déploiement MLflow

| Cible | Action |
|---|---|
| `make docker-build` | Build image MLflow |
| `make docker-push` | Build + push vers Artifact Registry |
| `make deploy` | `docker-push` + `terraform apply` |

### Déploiement FastAPI (prod)

| Cible | Action |
|---|---|
| `make api-build` | Build image FastAPI depuis la racine du projet |
| `make api-push` | Build + push vers Artifact Registry |
| `make prod-deploy` | `api-push` + `terraform apply` |

### Inférence — VM vLLM

| Cible | Action |
|---|---|
| `make vllm-start` | Démarrer la VM GPU (async) |
| `make vllm-stop` | Arrêter la VM GPU (async) |
| `make vllm-status` | État GCE + état Docker sur la VM (via SSH IAP) |
| `make vllm-logs` | Logs du conteneur vLLM (`docker logs -f`) |
| `make vllm-ssh` | Connexion SSH à la VM (via IAP, sans IP publique) |
| `make vllm-retry` | Boucle de démarrage toutes les 2 min (gestion stock GPU) |
| `make vllm-quotas` | Quotas GPU L4 disponibles en `europe-west4` |

### Cloud Run — Observabilité MLflow

| Cible | Action |
|---|---|
| `make cloudrun-status` | Description complète du service |
| `make cloudrun-url` | URL publique |
| `make cloudrun-logs` | 100 dernières lignes de logs |
| `make cloudrun-redeploy` | Force une nouvelle révision (repull `:latest`) |
| `make mlflow-proxy` | Tunnel `gcloud run services proxy` → `localhost:8080` |

## Accès aux services

### MLflow

Ouvrir directement l'URL Cloud Run (voir output `cloudrun_service_url`) dans le navigateur.

Identifiants : `mlflow_admin_username` / `mlflow_admin_password` (définis dans `terraform.tfvars`).

> **Note** : la base `auth.db` est éphémère (réinitialisée au démarrage du conteneur avec
> les identifiants admin). Pour des utilisateurs additionnels persistants, monter `auth.db`
> sur GCS ou utiliser Secret Manager.

### API FastAPI `/triage`

```bash
# URL publique (voir output api_service_url)
curl -X POST <API_URL>/triage \
  -H "Content-Type: application/json" \
  -d '{"symptoms": "Douleur thoracique intense, sudation, nausées depuis 30 min."}'
```

La FastAPI proxy les requêtes vers le serveur vLLM interne via l'IP privée de la VM GCE.

## Outputs Terraform

| Output | Description |
|---|---|
| `cloudrun_service_url` | URL publique du service MLflow (Cloud Run) |
| `api_service_url` | URL publique de la FastAPI triage (Cloud Run) |
| `artifact_registry_url` | URL Artifact Registry (image MLflow) |
| `api_artifact_registry_url` | URL Artifact Registry (image FastAPI) |
| `cloudrun_service_account` | Email du service account MLflow |
| `private_ip` | IP privée Cloud SQL |
| `mlflow_backend_store_uri` | URI de connexion PostgreSQL (sensitive) |
| `mlflow_artifact_root` | URI GCS pour les artefacts MLflow |
| `vllm_internal_ip` | IP interne de la VM vLLM (pour debug) |

## CI/CD (GitHub Actions)

| Workflow | Déclencheur | Étapes |
|---|---|---|
| `ci.yml` | Push + PR → `main` | Ruff, Pyright, pytest, `terraform fmt` + `validate` + `plan` |
| `cd.yml` | Push → `main` (paths `infra/**`, `scripts/serving/**`, `Dockerfile`) | Bootstrap AR → build + push image → `terraform apply` |

L'authentification GCP utilise **Workload Identity Federation** (sans clé de service account statique).
Les secrets sont stockés dans GitHub Actions Secrets : `GCP_WIF_PROVIDER`, `GCP_WIF_SERVICE_ACCOUNT`,
`GCP_PROJECT_ID`, `HF_TOKEN`, `DB_PASSWORD`, `MLFLOW_ADMIN_PASSWORD`, `MLFLOW_FLASK_SECRET_KEY`.

## Destruction

```bash
terraform destroy
```

> Le bucket GCS artefacts est configuré avec `force_destroy = true` — tous les artefacts seront supprimés.
> Le bucket d'état Terraform (`oc-p14-terraform-state`) est géré indépendamment et n'est pas détruit.

## Notes GPU

- La VM vLLM est déployée en `europe-west4-c` (zone avec GPU L4 disponibles).
- En cas de rupture de stock GPU, utiliser `make vllm-retry` pour relancer automatiquement.
- Quotas actuels : `make vllm-quotas`.
