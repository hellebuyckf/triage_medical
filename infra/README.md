# Infrastructure Terraform — MLflow sur GCP

Déploie MLflow sur **Cloud Run** avec **Cloud SQL PostgreSQL** comme backend et **GCS** comme stockage d'artefacts.

## Architecture

```
Mac M3
  └── SSH tunnel (:8080)
        └── alpha-server
              └── gcloud run services proxy
                    └── Cloud Run (MLflow)
                          ├── Cloud SQL PostgreSQL (IP privée via VPC Connector)
                          └── GCS Bucket (artefacts, via service account)
```

## Prérequis

- [Terraform](https://developer.hashicorp.com/terraform/install) >= 1.3
- [gcloud CLI](https://cloud.google.com/sdk/docs/install) authentifié (`gcloud auth application-default login`)
- Docker (pour builder et pusher l'image MLflow)
- Accès au projet GCP `oc-p14`

## Structure

```
infra/
├── main.tf                  # Ressources racine + appel des modules
├── variables.tf             # Variables d'entrée
├── outputs.tf               # Outputs (URL Cloud Run, IPs, etc.)
├── terraform.tfvars.example # Template de configuration
├── modules/
│   ├── cloudsql/            # Instance PostgreSQL + VPC peering
│   └── cloudrun/            # Cloud Run + Artifact Registry + VPC Connector
```

## Déploiement

### 1. Configuration

```bash
cp terraform.tfvars.example terraform.tfvars
# Éditer terraform.tfvars avec les vraies valeurs
```

Variables requises dans `terraform.tfvars` :

| Variable | Description | Exemple |
|---|---|---|
| `project_id` | GCP Project ID | `oc-p14` |
| `db_password` | Mot de passe PostgreSQL | `monmotdepasse` |
| `authorized_invoker_email` | Email Google autorisé à accéder à MLflow | `moi@gmail.com` |

### 2. Init

```bash
terraform init
```

### 3. Builder et pusher l'image MLflow

L'image Docker doit exister dans Artifact Registry avant le déploiement du service Cloud Run.

```bash
# Créer le repository Artifact Registry en premier
terraform apply \
  -target=module.cloudrun.google_artifact_registry_repository.mlflow \
  -target=module.cloudrun.google_project_service.artifactregistry_api

# Authentifier Docker
gcloud auth configure-docker europe-west1-docker.pkg.dev

# Builder et pusher
docker build -t europe-west1-docker.pkg.dev/oc-p14/mlflow/mlflow:latest \
  ../docker/mlflow/
docker push europe-west1-docker.pkg.dev/oc-p14/mlflow/mlflow:latest
```

### 4. Déploiement complet

```bash
terraform apply
```

Les outputs affichent l'URL du service et l'URL Artifact Registry à la fin.

## Accès à l'interface MLflow

Le service est restreint à `authorized_invoker_email` — un token Google est requis.

**Depuis alpha-server :**
```bash
gcloud run services proxy mlflow --region=europe-west1 --project=oc-p14 --port=8080
```

**Depuis le Mac (tunnel SSH, dans un autre terminal) :**
```bash
ssh -L 8080:localhost:8080 alpha-server -N
```

Ouvrir **http://localhost:8080** dans le navigateur.

## Outputs

| Output | Description |
|---|---|
| `cloudrun_service_url` | URL publique du service Cloud Run |
| `artifact_registry_url` | URL Artifact Registry pour `docker push` |
| `cloudrun_service_account` | Email du service account Cloud Run |
| `private_ip` | IP privée Cloud SQL |
| `mlflow_artifact_root` | URI GCS des artefacts |

## Destruction

```bash
terraform destroy
```

> Le bucket GCS est configuré avec `force_destroy = true` — tous les artefacts seront supprimés.
