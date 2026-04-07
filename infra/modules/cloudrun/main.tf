# ─────────────────────────────────────────────
# Activation des APIs
# ─────────────────────────────────────────────

resource "google_project_service" "run_api" {
  project            = var.project_id
  service            = "run.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "artifactregistry_api" {
  project            = var.project_id
  service            = "artifactregistry.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "vpcaccess_api" {
  project            = var.project_id
  service            = "vpcaccess.googleapis.com"
  disable_on_destroy = false
}

# ─────────────────────────────────────────────
# Artifact Registry — repository Docker
# ─────────────────────────────────────────────

resource "google_artifact_registry_repository" "mlflow" {
  project       = var.project_id
  location      = var.region
  repository_id = var.image_name
  format        = "DOCKER"
  description   = "Images Docker MLflow"

  depends_on = [google_project_service.artifactregistry_api]
}

# ─────────────────────────────────────────────
# Service Account dédié Cloud Run
# ─────────────────────────────────────────────

resource "google_service_account" "cloudrun_sa" {
  project      = var.project_id
  account_id   = "mlflow-cloudrun-sa"
  display_name = "Service Account MLflow Cloud Run"
}

# Rôle storage.objectAdmin sur le bucket GCS des artefacts
resource "google_storage_bucket_iam_member" "cloudrun_sa_gcs" {
  bucket = var.gcs_bucket_name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.cloudrun_sa.email}"
}

# Rôle permettant de lire les images depuis Artifact Registry
resource "google_artifact_registry_repository_iam_member" "cloudrun_sa_ar" {
  project    = var.project_id
  location   = var.region
  repository = google_artifact_registry_repository.mlflow.repository_id
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${google_service_account.cloudrun_sa.email}"
}

# ─────────────────────────────────────────────
# VPC Access Connector
# Nécessaire pour atteindre Cloud SQL en IP privée depuis Cloud Run
# ─────────────────────────────────────────────

resource "google_vpc_access_connector" "mlflow" {
  project       = var.project_id
  name          = "mlflow-vpc-connector"
  region        = var.region
  network       = var.network_name
  ip_cidr_range = "10.8.0.0/28"

  min_instances = 2
  max_instances = 3
  machine_type  = "e2-micro"

  depends_on = [google_project_service.vpcaccess_api]
}

# ─────────────────────────────────────────────
# Service Cloud Run v2 — MLflow
# ─────────────────────────────────────────────

resource "google_cloud_run_v2_service" "mlflow" {
  project  = var.project_id
  name     = var.image_name
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.cloudrun_sa.email

    scaling {
      min_instance_count = 0
      max_instance_count = 1
    }

    vpc_access {
      connector = google_vpc_access_connector.mlflow.id
      egress    = "PRIVATE_RANGES_ONLY"
    }

    containers {
      # L'image doit être buildée et pushée avant terraform apply
      # Voir outputs.artifact_registry_url pour l'URL de push
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.image_name}/${var.image_name}:latest"

      ports {
        container_port = 5000
      }

      # Le démarrage est géré par start.sh (ENTRYPOINT de l'image Docker).
      # start.sh lit ces variables d'env pour construire basic_auth.ini
      # et lancer mlflow server --app-name basic-auth.
      env {
        name  = "DB_USER"
        value = var.db_user
      }
      env {
        name  = "DB_PASSWORD"
        value = var.db_password
      }
      env {
        name  = "DB_HOST"
        value = var.db_host
      }
      env {
        name  = "DB_NAME"
        value = var.db_name
      }
      env {
        name  = "ARTIFACT_ROOT"
        value = var.artifact_root
      }
      env {
        name  = "MLFLOW_ADMIN_USERNAME"
        value = var.mlflow_admin_username
      }
      env {
        name  = "MLFLOW_ADMIN_PASSWORD"
        value = var.mlflow_admin_password
      }
      env {
        name  = "MLFLOW_AUTH_CONFIG_PATH"
        value = "/tmp/basic_auth.ini"
      }
      env {
        name  = "MLFLOW_FLASK_SERVER_SECRET_KEY"
        value = var.mlflow_flask_secret_key
      }
      # Autoriser tous les Host headers (l'URL Cloud Run est inconnue à l'avance)
      env {
        name  = "MLFLOW_SERVER_ALLOWED_HOSTS"
        value = "*"
      }
      env {
        name  = "MLFLOW_SERVER_CORS_ALLOWED_ORIGINS"
        value = "*"
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "2Gi"
        }
        cpu_idle = true
      }

      startup_probe {
        http_get {
          path = "/health"
          port = 5000
        }
        initial_delay_seconds = 10
        period_seconds        = 10
        failure_threshold     = 5
      }
    }
  }

  depends_on = [
    google_project_service.run_api,
    google_artifact_registry_repository.mlflow,
    google_vpc_access_connector.mlflow,
  ]
}

# ─────────────────────────────────────────────
# IAM — accès public (authentification déléguée à MLflow basic auth)
# ─────────────────────────────────────────────

resource "google_cloud_run_v2_service_iam_member" "public_invoker" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.mlflow.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
