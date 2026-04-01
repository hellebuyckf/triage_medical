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

      # Cloud Run n'interpole pas les variables d'env dans les args :
      # l'URI PostgreSQL est construite directement en HCL.
      command = ["mlflow"]
      args = [
        "server",
        "--backend-store-uri",
        "postgresql://${var.db_user}:${var.db_password}@${var.db_host}:5432/${var.db_name}",
        "--default-artifact-root",
        var.artifact_root,
        "--host",
        "0.0.0.0",
        "--port",
        "5000",
      ]

      # Variables d'env disponibles pour le code Python (os.environ)
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
# IAM — accès public au service (POC)
# ─────────────────────────────────────────────

resource "google_cloud_run_v2_service_iam_member" "authorized_invoker" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.mlflow.name
  role     = "roles/run.invoker"
  member   = "user:${var.authorized_invoker_email}"
}
