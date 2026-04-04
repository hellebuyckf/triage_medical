# Activation des APIs
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

# Artifact Registry pour l'image Docker de l'API
resource "google_artifact_registry_repository" "api_repo" {
  project       = var.project_id
  location      = var.region
  repository_id = var.image_name
  format        = "DOCKER"
  description   = "Image Docker API FastAPI"

  depends_on = [google_project_service.artifactregistry_api]
}

# Service Account dédié pour Cloud Run API
resource "google_service_account" "cloudrun_api_sa" {
  project      = var.project_id
  account_id   = "cloudrun-api-sa"
  display_name = "Service Account API Cloud Run"
}

# Rôle pour lire l'image depuis Artifact Registry
resource "google_artifact_registry_repository_iam_member" "cloudrun_api_sa_ar" {
  project    = var.project_id
  location   = var.region
  repository = google_artifact_registry_repository.api_repo.repository_id
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${google_service_account.cloudrun_api_sa.email}"
}

# VPC Access Connector
resource "google_vpc_access_connector" "api_connector" {
  project       = var.project_id
  name          = "api-vpc-connector"
  region        = var.region
  network       = var.network_name
  ip_cidr_range = "10.9.0.0/28"

  min_instances = 2
  max_instances = 3
  machine_type  = "e2-micro"

  depends_on = [google_project_service.vpcaccess_api]
}

# Service Cloud Run v2 - API
resource "google_cloud_run_v2_service" "api" {
  project  = var.project_id
  name     = var.image_name
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.cloudrun_api_sa.email

    scaling {
      min_instance_count = 0
      max_instance_count = 10
    }

    vpc_access {
      connector = google_vpc_access_connector.api_connector.id
      egress    = "PRIVATE_RANGES_ONLY"
    }

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.image_name}/${var.image_name}:latest"

      ports {
        container_port = 8080
      }

      env {
        name  = "VLLM_API_BASE_URL"
        value = var.vllm_api_base_url
      }
      env {
        name  = "VLLM_API_KEY"
        value = var.vllm_api_key
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
        cpu_idle = true
      }
    }
  }

  depends_on = [
    google_project_service.run_api,
    google_artifact_registry_repository.api_repo,
    google_vpc_access_connector.api_connector,
  ]
}

# IAM — accès public à l'API
resource "google_cloud_run_v2_service_iam_member" "public_invoker" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
