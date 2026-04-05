terraform {
  required_version = ">= 1.3"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  backend "gcs" {
    bucket = "oc-p14-terraform-state"
    prefix = "terraform/state"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Cloud SQL PostgreSQL

module "cloudsql" {
  source = "./modules/cloudsql"

  project_id       = var.project_id
  region           = var.region
  db_instance_name = var.db_instance_name
  db_name          = var.db_name
  db_user          = var.db_user
  db_password      = var.db_password
  network_name     = var.network_name
}

# Cloud Run — service MLflow

module "cloudrun" {
  source = "./modules/cloudrun"

  project_id              = var.project_id
  region                  = var.region
  network_name            = var.network_name
  image_name              = var.cloudrun_image_name
  db_user                 = var.db_user
  db_password             = var.db_password
  db_host                 = module.cloudsql.private_ip
  db_name                 = var.db_name
  artifact_root           = "gs://${google_storage_bucket.mlflow_artifacts.name}/artifacts"
  gcs_bucket_name         = google_storage_bucket.mlflow_artifacts.name
  mlflow_admin_username   = var.mlflow_admin_username
  mlflow_admin_password   = var.mlflow_admin_password
  mlflow_flask_secret_key = var.mlflow_flask_secret_key
}

# GCS Bucket — artefacts MLflow

resource "google_storage_bucket" "mlflow_artifacts" {
  name          = "${var.project_id}-mlflow-artifacts"
  location      = var.region
  force_destroy = true # POC : permet terraform destroy sans vider le bucket

  uniform_bucket_level_access = true

  lifecycle_rule {
    condition {
      age = 90 # Supprimer les artefacts de plus de 90 jours (optionnel)
    }
    action {
      type = "Delete"
    }
  }
}

# Module Inférence — Compute Engine vLLM (GPU)
module "vllm_gce" {
  source = "./modules/vllm_gce"

  project_id   = var.project_id
  region       = "europe-west4"   # Fixed region for GPU availability
  zone         = "europe-west4-c" # Fixed zone for GPU availability (moved from -a due to stockout)
  network_name = var.network_name
  hf_token     = var.hf_token
}

# Module Gateway API — Cloud Run FastAPI
module "cloudrun_api" {
  source = "./modules/cloudrun_api"

  project_id        = var.project_id
  region            = var.region
  network_name      = var.network_name
  image_name        = var.cloudrun_api_image_name
  vllm_api_base_url = "http://${module.vllm_gce.internal_ip}:8000/v1"
  vllm_api_key      = var.hf_token                         # Utilisé comme token d'API local dans ce POC
  model_id          = "FrancoisFormation/qwen3-triage-dpo" # Match GCE config
}
