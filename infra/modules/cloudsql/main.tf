# ─────────────────────────────────────────────
# Activation des APIs
# ─────────────────────────────────────────────

resource "google_project_service" "servicenetworking_api" {
  project            = var.project_id
  service            = "servicenetworking.googleapis.com"
  disable_on_destroy = false
}

# ─────────────────────────────────────────────
# VPC Peering — nécessaire pour l'IP privée
# ─────────────────────────────────────────────

resource "google_compute_global_address" "private_ip_range" {
  name          = "mlflow-sql-private-ip-range"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 20
  network       = "projects/${var.project_id}/global/networks/${var.network_name}"
}

resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = "projects/${var.project_id}/global/networks/${var.network_name}"
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_range.name]

  depends_on = [google_project_service.servicenetworking_api]
}

# ─────────────────────────────────────────────
# Instance Cloud SQL PostgreSQL (taille POC)
# ─────────────────────────────────────────────

resource "google_sql_database_instance" "mlflow" {
  name             = var.db_instance_name
  database_version = "POSTGRES_16"
  region           = var.region

  deletion_protection = false

  settings {
    tier = "db-f1-micro"

    availability_type = "ZONAL"

    disk_type       = "PD_SSD"
    disk_size       = 10
    disk_autoresize = false

    backup_configuration {
      enabled = false
    }

    ip_configuration {
      ipv4_enabled    = false
      private_network = "projects/${var.project_id}/global/networks/${var.network_name}"
    }

    database_flags {
      name  = "max_connections"
      value = "50"
    }
  }

  depends_on = [google_service_networking_connection.private_vpc_connection]
}

# ─────────────────────────────────────────────
# Base de données MLflow
# ─────────────────────────────────────────────

resource "google_sql_database" "mlflow" {
  name     = var.db_name
  instance = google_sql_database_instance.mlflow.name
}

# ─────────────────────────────────────────────
# Utilisateur MLflow
# ─────────────────────────────────────────────

resource "google_sql_user" "mlflow" {
  name     = var.db_user
  instance = google_sql_database_instance.mlflow.name
  password = var.db_password
}
