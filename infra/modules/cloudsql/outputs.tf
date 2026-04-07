output "instance_name" {
  description = "Nom de l'instance Cloud SQL"
  value       = google_sql_database_instance.mlflow.name
}

output "private_ip" {
  description = "IP privée de l'instance Cloud SQL"
  value       = google_sql_database_instance.mlflow.private_ip_address
}

output "connection_name" {
  description = "Connection name Cloud SQL (pour Cloud SQL Proxy / Cloud Run)"
  value       = google_sql_database_instance.mlflow.connection_name
}

output "backend_store_uri" {
  description = "URI backend MLflow"
  value       = "postgresql://${var.db_user}:${var.db_password}@${google_sql_database_instance.mlflow.private_ip_address}:5432/${var.db_name}"
  sensitive   = true
}
