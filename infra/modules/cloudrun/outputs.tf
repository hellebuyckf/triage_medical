output "service_url" {
  description = "URL publique du service Cloud Run MLflow"
  value       = google_cloud_run_v2_service.mlflow.uri
}

output "service_account_email" {
  description = "Email du service account Cloud Run"
  value       = google_service_account.cloudrun_sa.email
}

output "artifact_registry_url" {
  description = "URL du repository Artifact Registry (pour docker push)"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${var.image_name}"
}

output "vpc_connector_id" {
  description = "ID du VPC Access Connector"
  value       = google_vpc_access_connector.mlflow.id
}
