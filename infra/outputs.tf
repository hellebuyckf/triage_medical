output "instance_name" {
  description = "Nom de l'instance Cloud SQL"
  value       = module.cloudsql.instance_name
}

output "private_ip" {
  description = "IP privée de l'instance Cloud SQL"
  value       = module.cloudsql.private_ip
}

output "connection_name" {
  description = "Connection name Cloud SQL"
  value       = module.cloudsql.connection_name
}

output "mlflow_backend_store_uri" {
  description = "URI backend MLflow"
  value       = module.cloudsql.backend_store_uri
  sensitive   = true
}

output "mlflow_artifact_root" {
  description = "URI GCS pour les artefacts MLflow"
  value       = "gs://${google_storage_bucket.mlflow_artifacts.name}/artifacts"
}

output "cloudrun_service_url" {
  description = "URL publique du service MLflow sur Cloud Run"
  value       = module.cloudrun.service_url
}

output "artifact_registry_url" {
  description = "URL Artifact Registry pour docker push"
  value       = module.cloudrun.artifact_registry_url
}

output "cloudrun_service_account" {
  description = "Email du service account Cloud Run MLflow"
  value       = module.cloudrun.service_account_email
}
