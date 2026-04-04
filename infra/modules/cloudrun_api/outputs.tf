output "service_url" {
  description = "L'URL publique du service Cloud Run FastAPI"
  value       = google_cloud_run_v2_service.api.uri
}

output "artifact_registry_url" {
  description = "L'URL du registry Docker pour l'API"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${var.image_name}"
}
