variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
}

variable "network_name" {
  description = "Nom du VPC réseau GCP (doit être le même que Cloud SQL)"
  type        = string
  default     = "default"
}

variable "image_name" {
  description = "Nom du service Cloud Run et du repository Artifact Registry"
  type        = string
  default     = "mlflow"
}

variable "db_user" {
  description = "Utilisateur PostgreSQL pour MLflow"
  type        = string
}

variable "db_password" {
  description = "Mot de passe PostgreSQL"
  type        = string
  sensitive   = true
}

variable "db_host" {
  description = "IP privée de l'instance Cloud SQL"
  type        = string
}

variable "db_name" {
  description = "Nom de la base de données MLflow"
  type        = string
  default     = "mlflow"
}

variable "artifact_root" {
  description = "URI GCS pour les artefacts MLflow (ex: gs://bucket/artifacts)"
  type        = string
}

variable "gcs_bucket_name" {
  description = "Nom du bucket GCS (sans gs://) pour les IAM bindings"
  type        = string
}

variable "authorized_invoker_email" {
  description = "Email Google autorisé à invoquer le service Cloud Run (ex: user@gmail.com)"
  type        = string
}
