variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "europe-west1"
}

variable "db_instance_name" {
  description = "Nom de l'instance Cloud SQL"
  type        = string
  default     = "mlflow-poc-db"
}

variable "db_name" {
  description = "Nom de la base de données MLflow"
  type        = string
  default     = "mlflow"
}

variable "db_user" {
  description = "Utilisateur PostgreSQL pour MLflow"
  type        = string
  default     = "mlflow"
}

variable "db_password" {
  description = "Mot de passe de l'utilisateur MLflow"
  type        = string
  sensitive   = true
}

variable "network_name" {
  description = "Nom du VPC réseau GCP"
  type        = string
  default     = "default"
}

variable "cloudrun_image_name" {
  description = "Nom du service Cloud Run et du repository Artifact Registry"
  type        = string
  default     = "mlflow"
}

variable "mlflow_admin_username" {
  description = "Nom d'utilisateur administrateur MLflow (basic auth)"
  type        = string
  default     = "admin"
}

variable "mlflow_admin_password" {
  description = "Mot de passe administrateur MLflow (basic auth)"
  type        = string
  sensitive   = true
}

variable "mlflow_flask_secret_key" {
  description = "Clé secrète Flask pour la protection CSRF (basic auth)"
  type        = string
  sensitive   = true
}

variable "cloudrun_api_image_name" {
  description = "Nom du service Cloud Run pour la FastAPI"
  type        = string
  default     = "triage-api"
}

variable "hf_token" {
  description = "Hugging Face Token pour télécharger le modèle dans la VM"
  type        = string
  sensitive   = true
  default     = ""
}
