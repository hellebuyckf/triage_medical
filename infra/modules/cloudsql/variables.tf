variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
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
