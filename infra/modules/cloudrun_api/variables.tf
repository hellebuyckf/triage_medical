variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
}

variable "network_name" {
  description = "VPC network name"
  type        = string
}

variable "image_name" {
  description = "Nom du service Cloud Run et repository Artifact Registry"
  type        = string
}

variable "vllm_api_base_url" {
  description = "URL of the vLLM API on the Compute Engine instance"
  type        = string
}

variable "vllm_api_key" {
  description = "API Key to authenticate with vLLM"
  type        = string
  sensitive   = true
}
