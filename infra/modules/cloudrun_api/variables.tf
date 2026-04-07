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
  description = "Cloud Run service name and Artifact Registry repo name"
  type        = string
}

variable "vllm_api_base_url" {
  description = "URL of the remote vLLM API"
  type        = string
}

variable "vllm_api_key" {
  description = "API Key for vLLM"
  type        = string
  sensitive   = true
}

variable "model_id" {
  description = "Model ID used by vLLM (must match the one on GCE)"
  type        = string
  default     = "FrancoisFormation/qwen3-triage-dpo"
}
