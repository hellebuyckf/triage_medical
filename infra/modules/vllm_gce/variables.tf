variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
}

variable "zone" {
  description = "GCP zone for the VM"
  type        = string
}

variable "network_name" {
  description = "VPC network name"
  type        = string
}

variable "hf_token" {
  description = "Hugging Face Token"
  type        = string
  sensitive   = true
}

variable "model_id" {
  description = "HF Model ID to run"
  type        = string
  default     = "FrancoisFormation/qwen3-triage-dpo"
}
