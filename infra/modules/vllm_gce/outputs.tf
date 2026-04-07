output "internal_ip" {
  description = "Internal IP address of the vLLM VM"
  value       = google_compute_instance.vllm.network_interface.0.network_ip
}

output "instance_name" {
  description = "Name of the Compute Engine instance"
  value       = google_compute_instance.vllm.name
}
