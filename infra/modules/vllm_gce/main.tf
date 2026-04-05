# Activation de l'API Compute
resource "google_project_service" "compute_api" {
  project            = var.project_id
  service            = "compute.googleapis.com"
  disable_on_destroy = false
}

# Service account for the VM
resource "google_service_account" "vllm_sa" {
  project      = var.project_id
  account_id   = "vllm-gce-sa"
  display_name = "Service Account for vLLM GCE"
}

# Firewall rule to allow port 8000 only from internal VPC
resource "google_compute_firewall" "allow_vllm_internal" {
  name    = "allow-vllm-internal"
  network = var.network_name
  project = var.project_id

  allow {
    protocol = "tcp"
    ports    = ["8000"]
  }

  source_ranges = ["10.0.0.0/8"] # Allow internal VPC ranges
  target_tags   = ["vllm-server"]
}

# Firewall rule for SSH (optional, for debug)
resource "google_compute_firewall" "allow_ssh_iap" {
  name    = "allow-ssh-iap"
  network = var.network_name
  project = var.project_id

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["35.235.240.0/20"] # IAP range
  target_tags   = ["vllm-server"]
}

# The Compute Engine instance
resource "google_compute_instance" "vllm" {
  name         = "vllm-server"
  machine_type = "g2-standard-4"
  zone         = var.zone
  project      = var.project_id

  tags = ["vllm-server"]

  boot_disk {
    initialize_params {
      image = "projects/deeplearning-platform-release/global/images/family/common-cu128-ubuntu-2204-nvidia-570"
      size  = 100
      type  = "pd-ssd"
    }
  }

  network_interface {
    network = var.network_name
    # Ephemeral IP to allow downloading models and docker image.
    # In a prod environment, use Cloud NAT instead of a public IP.
    access_config {}
  }

  guest_accelerator {
    type  = "nvidia-l4"
    count = 1
  }

  # Deep Learning VM requires this to automatically install drivers if needed
  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = true
  }

  service_account {
    email  = google_service_account.vllm_sa.email
    scopes = ["cloud-platform"]
  }

  metadata = {
    install-nvidia-driver = "True"
  }

  metadata_startup_script = <<-EOT
    #!/bin/bash
    echo "Starting vLLM Setup"

    # Install Docker if not present
    if ! command -v docker &> /dev/null; then
      echo "Installing Docker..."
      curl -fsSL https://get.docker.com -o get-docker.sh
      sh get-docker.sh
      systemctl enable docker
      systemctl start docker
    fi

    # Wait for NVIDIA drivers to be ready
    for i in {1..30}; do
      if nvidia-smi; then
        echo "NVIDIA drivers are ready."
        break
      fi
      echo "Waiting for NVIDIA drivers..."
      sleep 10
    done

    # Run vLLM Docker container
    # Exposing on port 8000
    docker run -d --name vllm \
      --runtime nvidia --gpus all \
      -v /var/lib/vllm/cache:/root/.cache/huggingface \
      -p 8000:8000 \
      --ipc=host \
      --restart unless-stopped \
      -e HF_TOKEN="${var.hf_token}" \
      vllm/vllm-openai:v0.4.2 \
      --model "${var.model_id}" \
      --max-model-len 4096 \
      --dtype auto \
      --api-key "${var.hf_token}" # Using hf_token as API key for simplicity in POC
  EOT

  depends_on = [google_project_service.compute_api]
}
