# HIPAA-Hardened Infrastructure for Vertex AI
# This Terraform script sets up the security perimeter and encryption.

provider "google" {
  project = var.project_id
  region  = var.region
}

# 1. Cloud Audit Logs (Data Access)
resource "google_project_iam_audit_config" "audit_logs" {
  project = var.project_id
  service = "allServices"
  audit_log_config {
    log_type = "DATA_READ"
  }
  audit_log_config {
    log_type = "DATA_WRITE"
  }
}

# 2. CMEK (Customer-Managed Encryption Keys)
resource "google_kms_key_ring" "vertex_keyring" {
  name     = "vertex-ai-keyring"
  location = var.region
}

resource "google_kms_crypto_key" "vertex_key" {
  name     = "vertex-ai-key"
  key_ring = google_kms_key_ring.vertex_keyring.id
  purpose  = "ENCRYPT_DECRYPT"
}

# 3. VPC Service Controls (Simplified Perimeter)
# Note: Usually managed at the Organization level, but included here for completeness.
resource "google_access_context_manager_service_perimeter" "perimeter" {
  parent = "accessPolicies/${var.access_policy_id}"
  name   = "accessPolicies/${var.access_policy_id}/servicePerimeters/vertex_perimeter"
  title  = "Vertex AI HIPAA Perimeter"
  
  status {
    restricted_services = ["aiplatform.googleapis.com", "dlp.googleapis.com"]
    resources           = ["projects/${var.project_number}"]
  }
}

variable "project_id" {}
variable "project_number" {}
variable "region" { default = "us-central1" }
variable "access_policy_id" {}
