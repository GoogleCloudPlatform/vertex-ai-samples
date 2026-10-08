# HIPAA Compliance Guide for Vertex AI Agents

Building healthcare applications with Generative AI requires a multi-layered security approach. This guide outlines how to use the **HIPAA-Hardened Agent Template** to meet regulatory requirements on Google Cloud.

## 1. The Shared Responsibility Model
Google Cloud supports HIPAA compliance (under the [BAA](https://cloud.google.com/security/compliance/hipaa)), but you are responsible for configuring your environment correctly.

## 2. Technical Safeguards
This template implements three critical safeguards:

### A. PHI Redaction (DLP API)
The agent uses the **Cloud Data Loss Prevention (DLP) API** to inspect and redact Protected Health Information (PHI) from both user inputs and model outputs.
- **Input Redaction:** Prevents sensitive data from being stored in model logs or training sets.
- **Output Redaction:** Prevents the LLM from accidentally leaking PHI in its responses.

### B. Encryption at Rest (CMEK)
By default, Google encrypts data at rest. For HIPAA, many organizations require **Customer-Managed Encryption Keys (CMEK)** via Cloud KMS to have full control over the encryption lifecycle.

### C. Network Security (VPC Service Controls)
**VPC Service Controls** create a security perimeter around your Vertex AI and DLP resources, preventing data exfiltration to unauthorized services or external networks.

## 3. Deployment Steps
1. **Sign the BAA:** Ensure your organization has a Business Associate Agreement with Google Cloud.
2. **Deploy Infrastructure:** Use the provided Terraform scripts in `/infra` to set up the perimeter and KMS keys.
3. **Configure DLP:** Adjust the `infoTypes` in `agent/dlp_filter.py` to match your specific compliance needs.
4. **Deploy Agent:** Deploy the Python code as a **Vertex AI Reasoning Engine**.

## 4. Auditing
Enable **Cloud Audit Logs** (Data Access logs) to track who accessed PHI and when redactions occurred. These logs should be exported to a secure BigQuery dataset for long-term retention.
