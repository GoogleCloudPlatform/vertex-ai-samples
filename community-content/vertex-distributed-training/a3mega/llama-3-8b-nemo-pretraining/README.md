# Vertex AI Training: Llama 3.1 8B pre-training using Nvidia A3 Mega VMs (H100)
This document provides a step-by-step guide for pre-training a Llama 3.1 8B model on the `en-wiki` dataset using multiple [Vertex AI Custom Training](https://cloud.google.com/vertex-ai/docs/training/overview) `a3-megagpu-8g` nodes.

We will use a custom container based on NVIDIA's [NeMo Framework](https://docs.nvidia.com/nemo-framework/user-guide/24.07/overview.html) to demonstrate a scalable, multi-node training workflow. All required artifacts and commands are included.

## 1. Prerequisites

### 1.1. Google Cloud Project setup
- **Enable APIs:** Ensure the Vertex AI API is [enabled for your project](http://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com).
- **H100 Mega Quota:** A3 Mega VMs are powered by H100 GPUs. Request quota for `custom_model_training_nvidia_h100_mega_gpus` in one of the [supported regions](https://cloud.google.com/vertex-ai/docs/general/locations#accelerator_support). If using Spot VMs, request `custom_model_training_preemptible_nvidia_h100_mega_gpus` quota instead.
- **Reservations (Optional but recommended):** For guaranteed capacity, [create a reservation](https://cloud.google.com/compute/docs/instances/reservations-shared) and ensure the reservation is shared with the Vertex AI service account. This guide requires a minimum of **16 H100 GPUs** (2 full A3 Mega nodes).

### 1.2. GCS bucket
Create a [Cloud Storage bucket](https://cloud.google.com/storage/docs/creating-buckets) in the same region where you have quota. If you're using Hierarchical Namespace for your bucket, you may need to update permissions of the Vertex AI Custom Code Service Agent .

This bucket is used for:
- Staging the training application.
- Storing model checkpoints and logs.
- Storing data if you use your own data.


## 2. Setup & configuration

### 2.1. Clone the repo
First clone the repo into your development environment.

```bash
git clone https://github.com/GoogleCloudPlatform/vertex-ai-samples.git
```

Navigate to the root folder for this sample.

### 2.2. Environment Setup
First, configure your local environment. These variables are used in subsequent commands.

```bash
# Required: Update with your values
export PROJECT_ID="<your-project-id>"
export REPOSITORY="<your-artifact-registry-repo-name>" # e.g., "my-containers"
export BUCKET="<your-gcs-bucket-name>"

# Optional: Change if needed
export REGION="us-central1"

# --- Do not change the lines below ---
export ARTIFACT_REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}"
export REPO_ROOT=$(git rev-parse --show-toplevel)
```

## 3. Build and push a docker container image to Artifact Registry
Normally, you can use any custom training container on Vertex AI Training. In this example you build a NeMo Docker image that is based on the [Nvidia’s NeMo 24.09](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags) image. Use Cloud Build to build and push the container image.

This document picked NeMo as the demonstrating container since it’s a widely adopted GPU LLM training framework providing high performance and versatile training functionalities.

In addition to the base image, some customizations are included to form the final prebuilt image:
- Some dependencies are installed to integrate with Vertex AI Training.
- An entrypoint script that sets up required environments and calls the training job.
- Some patches are applied to the NeMo code to let it  load the dataset from a GCS bucket.

Run this command to build the container and push the container into the Google Artifact Registry.

```bash
cd "${REPO_ROOT}/community-content/vertex-distributed-training/a3mega/llama-3-8b-nemo-pretraining"
export IMAGE_NAME="vertex-nemo-llama"
gcloud builds submit . \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --config=docker/cloudbuild.yml \
    --substitutions="_ARTIFACT_REGISTRY=${ARTIFACT_REGISTRY},_IMAGE_NAME=${IMAGE_NAME}" \
    --timeout="2h" \
    --machine-type="e2-highcpu-32"
```

## 4. Launch the Training Job


###  4.1. Job Configuration File
Once the container is built, update the job_config.json to set up the training job.
File: job_config.json
```json
{
  "project_id": "<project-id>",
  "region": "<region>",
  "zone": "<zone if using reservation>",
  "bucket": "<bucket>",
  "dataset_bucket": "github-repo/data/third-party/enwiki-latest-pages-articles",
  "image_uri": "<docker image uri from artifact registry>",
  "strategy": "spot",
  "nodes": "2",
  "machine_type": "a3-megagpu-8g",
  "gpu_type": "NVIDIA_H100_MEGA_80GB",
  "gpus_per_node": "8",
  "recipe_name": "llama3_1_8b_pretrain_a3mega",
  "job_prefix": "vertex-spot-",
  "reservation_name": ""
}
```

### 4.2 Launch the Training Job

First, create a Python virtual environment using your tool of choice, then install
the requirements specified in `requirements.txt`. Using `pip`, the command would be:
```bash
pip install -r requirements.txt
```

Now launch the Vertex AI training job using the provided Python script.

```bash
python3 scripts/launch.py --config_file=job_config.json
```

This script reads job_config.json, defines the cluster specification (2 nodes, 8 GPUs each), and submits the custom training job to Vertex AI.

## 5. Monitor and Clean Up

### 5.1. Monitoring
Vertex AI Console: Track the job's status in the Google Cloud Console under Vertex AI > Training > Custom Jobs.
Logs: View detailed logs in Cloud Logging by filtering for your job name.
Checkpoints: Model checkpoints are saved to your GCS bucket at the path specified in your training script's configuration.

### 5.2. Cleaning Up
To avoid ongoing charges, delete the resources you created:
- The Artifact Registry image.
- The contents of the GCS bucket (checkpoints, logs).
- The Vertex AI Custom Job will eventually complete or fail, incurring no further cost.
