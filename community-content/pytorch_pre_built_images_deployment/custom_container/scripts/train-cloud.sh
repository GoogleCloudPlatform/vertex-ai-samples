#!/bin/bash
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# This script performs cloud training for a PyTorch model.

echo "Submitting PyTorch model training job to Vertex AI"

# PROJECT_ID: Change to your project id
PROJECT_ID=$(gcloud config list --format 'value(core.project)')

# BUCKET_NAME: Change to your bucket name.
BUCKET_NAME="[your-bucket-name]" # <-- CHANGE TO YOUR BUCKET NAME

# validate bucket name
if [ "${BUCKET_NAME}" = "[your-bucket-name]" ]
then
  echo "[ERROR] INVALID VALUE: Please update the variable BUCKET_NAME with valid Cloud Storage bucket name. Exiting the script..."
  exit 1
fi

# JOB_NAME: the name of your job running on AI Platform.
JOB_PREFIX="finetuned-bert-classifier-pytorch-cstm-cntr"
JOB_NAME=${JOB_PREFIX}-$(date +%Y%m%d%H%M%S)-custom-job

# REGION: select a region from https://cloud.google.com/vertex-ai/docs/general/locations#available_regions
# or use the default '`us-central1`'. The region is where the job will be run.
REGION="us-central1"

# JOB_DIR: Where to store prepared package and upload output model.
JOB_DIR=gs://${BUCKET_NAME}/${JOB_PREFIX}/models/${JOB_NAME}

# IMAGE_REPO_NAME: set a local repo name to distinquish our image
IMAGE_REPO_NAME=pytorch_gpu_train_finetuned-bert-classifier

# IMAGE_URI: the complete URI location for Cloud Container Registry
CUSTOM_TRAIN_IMAGE_URI=gcr.io/${PROJECT_ID}/${IMAGE_REPO_NAME}

# Build the docker image
docker build --no-cache -f Dockerfile -t $CUSTOM_TRAIN_IMAGE_URI ../python_package

# Deploy the docker image to Cloud Container Registry
docker push ${CUSTOM_TRAIN_IMAGE_URI}

# worker pool spec
worker_pool_spec="\
replica-count=1,\
machine-type=n1-standard-8,\
accelerator-type=NVIDIA_TESLA_V100,\
accelerator-count=1,\
container-image-uri=${CUSTOM_TRAIN_IMAGE_URI}"

# Submit Custom Job to Vertex AI
gcloud beta ai custom-jobs create \
    --display-name=${JOB_NAME} \
    --region ${REGION} \
    --worker-pool-spec="${worker_pool_spec}" \
    --args="--model-name","finetuned-bert-classifier","--job-dir",$JOB_DIR

echo "After the job is completed successfully, model files will be saved at $JOB_DIR/"

# uncomment following lines to monitor the job progress by streaming logs

# Stream the logs from the job
# gcloud ai custom-jobs stream-logs $(gcloud ai custom-jobs list --region=$REGION --filter="displayName:"$JOB_NAME --format="get(name)")

# # Verify the model was exported
# echo "Verify the model was exported:"
# gsutil ls ${JOB_DIR}/