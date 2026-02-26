#!/usr/bin/env bash

# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

# Default configurations (same as cloudbuild.yaml substitutions)
DEVICE_TYPE=${DEVICE_TYPE:-gpu}
BASE_IMAGE=${BASE_IMAGE:-vllm/vllm-openai}
REPOSITORY=${REPOSITORY:-my-docker-repo}
PROJECT_ID=${PROJECT_ID:-}
LOCATION=${LOCATION:-us-central1}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --device-type)
            DEVICE_TYPE="$2"
            shift 2
            ;;
        --base-image)
            BASE_IMAGE="$2"
            shift 2
            ;;
        --repository)
            REPOSITORY="$2"
            shift 2
            ;;
        --project-id)
            PROJECT_ID="$2"
            shift 2
            ;;
        --location)
            LOCATION="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --device-type    Device type: gpu or cpu (default: gpu)"
            echo "  --base-image     Base Docker image (default: vllm/vllm-openai)"
            echo "  --repository     Artifact Registry repository (default: my-docker-repo)"
            echo "  --project-id     GCP Project ID (required for push)"
            echo "  --location       GCP location (default: us-central1)"
            echo "  --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Build GPU container locally"
            echo "  $0 --device-type gpu"
            echo ""
            echo "  # Build CPU container locally"
            echo "  $0 --device-type cpu"
            echo ""
            echo "  # Build and push to Artifact Registry"
            echo "  $0 --device-type gpu --project-id my-project --location us-central1"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Convert device type to lowercase
device_type=${DEVICE_TYPE,,}
image_name="vllm-${DEVICE_TYPE}"

echo "========================================="
echo "Local Docker Build Configuration"
echo "========================================="
echo "Device Type: $device_type"
echo "Base Image: $BASE_IMAGE"
echo "Image Name: $image_name"
if [[ -n "$PROJECT_ID" ]]; then
    echo "Target: $LOCATION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$image_name"
else
    echo "Target: $image_name (local only)"
fi
echo "========================================="
echo ""

# Handle CPU build - clone and build vLLM base image if needed
if [[ $device_type == "cpu" ]]; then
    echo "Building open source vLLM CPU container image..."
    if [[ ! -d "vllm" ]]; then
        echo "Cloning vLLM repository..."
        git clone --branch v0.5.1 https://github.com/vllm-project/vllm.git --depth 1
    else
        echo "vLLM directory already exists, using existing clone"
    fi

    cd vllm
    echo "Building vLLM CPU base image..."
    DOCKER_BUILDKIT=1 docker build -t "$BASE_IMAGE" -f docker/Dockerfile.cpu .
    cd ..
    echo "vLLM CPU base image built successfully"
    echo ""
fi

# Build the custom container
echo "Building custom container image for: $device_type"
docker build -t "$image_name" --build-arg BASE_IMAGE="$BASE_IMAGE" .

echo ""
echo "Build completed successfully!"
echo "Local image tag: $image_name"

# Optionally push to Artifact Registry
if [[ -n "$PROJECT_ID" ]]; then
    remote_tag="$LOCATION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$image_name"
    echo ""
    echo "Tagging image for Artifact Registry..."
    docker tag "$image_name" "$remote_tag"

    echo "Pushing to Artifact Registry: $remote_tag"
    docker push "$remote_tag"
    echo ""
    echo "Push completed successfully!"
    echo "Remote image: $remote_tag"
else
    echo ""
    echo "Skipping push (no PROJECT_ID provided)"
    echo "To push to Artifact Registry, run with --project-id option"
fi

echo ""
echo "Done!"
