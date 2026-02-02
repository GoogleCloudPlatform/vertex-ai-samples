Architecture Overview

  The custom container consists of three main components:

  1. Dockerfile (Dockerfile)

  Builds a custom container image that:
  - Takes a base image (vLLM) via build argument
  - Installs Google Cloud SDK to enable GCS (Google Cloud Storage) access
  - Copies and configures the entrypoint script
  - Sets the entrypoint to /workspace/vllm/vertexai/entrypoint.sh

  2. Entrypoint Script (entrypoint.sh:1-54)

  This is the key component that provides smart model loading:

  What it does:
  - Intercepts all command-line arguments passed to the container
  - Detects if the --model= argument points to a GCS path (starts with gs://)
  - If it's a GCS path:
    - Downloads the model from GCS to /tmp/model_dir using gcloud storage cp
    - Rewrites the --model= argument to point to the local directory
  - If it's a local path, passes it through unchanged
  - Executes the original command with updated arguments

  Key logic (lines 37-51):
  for a in "$@"; do
      if [[ $a == $model_arg* ]]; then  # Detects --model=
          model_path=${a#*=}
          if [[ $model_path == $gcs_protocol* ]]; then  # Is it gs://?
              download_model_from_gcs $model_path
              updated_args+=("--model=${LOCAL_MODEL_DIR}")  # Use local path
          else
              updated_args+=("--model=${model_path}")
          fi
      else
          updated_args+=("$a")
      fi
  done

  3. Cloud Build Configuration (cloudbuild.yaml:15-37)

  Automates the container build and push process:

  Build steps:
  1. If _DEVICE_TYPE=cpu: Clones vLLM repo and builds the CPU base image
  2. Builds the custom container with the entrypoint on top of the base image
  3. Pushes to Artifact Registry at $LOCATION-docker.pkg.dev/$PROJECT_ID/${_REPOSITORY}/vllm-${_DEVICE_TYPE}

  Configurable substitutions:
  - _DEVICE_TYPE: gpu (default) or cpu
  - _BASE_IMAGE: vllm/vllm-openai (default)
  - _REPOSITORY: my-docker-repo (default)

  How It Works End-to-End

  1. Build: Cloud Build creates a container with gcloud SDK + entrypoint script
  2. Deploy: Container deployed to Vertex AI
  3. Runtime: When Vertex AI starts the container with arguments like --model=gs://my-bucket/my-model
  4. Entrypoint intercepts: Downloads the model from GCS to local disk
  5. vLLM starts: With the model loaded from local path instead of GCS

  This design allows you to store models in GCS and have them automatically downloaded at container startup, rather than baking them into the container image.