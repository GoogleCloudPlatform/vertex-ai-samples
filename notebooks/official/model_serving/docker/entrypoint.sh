#!/bin/bash

set -e

readonly LOCAL_MODEL_DIR=${LOCAL_MODEL_DIR:-"/tmp/model_dir"}

download_model_from_gcs() {
    gcs_uri=$1
    mkdir -p $LOCAL_MODEL_DIR
    echo "Downloading model from $gcs_uri to local directory..."
    if gcloud storage cp -r "$gcs_uri/*" "$LOCAL_MODEL_DIR"; then
      echo "Model downloaded successfully to ${LOCAL_MODEL_DIR}."
    else
      echo "Failed to download model from Cloud Storage: $gcs_uri."
      exit 1
    fi
}


updated_args=()
model_arg="--model="
gcs_protocol="gs://"
for a in "$@"; do
    if [[ $a == $model_arg* ]]; then
        model_path=$(cut -d'=' -f2 <<<"$a")
        if [[ $model_path == $gcs_protocol* ]]; then
            download_model_from_gcs $model_path
            updated_args+=("--model=${LOCAL_MODEL_DIR}")
        else
            updated_args+=("--model=${model_path}")
        fi
        
    else
        updated_args+=("$a")
    fi
done

echo "Launch command: " "${updated_args[@]}"
exec "${updated_args[@]}"