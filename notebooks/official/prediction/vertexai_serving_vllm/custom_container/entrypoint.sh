#!/bin/bash

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

readonly LOCAL_MODEL_DIR=${LOCAL_MODEL_DIR:-"/tmp/model_dir"}
readonly LOCAL_LORA_DIR=${LOCAL_LORA_DIR:-"/tmp/lora_adapters"}

download_model_from_gcs() {
    gcs_uri=$1
    mkdir -p $LOCAL_MODEL_DIR
    echo "Downloading model from $gcs_uri to local directory..."
    if gcloud storage cp -r "$gcs_uri/*" "$LOCAL_MODEL_DIR"; then
      echo "Model downloaded successfully to ${LOCAL_MODEL_DIR}."
    else
      echo "Failed to download model from Cloud Storage: $gcs_uri." >&2
      exit 1
    fi
}


download_lora_adapters_from_gcs() {
    gcs_uri=$1
    local_dir=$2
    mkdir -p "$local_dir"
    echo "Downloading LoRA adapters from $gcs_uri to $local_dir..."
    if gcloud storage cp -r "$gcs_uri/*" "$local_dir"; then
      echo "LoRA adapters downloaded successfully to ${local_dir}."
    else
      echo "Failed to download LoRA adapters from Cloud Storage: $gcs_uri." >&2
      exit 1
    fi
}

updated_args=()
model_arg="--model="
lora_modules_arg="--lora-modules="
gcs_protocol="gs://"
lora_counter=0

for a in "$@"; do
    if [[ $a == $model_arg* ]]; then
        model_path=${a#*=}
        echo "Found model: $model_path"
        if [[ $model_path == $gcs_protocol* ]]; then
            download_model_from_gcs $model_path
            updated_args+=("--model=${LOCAL_MODEL_DIR}")
        else
            updated_args+=("--model=${model_path}")
        fi
    elif [[ $a == $lora_modules_arg* ]]; then
        lora_spec=${a#*=}
        echo "Found LoRA module: $lora_spec"

        # LoRA modules can be in format "name=path" or just "path"
        if [[ $lora_spec == *"="* ]]; then
            lora_name=${lora_spec%%=*}
            lora_path=${lora_spec#*=}
        else
            lora_name="lora_$lora_counter"
            lora_path=$lora_spec
            ((lora_counter++))
        fi

        if [[ $lora_path == $gcs_protocol* ]]; then
            local_lora_path="${LOCAL_LORA_DIR}/${lora_name}"
            download_lora_adapters_from_gcs "$lora_path" "$local_lora_path"
            updated_args+=("--lora-modules=${lora_name}=${local_lora_path}")
        else
            updated_args+=("$a")
        fi
    else
        updated_args+=("$a")
    fi
done

echo "Launch command: " "${updated_args[@]}"
exec "${updated_args[@]}"