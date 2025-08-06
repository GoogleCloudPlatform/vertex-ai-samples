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