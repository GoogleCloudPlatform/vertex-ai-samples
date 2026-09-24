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

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
# Install gcloud SDK
RUN apt-get update && \
    apt-get install -y apt-utils git apt-transport-https gnupg ca-certificates curl \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && apt-get update -y && apt-get install google-cloud-cli -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/vllm

# Copy entrypoint.sh to the container
COPY ./entrypoint.sh /workspace/vllm/vertexai/entrypoint.sh
RUN chmod +x /workspace/vllm/vertexai/entrypoint.sh

ENTRYPOINT ["/workspace/vllm/vertexai/entrypoint.sh"]