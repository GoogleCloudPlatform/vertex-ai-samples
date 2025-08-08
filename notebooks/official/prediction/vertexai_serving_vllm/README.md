<!---
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
-->

# Serving models in Vertex AI using vLLM

The notebooks in this directory demonstrate how Llama 3.2 3B open weight model can be served on Vertex AI using [vLLM](https://github.com/vllm-project/vllm.git).

## Using TPU
This [colab notebook](vertexai_serving_vllm_tpu_llama3_2_3B.ipynb) shows how Llama 3.2 3B model can be deployed (downloaded from Hugging Face) to Vertex AI Endpoint using this repository on TPUs.

This [colab notebook](vertexai_serving_vllm_tpu_gcs_llama3_2_3B.ipynb) shows how Llama 3.2 3B model can be deployed (downloaded from Google Cloud Storage) to Vertex AI Endpoint using this repository on TPUs.

## Using GPU
This [colab notebook](vertexai_serving_vllm_gpu_llama3_2_3B.ipynb) shows how Llama 3.2 3B model can be deployed (downloaded from Hugging Face) to Vertex AI Endpoint using this repository on GPUs.

## Using CPU
This [colab notebook](vertexai_serving_vllm_cpu_llama3_2_3B.ipynb) shows how Llama 3.2 3B model can be deployed (downloaded from Hugging Face) to Vertex AI Endpoint using this repository on CPUs.
