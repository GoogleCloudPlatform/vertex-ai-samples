# Serving models in Vertex AI using vLLM

The repository has customization required for serving open models on Vertex AI using [vLLM](https://github.com/vllm-project/vllm.git).

## Using TPU
This [colab notebook](vertexai_serving_vllm_tpu_llama3_2_3B.ipynb) shows how Llama 3.2 3B model can be deployed (downloaded from Hugging Face) to Vertex AI Endpoint using this repository on TPUs.  
This [colab notebook](vertexai_serving_vllm_tpu_gcs_llama3_2_3B.ipynb) shows how Llama 3.2 3B model can be deployed (downloaded from Google Cloud Storage) to Vertex AI Endpoint using this repository on TPUs.

## Using GPU
This [colab notebook](vertexai_serving_vllm_gpu_llama3_2_3B.ipynb) shows how Llama 3.2 3B model can be deployed (downloaded from Hugging Face) to Vertex AI Endpoint using this repository on GPUs.

## Using CPU
This [colab notebook](vertexai_serving_vllm_cpu_llama3_2_3B.ipynb) shows how Llama 3.2 3B model can be deployed (downloaded from Hugging Face) to Vertex AI Endpoint using this repository on CPUs.

## Creating vLLM Container Image

Clone the repository to local disk.

```
git clone https://github.com/GoogleCloudPlatform/vertex-ai-samples.git
```

## TPU
Run the following command to build vLLM container image TPU. Replace `vertexai-vllm-tpu` with actual image tag, which could be something like `us-central1-docker.pkg.dev/my-gcp-project/my-docker-repo/vllm-gcp-tpu`.

```
cd vertex-ai-samples/notebooks/official/prediction/vertexai_serving_vllm/docker && docker build . --file Dockerfile.tpu --tag vertexai-vllm-tpu
```

## GPU
Run the following command to build vLLM container image for GPU. Replace `vertexai-vllm-gpu` with actual image tag, which could be something like `us-central1-docker.pkg.dev/my-gcp-project/my-docker-repo/vllm-gcp-gpu`.

```
cd vertex-ai-samples/notebooks/official/prediction/vertexai_serving_vllm/docker && docker build . --file Dockerfile.gpu --tag vertexai-vllm-gpu
```

## CPU
Run the following command to build vLLM container image for CPU. Replace `vertexai-vllm-cpu` with actual image tag, which could be something like `us-central1-docker.pkg.dev/my-gcp-project/my-docker-repo/vllm-gcp-cpu`.

```
cd vertex-ai-samples/notebooks/official/prediction/vertexai_serving_vllm/docker && docker build . --file Dockerfile.cpu --tag vertexai-vllm-cpu
```
