# Serving models in Vertex AI using vLLM

The notebooks in this directory demonstrate how Llama 3.2 3B open weight model can be served on Vertex AI using [vLLM](https://github.com/vllm-project/vllm.git).

## Using TPU
This [colab notebook](vertexai_serving_vllm_tpu_llama3_2_3B.ipynb) shows how Llama 3.2 3B model can be deployed (downloaded from Hugging Face) to Vertex AI Endpoint using this repository on TPUs.  
This [colab notebook](vertexai_serving_vllm_tpu_gcs_llama3_2_3B.ipynb) shows how Llama 3.2 3B model can be deployed (downloaded from Google Cloud Storage) to Vertex AI Endpoint using this repository on TPUs.

## Using GPU
This [colab notebook](vertexai_serving_vllm_gpu_llama3_2_3B.ipynb) shows how Llama 3.2 3B model can be deployed (downloaded from Hugging Face) to Vertex AI Endpoint using this repository on GPUs.

## Using CPU
This [colab notebook](vertexai_serving_vllm_cpu_llama3_2_3B.ipynb) shows how Llama 3.2 3B model can be deployed (downloaded from Hugging Face) to Vertex AI Endpoint using this repository on CPUs.
