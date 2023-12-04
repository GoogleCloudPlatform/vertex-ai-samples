# Dockerfile for vLLM serving.
#
# To build:
# docker build -f model_oss/vllm/dockerfile/serve.Dockerfile . -t ${YOUR_IMAGE_TAG}
#
# To push to gcr:
# docker tag ${YOUR_IMAGE_TAG} gcr.io/{YOUR_PROJECT}/${YOUR_IMAGE_TAG}
# docker push gcr.io/{YOUR_PROJECT}/${YOUR_IMAGE_TAG}

# The base image is required by vllm
# https://vllm.readthedocs.io/en/latest/getting_started/installation.html
FROM nvcr.io/nvidia/pytorch:22.12-py3

USER root

# Install tools.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -y --no-install-recommends curl
RUN apt-get install -y --no-install-recommends wget
RUN apt-get install -y --no-install-recommends git
RUN apt-get install -y --no-install-recommends jq
RUN apt-get install -y --no-install-recommends gnupg

# Copy license.
RUN wget https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/main/LICENSE

# Install libraries.
ENV PIP_ROOT_USER_ACTION=ignore
RUN python3 -m pip install --upgrade pip
RUN pip install google-cloud-storage==2.7.0
RUN pip install absl-py==1.4.0

# Install pytorch
RUN pip install --upgrade torch==2.0.1

# Install vllm deps.
RUN pip install xformers==0.0.20
RUN pip install ninja==1.11.1
RUN pip install psutil==5.9.5
RUN pip install ray==2.6.2
RUN pip install sentencepiece==0.1.99
RUN pip install fastapi==0.100.1
RUN pip install uvicorn==0.23.2
RUN pip install pydantic==1.10.12

# Install transformers from source.
WORKDIR /workspace
RUN git clone https://github.com/huggingface/transformers.git
WORKDIR transformers
# Pin the commit to add-code-llama at 08/25/2023
RUN git reset --hard 015f8e110d270a0ad42de4ae5b98198d69eb1964
RUN pip install -e .
WORKDIR /workspace

# Install vllm from source.
RUN git clone https://github.com/vllm-project/vllm.git
WORKDIR vllm
# Pin the version to a fixed git commit on 08/16/2023.
RUN git reset --hard d1744376ae9fdbfa6a2dc763e1c67309e138fa3d
# Apply a patch to vllm source:
# 1) For models on Huggingface hub: if the model has multiple bin files, each
#    bin file is downloaded separately and gets deleted after loading to GPU
# 2) For models on GCS bucket: each model bin files is download separately
#    and gets deleted after loading to GPU.
# 3) Support code-llama model loading.
COPY model_oss/vllm/vllm.patch /tmp/vllm.patch
RUN git apply /tmp/vllm.patch
RUN pip install -e .

# Expose port 7080 for host serving.
EXPOSE 7080