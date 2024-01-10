# Dockerfile for lm-evaluation-harness evaluation.
#
# To build:
# docker build -f model_oss/lm-evaluation-harness/dockerfile/eval.Dockerfile . -t ${YOUR_IMAGE_TAG}
#
# To push to gcr:
# docker tag ${YOUR_IMAGE_TAG} gcr.io/{YOUR_PROJECT}/${YOUR_IMAGE_TAG}
# docker push gcr.io/{YOUR_PROJECT}/${YOUR_IMAGE_TAG}

FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

USER root

# Install tools.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -y --no-install-recommends curl
RUN apt-get install -y --no-install-recommends wget
RUN apt-get install -y --no-install-recommends git

# Copy license.
RUN wget https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/main/LICENSE

# Install libraries.
ENV PIP_ROOT_USER_ACTION=ignore
RUN python3 -m pip install --upgrade pip
RUN pip install google-cloud-storage==2.7.0
RUN pip install absl-py==1.4.0

# Install lm-evaluation-harness
RUN git clone https://github.com/EleutherAI/lm-evaluation-harness
WORKDIR lm-evaluation-harness
# Pin version up to date 08/08/2023
RUN git reset --hard b952a206de210b72b1bf750fbab38c26121e0dc0
# Edit tokenizer loading function to avoid using fast tokenizer for OpenLLaMA
RUN sed -i '355 i\        use_fast = not pretrained.startswith("openlm-research/open_llama")' lm_eval/models/huggingface.py
RUN sed -i '360 i\            use_fast=use_fast,' lm_eval/models/huggingface.py
# Install from source while including the sentencepiece dependency
RUN pip install -e ".[sentencepiece]"
