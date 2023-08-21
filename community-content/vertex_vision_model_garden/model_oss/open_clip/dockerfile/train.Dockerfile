# Dockerfile for training dockers with OpenCLIP.
#
# To build:
# docker build -f model_oss/open_clilp/dockerfile/train.Dockerfile . -t ${YOUR_IMAGE_TAG}
#
# To push to gcr:
# docker tag ${YOUR_IMAGE_TAG} gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}
# docker push gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}

FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

# Install tools.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -y --no-install-recommends curl
RUN apt-get install -y --no-install-recommends wget
RUN apt-get install -y --no-install-recommends git
RUN apt-get install -y --no-install-recommends jq
RUN apt-get install -y --no-install-recommends gnupg
RUN apt-get install -y --no-install-recommends build-essential

ENV PIP_ROOT_USER_ACTION=ignore
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        wget \
        vim

# Copy license.
RUN wget https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/main/LICENSE

# Prepare artifacts.
WORKDIR /workspace
RUN git clone --branch main https://github.com/mlfoundations/open_clip.git
WORKDIR ./open_clip
RUN git reset --hard 67e5e5ec8741281eb9b30f640c26f91c666308b7

# Install libraries.
RUN pip install webdataset==0.2.5
RUN pip install regex==2023.6.3
RUN pip install ftfy==6.1.1
RUN pip install pandas==2.0.3
RUN pip install braceexpand==0.1.7
RUN pip install huggingface_hub==0.16.4
RUN pip install transformers==4.31.0
RUN pip install timm==0.9.2
RUN pip install fsspec==2023.6.0
RUN pip install sentencepiece==0.1.99
RUN pip install protobuf==3.20.3
RUN pip install tensorboard==2.12.2

# Switch work folder for training.
WORKDIR ./src
