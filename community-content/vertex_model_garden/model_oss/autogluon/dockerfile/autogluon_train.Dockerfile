# Dockerfile for training dockers with Autogluon.
#
# To build:
# docker build -f model_oss/autogluon/dockerfile/train.Dockerfile . -t ${YOUR_IMAGE_TAG}
#
# To push to gcr:
# docker tag ${YOUR_IMAGE_TAG} gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}
# docker push gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}

FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# Install tools.
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_ROOT_USER_ACTION=ignore

RUN apt-get update && apt-get -y upgrade && apt-get install -y --no-install-recommends \
apt-utils \
curl \
wget \
git \
jq \
gnupg \
build-essential \
tesseract-ocr \
vim

# Install libraries.
RUN pip install autogluon==1.0.0

COPY model_oss/autogluon /autogluon
WORKDIR /autogluon

# Copy license.
RUN wget https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/main/LICENSE

ENTRYPOINT ["python", "train.py"]
