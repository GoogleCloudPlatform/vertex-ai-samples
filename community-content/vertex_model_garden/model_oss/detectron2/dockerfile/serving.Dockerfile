# Dockerfile for Detectron2 serving.
#
# To build:
# docker build -f model_oss/detectron2/dockerfile/serving.Dockerfile . -t ${YOUR_IMAGE_TAG}
#
# To push to gcr:
# docker tag ${YOUR_IMAGE_TAG} gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}
# docker push gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}
FROM pytorch/torchserve:0.7.0-cpu

USER root

# Install tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        wget \
        vim

# run and update some basic packages software packages, including security libs
RUN apt-get update &&  apt-get install -y \
    software-properties-common && \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get update && apt-get install -y \
    gcc-9 g++-9 apt-transport-https ca-certificates gnupg curl

# Install gcloud tools for gsutil as well as debugging
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | \
    tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update -y && apt-get install google-cloud-sdk -y

USER model-server

# install detectron2 dependencies
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --user numpy==1.24.2
RUN python3 -m pip install --user opencv-python==4.7.0.72
RUN python3 -m pip install --user 'git+https://github.com/facebookresearch/detectron2.git@v0.6'

# Install GCS storage library.
RUN pip install google-cloud-storage==2.6.0

# For mask encoding.
RUN pip install --upgrade pycocotools==2.0.6

ARG MODEL_NAME=detectron2_serving
ENV MODEL_NAME="${MODEL_NAME}"

# health and prediction listener ports
ARG AIP_HTTP_PORT=7080
ENV AIP_HTTP_PORT="${AIP_HTTP_PORT}"

ARG MODEL_MGMT_PORT=7081

# expose health and prediction listener ports from the image
EXPOSE "${AIP_HTTP_PORT}"
EXPOSE "${MODEL_MGMT_PORT}"
EXPOSE 8080 8081 8082 7070 7071

# create torchserve configuration file
USER root
RUN echo "service_envelope=json\n" \
    "inference_address=http://0.0.0.0:${AIP_HTTP_PORT}\n"  \
    "management_address=http://0.0.0.0:${MODEL_MGMT_PORT}" >> /home/model-server/config.properties
USER model-server

# Copy license.
RUN wget https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/main/LICENSE

# Copy model artifacts.
COPY ./model_oss/detectron2/handler.py /home/model-server/handler.py
WORKDIR /home/model-server/

# Create model archive file packaging model artifacts and dependencies.
# Note(lavrai): The model `.pth` file and `cfg.yaml` file will be set by the
# customer as an environment variable and will be later loaded by the
# `handler.py` file.
RUN torch-model-archiver \
  --model-name="${MODEL_NAME}" \
  --version=1.0 \
  --handler=/home/model-server/handler.py \
  --export-path=/home/model-server/model-store \
  -f

# run Torchserve HTTP serve to respond to prediction requests
CMD ["ls", "-ltr", "/home/model-server/model-store/", ";", \
    "torchserve", "--start", "--ts-config=/home/model-server/config.properties", \
    "--models", "${MODEL_NAME}=${MODEL_NAME}.mar", \
    "--model-store", "/home/model-server/model-store"]