# Dockerfile for basic training dockers with Keras.
#
# To build:
# docker build -f model_oss/keras/dockerfile/train.Dockerfile . -t ${YOUR_IMAGE_TAG}
#
# To push to gcr:
# docker tag ${YOUR_IMAGE_TAG} gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}
# docker push gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}

FROM tensorflow/tensorflow:2.12.0-gpu

ENV DEBIAN_FRONTEND=noninteractive

# This is added to fix docker build error related to Nvidia key update.
RUN rm -f /etc/apt/sources.list.d/cuda.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

# Install basic libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        curl \
        wget \
        sudo \
        gnupg \
        libsm6 \
        libxext6 \
        libxrender-dev \
        lsb-release \
        ca-certificates \
        build-essential \
        git \
        vim \
        screen \
        libtcmalloc-minimal4


# Install google cloud SDK.
RUN wget -q https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-359.0.0-linux-x86_64.tar.gz
RUN tar xzf google-cloud-sdk-359.0.0-linux-x86_64.tar.gz
RUN ./google-cloud-sdk/install.sh -q
# Make sure gsutil will use the default service account.
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg


# Install required libs.
RUN pip install --upgrade pip
RUN pip install cloud-tpu-client==0.10
RUN pip install pyyaml==5.4.1
RUN pip install fsspec==2021.10.1
RUN pip install gcsfs==2021.10.1
RUN pip install tensorflow-text==2.11.0
RUN pip install pyglove==0.1.0
RUN pip install cloudml-hypertune==0.1.0.dev6
RUN pip install pylint==2.17.2
RUN pip install keras-cv==0.4.0
RUN pip install tensorflow-datasets==4.8.3
RUN pip install tensorflow-estimator==2.12.0
RUN pip install tensorflow-gcs-config==2.12.0
RUN pip install tensorflow-hub==0.13.0
RUN pip install tensorflow-io-gcs-filesystem==0.32.0
RUN pip install tensorflow-metadata==1.13.1
RUN pip install tensorflow-probability==0.19.0
RUN pip install tensorboard==2.12.2
RUN pip install tensorboard-data-server==0.7.0
RUN pip install tensorboard-plugin-wit==1.8.1
RUN pip install protobuf==3.20.3
RUN pip install pandas==1.5.3
RUN pip install pandas-datareader==0.10.0
RUN pip install pandas-gbq==0.17.9
RUN pip install pycocotools==2.0.6

# Installs Reduction Server NCCL plugin.
RUN echo "deb https://packages.cloud.google.com/apt google-fast-socket main" | tee /etc/apt/sources.list.d/google-fast-socket.list \
&&  curl -s -L https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
&&  apt update && apt install -y google-reduction-server

RUN wget https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/main/LICENSE

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp

# Lower the memory fragmentation, and speed up the training.
# https://github.com/tensorflow/tensorflow/issues/44176#issuecomment-783768033
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4

# Enable userspace DNS cache
ENV GCS_RESOLVE_REFRESH_SECS=60
ENV GCS_REQUEST_CONNECTION_TIMEOUT_SECS=300
ENV GCS_METADATA_REQUEST_TIMEOUT_SECS=300
ENV GCS_READ_REQUEST_TIMEOUT_SECS=300
ENV GCS_WRITE_REQUEST_TIMEOUT_SECS=600
# Each opened GCS file takes GCS_READ_CACHE_BLOCK_SIZE_MB of RAM, reduce the
# value from the default 64MB to 8MB to decrease memory footprint.
ENV GCS_READ_CACHE_BLOCK_SIZE_MB=8

WORKDIR /usr/local/lib/python3.8/dist-packages/official/vision

COPY model_oss/keras /automl_vision/keras
COPY model_oss/util /automl_vision/util

WORKDIR /automl_vision

# Keras stable diffusion training codes set width and height as RESOLUTION.
ENV RESOLUTION "512"

ENV PYTHONPATH "${PYTHONPATH}:/automl_vision/util"

# Run pylint to validate code.
COPY .pylintrc /automl_vision/.pylintrc
RUN find . -type f -name "*.py" | xargs pylint --rcfile=./.pylintrc --errors-only

ENTRYPOINT ["python3","keras/train.py"]