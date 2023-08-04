FROM tensorflow/build:2.12-python3.9

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
        libtcmalloc-minimal4


# Install google cloud CLI.
RUN wget -q https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-430.0.0-linux-x86.tar.gz
RUN tar xzf google-cloud-cli-430.0.0-linux-x86.tar.gz
RUN ./google-cloud-sdk/install.sh -q
# Make sure gsutil will use the default service account.
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg


# Install required libs.
RUN pip install --upgrade pip
RUN pip install cloud-tpu-client==0.10
RUN pip install pyyaml==6.0
RUN pip install fsspec==2023.4.0
RUN pip install gcsfs==2023.4.0
RUN pip install tf-models-official==2.12.0
RUN pip install cloudml-hypertune==0.1.0.dev6
RUN pip install pylint==2.17.3

# Installs Reduction Server NCCL plugin.
RUN echo "deb https://packages.cloud.google.com/apt google-fast-socket main" | tee /etc/apt/sources.list.d/google-fast-socket.list \
&&  curl -s -L https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
&&  apt update && apt install -y google-reduction-server

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
