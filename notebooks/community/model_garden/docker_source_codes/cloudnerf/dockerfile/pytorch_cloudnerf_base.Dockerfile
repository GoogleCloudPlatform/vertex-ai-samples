# Dockerfile for ZipNeRF base image.
#
# To build:
# docker build -f model_oss/cloudnerf/dockerfile/pytorch_cloudnerf_base.Dockerfile . -t ${YOUR_IMAGE_TAG}
#
# To push to gcr:
# docker tag ${YOUR_IMAGE_TAG} gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}
# docker push gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

USER root

ARG COLMAP_GIT_COMMIT=main
ARG CUDA_ARCHITECTURES=60;70;75;80;86

# Prevent stop building ubuntu at time zone selection.
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y --allow-releaseinfo-change && apt-get -y upgrade && apt-get install -y --no-install-recommends \
        curl \
        g++ \
        wget \
        vim \
        bash \
        cmake \
        imagemagick \
        ninja-build \
        build-essential \
        libboost-program-options-dev \
        libboost-filesystem-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libeigen3-dev \
        libflann-dev \
        libfreeimage-dev \
        libmetis-dev \
        libgoogle-glog-dev \
        libgtest-dev \
        libsqlite3-dev \
        libglew-dev \
        qtbase5-dev \
        libqt5opengl5-dev \
        libcgal-dev \
        libceres-dev \
        git \
        git-lfs \
        python3-cffi \
        python3-cryptography \
        libffi-dev \
        python-dev

# Copy license.
RUN wget https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/main/LICENSE

# Install google cloud CLI.
RUN wget -q https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-430.0.0-linux-x86.tar.gz
RUN tar xzf google-cloud-cli-430.0.0-linux-x86.tar.gz
RUN ./google-cloud-sdk/install.sh -q
# Make sure gsutil will use the default service account.
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

# Install deps and install gsutil.
RUN pip install gsutil==5.27

# When building colmap in colab, the link error "undefined reference.
# to '_glapi_tls_Current'" happens. A solution is to install "libglvnd"
# as described in this page https://github.com/colmap/colmap/issues/1271.
RUN git clone --depth 1 --branch v1.7.0 https://github.com/NVIDIA/libglvnd && \
    apt-get install -y libxext-dev libx11-dev x11proto-gl-dev && \
    cd libglvnd/  && \
    apt-get install -y autoconf automake libtool && \
    apt-get install -y libffi-dev && \
    ./autogen.sh && \
    ./configure && \
    make  -j4 && \
    make install

RUN apt remove nvidia-cuda-toolkit -y \
    nvidia-cuda-toolkit \
    nvidia-cuda-toolkit-gcc

# Install libraries.
ENV PIP_ROOT_USER_ACTION=ignore
RUN python3 -m pip install --upgrade pip

ENV CUDA_HOME=/usr/local/cuda

RUN git clone --branch main https://github.com/SuLvXiangXin/zipnerf-pytorch.git
# Set current directory to the downloaded 'zipnerf-pytorch' repository.
WORKDIR ./zipnerf-pytorch
# Using git reset command to pin it down to a specific version.
RUN git reset --hard 4de3d21ebb9e15412d36951b56e2d713fddd812b
COPY model_oss/cloudnerf/requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Install gridencoder extensions and nvdiffrast (for textured mesh).
RUN cd .. && \
    TORCH_CUDA_ARCH_LIST="6.0 7.0 7.5 8.0 8.6+PTX" CXX=g++ pip install ./zipnerf-pytorch/gridencoder

# Install cuda version of torch_scatter.
RUN pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
RUN pip install google-cloud-aiplatform==1.25.0
RUN pip install google-cloud-storage==2.9.0

# Build and install COLMAP.
RUN git clone  --depth 1 --branch 3.8 https://github.com/colmap/colmap.git
RUN cd colmap && \
    git fetch https://github.com/colmap/colmap.git ${COLMAP_GIT_COMMIT} && \
    mkdir build && \
    cd build && \
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} && \
    ninja && \
    ninja install && \
    cd .. && rm -rf colmap

RUN git clone  --depth 1 --branch v1.0.2 https://github.com/dranjan/python-plyfile.git

RUN sed -i "20 i\sys.path.append('/workspace/zipnerf-pytorch/internal/pycolmap')" /workspace/zipnerf-pytorch/internal/datasets.py
RUN sed -i "21 i\sys.path.append('/workspace/zipnerf-pytorch/internal/pycolmap/pycolmap')" /workspace/zipnerf-pytorch/internal/datasets.py
