FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

# Install basic libs
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
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
        software-properties-common \
        cuda-toolkit \
        libcudnn8 \
        apt-transport-https

RUN apt install -y --no-install-recommends python3.10 \
        python3.10-venv \
        python3.10-dev \
        python3-pip

Run apt-get autoremove -y

RUN pip install --upgrade pip
RUN pip install --upgrade --ignore-installed \
        "jax[cuda12]==0.4.26" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
        numpy==1.26.4 \
        paxml==1.4.0 \
        praxis==1.4.0 \
        jaxlib==0.4.26 \
        pandas==2.1.4 \
        einshape==1.0.0 \
        utilsforecast==0.1.10 \
        huggingface_hub[cli]==0.23.0 \
        google-cloud-aiplatform[prediction]==1.51.0 \
        fastapi==0.109.1 \
        flask==3.0.3 \
        smart_open[gcs]==7.0.4 \
        protobuf==3.19.6 \
        scikit-learn==1.0.2 \
        timesfm==1.0.1

# Download license.
RUN wget https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/main/LICENSE

# Move scaffold.
COPY model_oss/timesfm/main.py /app/main.py
COPY model_oss/timesfm/predictor.py /app/predictor.py

WORKDIR ..

# Spin off inference server.
CMD ["python3", "/app/main.py"]
