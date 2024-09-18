# Dockerfile for PEFT Training.
#
# To build:
# docker build -f model_oss/peft/dockerfile/train.Dockerfile . -t ${YOUR_IMAGE_TAG}
#
# To push to gcr:
# docker tag ${YOUR_IMAGE_TAG} gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}
# docker push gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}

# Builds GPU docker image of PyTorch
# Uses multi-staged approach to reduce size
# Stage 1
# Use base conda image to reduce time
FROM continuumio/miniconda3:latest AS compile-image
# Specify py version
ENV PYTHON_VERSION=3.8
# Install apt libs - copied from https://github.com/huggingface/accelerate/blob/main/docker/accelerate-gpu/Dockerfile
RUN apt-get update && \
    apt-get install -y curl git wget software-properties-common git-lfs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists*

# Install audio-related libraries
RUN apt-get update && \
    apt install -y ffmpeg

RUN apt install -y libsndfile1-dev
RUN git lfs install

# Create our conda env - copied from https://github.com/huggingface/accelerate/blob/main/docker/accelerate-gpu/Dockerfile
RUN conda create --name peft python=${PYTHON_VERSION} ipython jupyter pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Below is copied from https://github.com/huggingface/accelerate/blob/main/docker/accelerate-gpu/Dockerfile
# We don't install pytorch here yet since CUDA isn't available
# instead we use the direct torch wheel
ENV PATH /opt/conda/envs/peft/bin:$PATH
# Activate our bash shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]
# Activate the conda env and install transformers + accelerate from source
RUN source activate peft
RUN python3 -m pip install --no-cache-dir git+https://github.com/huggingface/transformers
RUN python3 -m pip install --no-cache-dir git+https://github.com/huggingface/accelerate
RUN python3 -m pip install --no-cache-dir git+https://github.com/huggingface/peft#egg=peft[test]
RUN python3 -m pip install --no-cache-dir bitsandbytes

# Stage 2
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04 AS build-image
COPY --from=compile-image /opt/conda /opt/conda
ENV PATH /opt/conda/bin:$PATH

# Install apt libs
RUN apt-get update && \
    apt-get install -y curl git wget vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists*

# Copy license.
RUN wget https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/main/LICENSE

RUN echo "source activate peft" >> ~/.profile

# Install libraries.
RUN pip install --upgrade torch==2.0.1
RUN pip install torchvision==0.15.2
RUN pip install git+https://github.com/huggingface/transformers@de9255de27abfcae4a1f816b904915f0b1e23cd9
RUN pip install transformers -U
RUN pip install accelerate==0.21.0
RUN pip install sentencepiece==0.1.99
RUN pip install grpcio-status==1.33.2
RUN pip install protobuf==3.19.6
RUN python3 -m pip install --no-cache-dir git+https://github.com/huggingface/peft.git
RUN pip install datasets==2.9.0
RUN pip install triton==2.0.0.dev20221120
RUN pip install xformers==0.0.20
RUN pip install Jinja2==3.1.2
RUN pip install ftfy==6.1.1
RUN pip install cloudml-hypertune==0.1.0.dev6
RUN pip install tensorboard==2.12.0
RUN pip install scipy==1.10.1
RUN pip install evaluate==0.4.0
RUN pip install scikit-learn==1.2.2
RUN pip install loralib==0.1.1
RUN pip install bitsandbytes==0.39.0
RUN pip install trl==0.4.4
RUN pip install einops==0.6.1
RUN pip install google-cloud-storage==2.7.0

RUN git clone --depth 1 --branch v0.16.1 https://github.com/huggingface/diffusers.git
WORKDIR diffusers
RUN pip install -e .

# Switch to diffusers examples folder.
WORKDIR examples

# NOTE: use 'sed' to modify train_text_to_image_lora.py to
#       fix the bug for accelerator.
RUN sed -i \
"s#logging_dir=logging_dir#project_dir=logging_dir#g" \
text_to_image/train_text_to_image_lora.py

# Config accelerate.
RUN mkdir -p ./vertex_vision_model_garden_peft/
COPY model_oss/peft/train.sh ./vertex_vision_model_garden_peft/train.sh
COPY model_oss/peft/*.py     ./vertex_vision_model_garden_peft/
COPY model_oss/util /diffusers/examples/util
ENV PYTHONPATH /diffusers/examples/

# Generate accelerate config at the beginning of docker run.
ENTRYPOINT ["python3", "vertex_vision_model_garden_peft/main.py"]
