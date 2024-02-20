# Dockerfile for PEFT Serving.
#
# To build:
# docker build -f model_oss/peft/dockerfile/serve.Dockerfile . -t ${YOUR_IMAGE_TAG}
#
# To push to gcr:
# docker tag ${YOUR_IMAGE_TAG} gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}
# docker push gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}

FROM pytorch/torchserve:0.7.0-gpu

USER root

ENV infer_port=7080
ENV mng_port=7081
ENV model_name="peft_serving"
ENV PATH="/home/model-server/:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        wget \
        vim \
        git \
        git-lfs
RUN git lfs install

# Install libraries.
ENV PIP_ROOT_USER_ACTION=ignore
RUN python3 -m pip install --upgrade pip
RUN pip install --upgrade torch==2.0.1
RUN pip install torchvision==0.15.2
RUN pip install tokenizers==0.13.3
RUN pip install accelerate==0.21.0
RUN pip install sentencepiece==0.1.99
RUN pip install grpcio-status==1.33.2
RUN pip install protobuf==3.19.6
RUN python3 -m pip install --no-cache-dir git+https://github.com/huggingface/peft.git
RUN pip install datasets==2.14.4
RUN pip install triton==2.0.0.dev20221120
RUN pip install xformers==0.0.20
RUN pip install google-cloud-storage==2.7.0
RUN pip install absl-py==1.4.0
RUN pip install scipy==1.10.1
RUN pip install evaluate==0.4.0
RUN pip install scikit-learn==1.2.2
RUN pip install loralib==0.1.1
RUN pip install bitsandbytes==0.39.0
RUN pip install trl==0.4.4
RUN pip install einops==0.6.1

# Install diffusers from source.
RUN git clone --depth 1 --branch v0.16.1 https://github.com/huggingface/diffusers.git
WORKDIR diffusers
RUN pip install -e .
WORKDIR /home/model-server

# Install transformers from source.
RUN git clone --depth 1 --branch v4.31.0 https://github.com/huggingface/transformers.git
# The patch is used to change the transformers loading model behavior:
# 1) For models on Huggingface hub: if the model has multiple shards, each shard
#    will be downloaded separately and get deleted after loading to GPU.
# 2) For models on local disk: if a model bin file is actually a text file
#    recording a GCS path, the model file will be downloaded and get deleted
#    after loading to GPU.
COPY model_oss/peft/hf_transformers_lazy_download.patch /home/model-server/hf_transformers_lazy_download.patch
WORKDIR transformers
RUN git apply /home/model-server/hf_transformers_lazy_download.patch
RUN pip install -e .
WORKDIR /home/model-server

# Copy license.
RUN wget https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/main/LICENSE

# Copy model artifacts.
COPY model_oss/peft/handler.py /home/model-server/handler.py
COPY model_oss/peft/config.properties /home/model-server/config.properties
COPY model_oss/util/ /home/model-server/util/
ENV PYTHONPATH /home/model-server/

# Expose ports.
EXPOSE ${infer_port}
EXPOSE ${mng_port}

# Set environments.
ENV TASK "causal-language-modeling-lora"
ENV MODEL_ID "openlm-research/open_llama_7b"
ENV PRECISION_LOADING_MODE "float16"
ENV FINETUNED_LORA_MODEL_PATH ""


# Archive model artifacts and dependencies.
# Do not set --model-file and --serialized-file because model and checkpoint
# will be dynamically loaded in handler.py.
RUN torch-model-archiver \
--model-name=${model_name} \
--version=1.0 \
--handler=/home/model-server/handler.py \
--runtime=python3 \
--export-path=/home/model-server/model-store \
--archive-format=default \
--force

# Run Torchserve HTTP serve to respond to prediction requests.
CMD ["torchserve", "--start", \
     "--ts-config", "/home/model-server/config.properties", \
     "--models", "${model_name}=${model_name}.mar", \
     "--model-store", "/home/model-server/model-store"]
