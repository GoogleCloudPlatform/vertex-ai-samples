# Dockerfile for PEFT Serving.
#
# To build:
# docker build -f model_oss/peft/dockerfile/serve.Dockerfile . -t ${YOUR_IMAGE_TAG}
#
# To push to gcr:
# docker tag ${YOUR_IMAGE_TAG} gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}
# docker push gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}

FROM pytorch/torchserve:0.11.0-gpu

USER root

ENV INFER_PORT=7080
ENV MNG_PORT=7081
ENV MODEL="peft_serving"
ENV PATH="/home/model-server/:${PATH}"

RUN apt-get update && apt-get -y upgrade && apt-get install -y --no-install-recommends \
        curl \
        wget \
        vim \
        git \
        git-lfs
RUN git lfs install
RUN apt-get autoremove -y

# Install libraries.
ENV PIP_ROOT_USER_ACTION=ignore
RUN python3 -m pip install --upgrade pip
RUN pip install --upgrade torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
RUN pip install torchvision==0.15.2
RUN pip install tokenizers==0.13.3
RUN pip install accelerate==0.21.0
RUN pip install sentencepiece==0.1.99
RUN pip install grpcio-status==1.33.2
RUN pip install protobuf==3.19.6
RUN pip install peft==0.5.0
RUN pip install datasets==2.14.4
RUN pip install triton==3.0.0
RUN pip install xformers==0.0.20
RUN pip install google-cloud-storage
RUN pip install absl-py
RUN pip install scipy==1.10.1
RUN pip install evaluate==0.4.0
RUN pip install scikit-learn==1.2.2
RUN pip install loralib==0.1.1
RUN pip install bitsandbytes==0.39.0
RUN pip install trl==0.4.4
RUN pip install einops==0.6.1
RUN pip install optimum==1.13.2
RUN pip install auto-gptq==0.4.2
RUN pip install https://github.com/casper-hansen/AutoAWQ/releases/download/v0.1.7/autoawq-0.1.7+cu118-cp39-cp39-linux_x86_64.whl
RUN pip install diffusers==0.27.2
RUN pip install tiktoken==0.6.0
RUn pip install git+https://github.com/huggingface/transformers.git@76fa17c1663a0efeca7208c20579833365584889
RUN pip install pynvml==11.4.0
RUN pip install -i https://test.pypi.org/simple/ bitsandbytes

# Copy license.
WORKDIR /home/model-server
RUN wget https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/main/LICENSE

# Copy model artifacts.
COPY model_oss/peft/handler.py /home/model-server/handler.py
COPY model_oss/peft/config.properties /home/model-server/config.properties
COPY model_oss/util/ /home/model-server/util/
COPY model_oss/util/pytorch_startup_prober.sh /model_garden/scripts/pytorch_startup_prober.sh
ENV PYTHONPATH /home/model-server/

# Expose ports.
EXPOSE ${INFER_PORT}
EXPOSE ${MNG_PORT}

# Set environments.
ENV TASK "causal-language-modeling-lora"
ENV BASE_MODEL_ID ""
ENV MODEL_ID ""
ENV PRECISION_LOADING_MODE "float16"
ENV FINETUNED_LORA_MODEL_PATH ""
ENV TRUST_REMOTE_CODE ""

# Archive model artifacts and dependencies.
# Do not set --model-file and --serialized-file because model and checkpoint
# will be dynamically loaded in handler.py.
RUN torch-model-archiver \
--model-name=${MODEL} \
--version=1.0 \
--handler=/home/model-server/handler.py \
--runtime=python3 \
--export-path=/home/model-server/model-store \
--archive-format=default \
--force

# Run Torchserve HTTP serve to respond to prediction requests.
CMD ["torchserve", "--start", \
     "--ts-config", "/home/model-server/config.properties", \
     "--models", "${MODEL}=${MODEL}.mar", \
     "--model-store", "/home/model-server/model-store"]
