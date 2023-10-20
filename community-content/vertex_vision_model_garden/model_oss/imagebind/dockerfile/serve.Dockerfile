# Dockerfile for the serving docker for ImageBind.
#
# To build:
# docker build -f model_oss/imagebind/dockerfile/serve.Dockerfile . -t ${YOUR_IMAGE_TAG}
#
# To push to gcr:
# docker tag ${YOUR_IMAGE_TAG} gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}
# docker push gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}

FROM pytorch/torchserve:0.7.0-gpu

USER root

ENV infer_port=7080
ENV mng_port=7081
ENV model_name="imagebind_serving"
ENV PATH="/home/model-server/:${PATH}"

# Install tools.
RUN apt-get update && apt-get -y upgrade && apt-get install -y --no-install-recommends \
        curl \
        wget \
        vim \
        git \
        libgeos-dev

# Install libraries.
ENV PIP_ROOT_USER_ACTION=ignore
RUN python3 -m pip install --upgrade pip
RUN pip install absl-py==1.4.0
RUN pip install google-cloud-storage==2.7.0

# Install ImageBind and dependencies.
RUN git clone https://github.com/facebookresearch/ImageBind.git
WORKDIR ImageBind
# Pin the commit at 07/14/2023.
RUN git reset --hard 95d27c7fd5a8362f3527e176c3a80ae5a4d880c0
# Modify tokenizer file path from ImageBind repo to work with the server.
RUN sed -i '25d' imagebind/data.py
RUN sed -i '25 i\BPE_PATH = "/home/model-server/ImageBind/bpe/bpe_simple_vocab_16e6.txt.gz"' imagebind/data.py
RUN pip install .

WORKDIR /home/model-server

# Copy license.
RUN wget https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/main/LICENSE

# Copy model artifacts.
COPY model_oss/imagebind/handler.py /home/model-server/handler.py
COPY model_oss/imagebind/config.properties /home/model-server/config.properties
COPY model_oss/util/ /home/model-server/util/
ENV PYTHONPATH /home/model-server/

# Expose ports.
EXPOSE ${infer_port}
EXPOSE ${mng_port}

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