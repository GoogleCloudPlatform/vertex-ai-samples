FROM pytorch/torchserve:0.9.0-gpu

USER root

# Install tools.
RUN apt-get update -y --allow-releaseinfo-change && apt-get -y upgrade && apt-get install -y --no-install-recommends \
        curl \
        wget \
        vim \
        git

# Copy license.
RUN wget https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/main/LICENSE

ENV INFER_PORT=7080
ENV MNG_PORT=7081
ENV MODEL_NAME="llava_serving"
ENV PATH="/home/model-server/:${PATH}"
ENV PATH="/usr/local/cuda-12.1/bin:${PATH}"
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
ENV NVIDIA_VISIBLE_DEVICES=all

# Get 'LLaVA' repository from github.
RUN git clone https://github.com/haotian-liu/LLaVA /home/model-server/LLaVA
WORKDIR /home/model-server/LLaVA
# Using git reset command to pin it down to a specific version.
RUN git reset --hard 7775b12d6b20cd69089be7a18ea02615a59621cd

# Install the package.
RUN python3 -m pip install --upgrade pip
RUN pip install google-cloud-storage==2.13.0
RUN pip install absl-py==2.0.0
RUN pip install -e .

# Copy model artifacts.
COPY model_oss/llava/handler.py /home/model-server/handler.py
COPY model_oss/llava/model_handler_setup.py /home/model-server/model_handler_setup.py
COPY model_oss/util/ /home/model-server/util/
ENV PYTHONPATH /home/model-server
WORKDIR /home/model-server

# Create torchserve configuration file.
RUN echo \
    "default_response_timeout=1800\n" \
    "service_envelope=json\n" \
    "inference_address=http://0.0.0.0:${INFER_PORT}\n"  \
    "management_address=http://0.0.0.0:${MNG_PORT}\n"  \
    "default_workers_per_model=DEFAULT_WORKERS_PER_MODEL" >> /home/model-server/config.properties

# Expose ports.
EXPOSE ${INFER_PORT}
EXPOSE ${MNG_PORT}

# Archive model artifacts and dependencies.
# Do not set --model-file and --serialized-file because model and checkpoint will be dynamically loaded in handler.py.
RUN torch-model-archiver \
--model-name=${MODEL_NAME} \
--version=1.0 \
--handler=/home/model-server/handler.py \
--runtime=python3 \
--export-path=/home/model-server/model-store \
--archive-format=default \
--force

# Run Torchserve HTTP serve to respond to prediction requests.
# Use $NUM_GPU workers unless overriden by $TS_NUM_WORKERS
CMD ["TOTAL=$(nvidia-smi", "--list-gpus","|","wc","-l)","&&", "TS_NUM_WORKERS=${TS_NUM_WORKERS:-$TOTAL}","&&", "sed","-i","\"s/DEFAULT_WORKERS_PER_MODEL/$TS_NUM_WORKERS/g\"","/home/model-server/config.properties", "&&", \
    "torchserve", "--start", \
     "--ts-config", "/home/model-server/config.properties", \
     "--models", "${MODEL_NAME}=${MODEL_NAME}.mar", \
     "--model-store", "/home/model-server/model-store"]
