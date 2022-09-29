FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-12

RUN apt-get update && \
    apt-get install -y curl gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && \
    apt-get update -y && \
    apt-get install google-cloud-sdk -y

COPY . /trainer-torchrun-gpu

WORKDIR /trainer-torchrun-gpu

RUN pip install -r requirements.txt
#RUN export CURRENT_TIME=$(date "+%Y.%m.%d-%H.%M.%S")
#RUN export MASTER_ADDR="${MASTER_ADDR:-localhost}:${MASTER_PORT:-2222}"
#RUN export RUN_ID=$MASTER_ADDR-$CURRENT_TIME

#RUN echo "${MASTER_ADDR}"
#RUN echo "${RUN_ID}"

#--nnodes=MIN_SIZE:MAX_SIZE
#--nproc_per_node=TRAINERS_PER_NODE

#ENTRYPOINT ["torchrun", "--rdzv_id=${RUN_ID}", "--rdzv_backend=c10d", "--rdzv_endpoint=${MASTER_ADDR}", "main.py"]
#ENTRYPOINT ["sh", "main.sh"]
ENTRYPOINT ["torchrun", "main.py"]