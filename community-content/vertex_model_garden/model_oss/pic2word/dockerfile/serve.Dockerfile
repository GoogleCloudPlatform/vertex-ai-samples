FROM pytorch/torchserve:0.7.1-gpu

USER root

ENV infer_port=7080
ENV mng_port=7081
ENV model_name="pic2word"
ENV PATH="/home/model-server/:${PATH}"
ENV PYTHONPATH="$PYTHONPATH:/home/model-server/composed_image_retrieval:/home/model-server/composed_image_retrieval/src:/home/model-server"

# Copy license.
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget

RUN wget https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/main/LICENSE

# Install dependencies.
ENV PIP_ROOT_USER_ACTION=ignore
RUN python3 -m pip install --upgrade pip
RUN pip install google-cloud-storage==2.7.0
RUN pip install open_clip_torch==2.20.0
RUN pip install numpy==1.22.0
RUN pip install scikit-image==0.21.0
RUN pip install scikit-learn==1.0.2
RUN pip install torch==2.0.0
RUN pip install torchvision==0.15.2
RUN pip install tensorboard==2.13.0
RUN pip install ase==3.21.1
RUN pip install braceexpand==0.1.7
RUN pip install cached-property==1.5.2
RUN pip install configparser==5.0.2
RUN pip install cycler==0.10.0
RUN pip install decorator==4.4.2
RUN pip install docker-pycreds==0.4.0
RUN pip install gitdb==4.0.7
RUN pip install gitpython==3.1.30
RUN pip install googledrivedownloader==0.4
RUN pip install h5py==3.1.0
RUN pip install isodate==0.6.0
RUN pip install jinja2==3.0.1
RUN pip install kiwisolver==1.3.1
RUN pip install littleutils==0.2.2
RUN pip install llvmlite==0.36.0
RUN pip install markupsafe==2.0.1
RUN pip install matplotlib==3.3.4
RUN pip install networkx==2.5.1
RUN pip install numba==0.53.1
RUN pip install ogb==1.3.1
RUN pip install outdated==0.2.1
RUN pip install pathtools==0.1.2
RUN pip install promise==2.3
RUN pip install psutil==5.8.0
RUN pip install pyarrow==4.0.0
RUN pip install pyparsing==2.4.7
RUN pip install python-louvain==0.15
RUN pip install pyyaml==5.4.1
RUN pip install rdflib==5.0.0
RUN pip install sentry-sdk==1.14.0
RUN pip install shortuuid==1.0.1
RUN pip install sklearn==0.0
RUN pip install smmap==4.0.0
RUN pip install subprocess32==3.5.4
RUN pip install torch-geometric==1.7.0
RUN pip install wandb==0.10.30
RUN pip install wilds==1.1.0
RUN pip install ftfy==6.1.1
RUN pip install regex==2023.6.3
RUN pip install webdataset==0.2.48
RUN pip install requests==2.31.0
RUN pip install hydra-core==1.3.2
RUN pip install omegaconf==2.3.0
RUN pip install fairseq==0.10.0
RUN pip install bitarray==2.7.6

# Get 'composed_image_retrieval' repository from github.
RUN git clone https://github.com/google-research/composed_image_retrieval
# Set workdir to composed_image_retrieval.
WORKDIR ./composed_image_retrieval
# Using git reset command to pin it down to a specific version.
RUN git reset --hard 8c053297c2fae9cd17ddcded48445a4f47208dbd

# Fix issue introduced by installing composed_image_retrieval
# https://github.com/huggingface/transformers/issues/8638#issuecomment-790772391
RUN pip uninstall dataclasses -y

# Copy model artifacts.
COPY model_oss/pic2word/handler.py /home/model-server/handler.py
COPY model_oss/util/ /home/model-server/util/

# Create torchserve configuration file.
RUN echo \
    "default_response_timeout=1800\n" \
    "service_envelope=json\n" \
    "inference_address=http://0.0.0.0:${infer_port}\n"  \
    "management_address=http://0.0.0.0:${mng_port}" >> /home/model-server/config.properties

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