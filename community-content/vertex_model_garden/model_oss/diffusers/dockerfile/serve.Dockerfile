# Dockerfile for Diffuser Serving.
#
# To build:
# docker build -f model_oss/diffusers/dockerfile/serve.Dockerfile . -t ${YOUR_IMAGE_TAG}
#
# To push to gcr:
# docker tag ${YOUR_IMAGE_TAG} gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}
# docker push gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}

FROM pytorch/torchserve:0.7.0-gpu

USER root

ENV infer_port=7080
ENV mng_port=7081
ENV model_name="diffusers_serving"
ENV PATH="/home/model-server/:${PATH}"

# Install libraries.
ENV PIP_ROOT_USER_ACTION=ignore
RUN python3 -m pip install --upgrade pip
RUN pip install torch==1.13.1
RUN pip install torchvision==0.14.1
RUN pip install transformers==4.27.4
RUN pip install datasets==2.9.0
RUN pip install accelerate==0.17.0
RUN pip install triton==2.0.0.dev20221120
RUN pip install xformers==0.0.16
RUN pip install google-cloud-storage==2.7.0
RUN pip install imageio[ffmpeg]==2.31.0
RUN pip install absl-py==1.4.0

# Copy LICENSE file
RUN apt-get update && apt-get install wget
RUN wget https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/main/LICENSE

# Install diffusers from main branch source code with a pinned commit.
RUN git clone --depth 1 --branch v0.18.1 https://github.com/huggingface/diffusers.git
WORKDIR diffusers
RUN pip install -e .

# Copy model artifacts.
COPY model_oss/diffusers/handler.py /home/model-server/handler.py
COPY model_oss/util/ /home/model-server/util/
ENV PYTHONPATH /home/model-server/

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
