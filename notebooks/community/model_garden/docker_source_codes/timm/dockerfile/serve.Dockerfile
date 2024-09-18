# Dockerfile for serving dockers with timm.
#
# To build:
# docker build -f model_oss/timm/dockerfile/serve.Dockerfile . -t ${YOUR_IMAGE_TAG}
#
# To push to gcr:
# docker tag ${YOUR_IMAGE_TAG} gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}
# docker push gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}

FROM pytorch/torchserve:0.7.0-gpu

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        wget \
        vim

# Copy license.
RUN wget https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/main/LICENSE

ENV infer_port=7080
ENV mng_port=7081
ENV model_name="timm_serving"

# Install timm.
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install timm==0.6.12
RUN python3 -m pip install google-cloud-storage==2.9.0

# Copy model artifacts.
COPY model_oss/timm/handler.py /home/model-server/handler.py

# Create torchserve configuration file.
RUN echo \
    "default_response_timeout=1200\n" \
    "service_envelope=json\n" \
    "inference_address=http://0.0.0.0:${infer_port}\n"  \
    "management_address=http://0.0.0.0:${mng_port}" >> /home/model-server/config.properties

# Expose ports.
EXPOSE ${infer_port}
EXPOSE ${mng_port}

# Archive eager mode model artifacts and dependencies.
# Do not set --model-file and --serialized-file because model and checkpoint will be dynamically loaded in handler.py.
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