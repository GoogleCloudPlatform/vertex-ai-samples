# Dockerfile for ZipNeRF rendering.
#
# To build:
# docker build -f model_oss/cloudnerf/dockerfile/pytorch_cloudnerf_render.Dockerfile . -t ${YOUR_IMAGE_TAG}
#
# To push to gcr:
# docker tag ${YOUR_IMAGE_TAG} gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}
# docker push gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}

FROM us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-cloudnerf-base:20231206_0923_RC00

COPY model_oss/cloudnerf/render.sh /workspace/zipnerf-pytorch/scripts/render.sh
COPY model_oss/cloudnerf/configs/360.gin /workspace/zipnerf-pytorch/configs/360.gin
COPY model_oss/cloudnerf/configs/360_glo.gin /workspace/zipnerf-pytorch/configs/360_glo.gin
COPY model_oss/cloudnerf/configs/accelerate_config.yaml /root/.cache/huggingface/accelerate/default_config.yaml
RUN sed -i '324s/.*/            keyframe_names = fp.read().splitlines()/' /workspace/zipnerf-pytorch/internal/camera_utils.py

ENV PYTHONPATH "${PYTHONPATH}:/workspace/zipnerf-pytorch/util"

WORKDIR /workspace/zipnerf-pytorch/

ENTRYPOINT ["bash", "scripts/render.sh"]