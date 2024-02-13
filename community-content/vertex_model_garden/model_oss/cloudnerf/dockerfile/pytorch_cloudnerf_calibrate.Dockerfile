# Dockerfile for ZipNeRF COLMAP image calibration.
#
# To build:
# docker build -f model_oss/cloudnerf/dockerfile/cloudnerf_pytorch_calibrate.Dockerfile . -t ${YOUR_IMAGE_TAG}
#
# To push to gcr:
# docker tag ${YOUR_IMAGE_TAG} gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}
# docker push gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}

FROM us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-cloudnerf-base:20231206_0923_RC00

COPY model_oss/cloudnerf/local_colmap_and_resize.sh /workspace/zipnerf-pytorch/scripts/local_colmap_and_resize.sh

WORKDIR /workspace/zipnerf-pytorch/

ENTRYPOINT ["bash","scripts/local_colmap_and_resize.sh"]