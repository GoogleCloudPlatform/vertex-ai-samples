#!/bin/bash

# Run copybara first:
# cloud/ml/applications/vision/model_garden/copybara/run_copybara_local.sh
# Run docker build:
# cloud/ml/applications/vision/model_garden/model_oss/peft/train/axolotl/scripts/build_train_docker.sh

set -x

COPYBARA_DIR="/tmp/train_docker/"

pushd "${COPYBARA_DIR}"

PROJECT="cloud-nas-260507"
IMAGE_TAG="gcr.io/${PROJECT}/axolotl-train:${USER}-test"

docker build -f model_oss/peft/train/axolotl/dockerfile/train.Dockerfile . -t "${IMAGE_TAG}"
docker push "${IMAGE_TAG}"

popd
