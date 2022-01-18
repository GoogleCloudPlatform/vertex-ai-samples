#!/usr/bin/env bash
set -e

# Prod (Publicly viewable)
PROJECT=cloud-devrel-public-resources
REPOSITORY=alphafold
LOCAL_IMAGE=alphafold-on-gcp
REMOTE_IMAGE=${LOCAL_IMAGE?}
TAG=latest
REGISTRY="us-west1-docker.pkg.dev/${PROJECT?}/${REPOSITORY?}/${REMOTE_IMAGE?}:${TAG?}"

git clone https://github.com/deepmind/alphafold.git

cp Dockerfile alphafold/docker/Dockerfile
cp AlphaFold.ipynb alphafold/notebooks/AlphaFold.ipynb

cd alphafold && sudo docker build --tag ${LOCAL_IMAGE?}:${TAG?} -f docker/Dockerfile .

sudo docker tag ${LOCAL_IMAGE?}:${TAG?} ${REGISTRY?}
sudo docker push ${REGISTRY?}
