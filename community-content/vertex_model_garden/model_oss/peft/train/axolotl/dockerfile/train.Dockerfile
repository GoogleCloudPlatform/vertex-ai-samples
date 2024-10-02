# Dockerfile for axolotl training.
#
# To build:
# docker build -f model_oss/peft/train/axolotol/dockerfile/train.Dockerfile . -t ${YOUR_IMAGE_TAG}
#
# To push to gcr:
# docker tag ${YOUR_IMAGE_TAG} gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}
# docker push gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}

FROM winglian/axolotl:main-latest

RUN mkdir -p ./vertex_vision_model_garden/

COPY model_oss/peft/train/axolotl/*.py ./vertex_vision_model_garden/

ENTRYPOINT ["python3", "./vertex_vision_model_garden/train_entrypoint.py"]
