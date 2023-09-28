# Dockerfile for AutoML vision training dockers with tfvision.
#
# To build:
# docker build -f model_oss/tfvision/dockerfile/train_oss.Dockerfile . -t ${YOUR_IMAGE_TAG}
#
# To push to gcr:
# docker tag ${YOUR_IMAGE_TAG} gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}
# docker push gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}

FROM us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/tfvision-base:latest

# Copy license.
RUN wget https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/main/LICENSE

# Fix yolo and retinanet issues.
RUN mkdir /tmp/buffer && \
    cd /tmp/buffer && \
    git clone https://github.com/tensorflow/models.git && \
    cd models && \
    git checkout fbd4c57fd7e9f7d73da30ed3fc755b8c4c682df7 && \
    cd /usr/local/lib/python3.8/dist-packages/official/projects/yolo && \
    cp /tmp/buffer/models/official/projects/yolo/optimization/optimizer_factory.py ./optimization/ && \
    cp /tmp/buffer/models/official/projects/yolo/configs/yolo.py ./configs && \
    cp /tmp/buffer/models/official/projects/yolo/modeling/factory.py ./modeling && \
    cp /tmp/buffer/models/official/projects/yolo/modeling/layers/detection_generator.py ./modeling/layers && \
    cd /usr/local/lib/python3.8/dist-packages/official/vision && \
    cp /tmp/buffer/models/official/vision/configs/retinanet.py ./configs && \
    cp /tmp/buffer/models/official/vision/modeling/layers/detection_generator.py ./modeling/layers && \
    cp /tmp/buffer/models/official/vision/modeling/layers/edgetpu.py ./modeling/layers && \
    rm -rf /tmp/buffer

COPY model_oss/tfvision /automl_vision/tfvision
COPY model_oss/util /automl_vision/util
RUN rm -rf /automl_vision/tfvision/serving

WORKDIR /automl_vision

ENV PYTHONPATH "${PYTHONPATH}:/automl_vision/util"

# Run pylint to validate code.
COPY .pylintrc /automl_vision/.pylintrc
RUN find . -type f -name "*.py" | xargs pylint --rcfile=./.pylintrc --errors-only

ENTRYPOINT ["python3","tfvision/train_hpt_oss.py"]

CMD ["--experiment=YOUR_EXPERIMENT",\
"--config_file=",\
"--mode=YOUR_MODE",\
"--model_dir=YOUR_MODEL_DIR",\
"--objective=YOUR_OBJECTIVE",\
"--learning_rate=",\
"--anchor_size="]
