# Dockerfile for AutoML vision training dockers with tfvision.
#
# To build:
# docker build -f model_oss/tfvision/dockerfile/train_oss_v2.Dockerfile . -t ${YOUR_IMAGE_TAG}
#
# To push to gcr:
# docker tag ${YOUR_IMAGE_TAG} gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}
# docker push gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}

FROM us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/tfvision-base-v2:latest

# Copy license.
RUN wget https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/main/LICENSE

# Fix yolo and retinanet issues.
RUN mkdir /tmp/buffer && \
    cd /tmp/buffer && \
    git clone https://github.com/tensorflow/models.git && \
    cd models && \
    # Add support for newly added config options.
    git reset --hard ed6d4d220b86237980d3f7563d261d19e040ef1a && \
    cd /usr/local/lib/python3.9/dist-packages/official/projects/yolo && \
    cp /tmp/buffer/models/official/projects/yolo/dataloaders/yolo_input.py ./dataloaders/ && \
    cp /tmp/buffer/models/official/projects/yolo/optimization/optimizer_factory.py ./optimization/ && \
    cp /tmp/buffer/models/official/projects/yolo/configs/yolo.py ./configs && \
    cp /tmp/buffer/models/official/projects/yolo/modeling/factory.py ./modeling && \
    cp /tmp/buffer/models/official/projects/yolo/modeling/layers/detection_generator.py ./modeling/layers && \
    cp /tmp/buffer/models/official/projects/yolo/common/registry_imports.py ./common && \
    cp /tmp/buffer/models/official/projects/yolo/configs/yolov7.py ./configs && \
    cp /tmp/buffer/models/official/projects/yolo/configs/decoders.py ./configs && \
    cp /tmp/buffer/models/official/projects/yolo/configs/backbones.py ./configs && \
    cp /tmp/buffer/models/official/projects/yolo/modeling/yolov7_model.py ./modeling && \
    cp /tmp/buffer/models/official/projects/yolo/modeling/backbones/yolov7.py ./modeling/backbones && \
    cp /tmp/buffer/models/official/projects/yolo/modeling/decoders/yolov7.py ./modeling/decoders && \
    cp /tmp/buffer/models/official/projects/yolo/modeling/heads/yolov7_head.py ./modeling/heads && \
    cp /tmp/buffer/models/official/projects/yolo/modeling/layers/nn_blocks.py ./modeling/layers && \
    cp /tmp/buffer/models/official/projects/yolo/losses/yolov7_loss.py ./losses && \
    cp /tmp/buffer/models/official/projects/yolo/tasks/yolov7.py ./tasks && \
    cp /tmp/buffer/models/official/projects/yolo/ops/initializer_ops.py ./ops && \
    cp /tmp/buffer/models/official/projects/yolo/ops/mosaic.py ./ops && \
    cd /usr/local/lib/python3.9/dist-packages/official/vision && \
    cp /tmp/buffer/models/official/vision/configs/retinanet.py ./configs && \
    cp /tmp/buffer/models/official/vision/modeling/layers/detection_generator.py ./modeling/layers && \
    cp /tmp/buffer/models/official/vision/modeling/layers/edgetpu.py ./modeling/layers && \
    cp /tmp/buffer/models/official/vision/ops/augment.py ./ops && \
    rm -rf /tmp/buffer

# Add MaxViT
RUN mkdir /tmp/buffer && \
    cd /tmp/buffer && \
    git clone https://github.com/tensorflow/models.git && \
    cd models && \
    git reset --hard 6138633a41097a3c0f320bd895ac5da65c33016f && \
    cd /usr/local/lib/python3.9/dist-packages/official/projects/ && \
    cp -R /tmp/buffer/models/official/projects/maxvit/ ./ && \
    rm -rf /tmp/buffer
ENV ENABLE_MAX_VIT "True"


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
