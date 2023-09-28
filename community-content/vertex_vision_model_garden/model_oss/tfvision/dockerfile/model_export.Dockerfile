# Dockerfile for AutoML vision model export dockers with tfvision.
#
# To build:
# docker build -f model_oss/tfvision/dockerfile/model_export.Dockerfile . -t ${YOUR_IMAGE_TAG}
#
# To push to gcr:
# docker tag ${YOUR_IMAGE_TAG} gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}
# docker push gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}

FROM us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/tfvision-base-v2:latest

# Copy license.
RUN wget https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/main/LICENSE

RUN PROTOC_ZIP=protoc-3.9.2-linux-x86_64.zip && \
  curl -OL https://github.com/google/protobuf/releases/download/v3.9.2/$PROTOC_ZIP && \
  unzip -o $PROTOC_ZIP -d /usr/local bin/protoc && \
  unzip -o $PROTOC_ZIP -d /usr/local include/* && \
  rm -f $PROTOC_ZIP

COPY model_oss/tfvision /automl_vision/tfvision
COPY model_oss/util /automl_vision/util

# Install tensorflow models following:
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md.
# https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/object_detection_tutorial.ipynb.
RUN cd /automl_vision && \
  git clone --depth 1 https://github.com/tensorflow/models && \
  cd models/research && \
  protoc object_detection/protos/*.proto --python_out=. && \
  cp object_detection/packages/tf2/setup.py . && \
  pip install . && \
  cd /automl_vision && \
  rm -rf ./models

RUN pip install tensorflow-io==0.25.0

RUN pip install "opencv-python-headless<4.3"
RUN pip install google-cloud-aiplatform==1.23.0

# Install yolov4, yolov7, and maxvit
RUN mkdir /tmp/buffer && \
    cd /tmp/buffer && \
    git clone https://github.com/tensorflow/models.git && \
    cd models && \
    git reset --hard 6138633a41097a3c0f320bd895ac5da65c33016f && \
    cd /usr/local/lib/python3.9/dist-packages/official/projects/ && \
    cp -R /tmp/buffer/models/official/projects/yolo/ ./ && \
    cp -R /tmp/buffer/models/official/projects/maxvit/ ./ && \
    rm -rf /tmp/buffer

ENV PYTHONPATH "${PYTHONPATH}:/automl_vision/tfvision"

WORKDIR /automl_vision

# Run pylint to validate code.
COPY .pylintrc /automl_vision/.pylintrc
RUN find . -type f -name "*.py" | xargs pylint --rcfile=./.pylintrc --errors-only

ENTRYPOINT ["python3","tfvision/serving/export_oss_saved_model.py"]

CMD ["--experiment=YOUR_EXPERIMENT",\
"--objective=YOUR_OBJECTIVE",\
"--config_file=YOUR_CONFIG_FILE",\
"--checkpoint_path=YOUR_CHECKPOINT_DIR",\
"--label_map_path=YOUR_LABEL_MAP_PATH",\
"--input_image_size=YOUR_INPUT_IMAGE_SIZE",\
"--export_dir=YOUR_EXPORT_DIR"]
