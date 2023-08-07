FROM gcr.io/automl-migration-test/movinet-base:latest

# Copy license.
RUN wget https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/main/LICENSE

RUN mkdir -p /automl_vision/movinet
COPY model_oss/movinet/*.py /automl_vision/movinet/
COPY model_oss/util /automl_vision/util

WORKDIR /automl_vision

ENV PYTHONPATH "${PYTHONPATH}:/automl_vision/util"

# Run pylint to validate code.
COPY .pylintrc /automl_vision/.pylintrc
RUN find . -type f -name "*.py" | xargs pylint --rcfile=./.pylintrc --errors-only

ENTRYPOINT ["python3","movinet/train.py"]
