FROM gcr.io/automl-migration-test/movinet-base:latest

# Copy license.
RUN wget https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/main/LICENSE

RUN pip install flask==2.3.2
RUN pip install waitress==2.1.2

RUN mkdir -p /automl_vision/movinet/serving
COPY model_oss/movinet/serving /automl_vision/movinet/serving
COPY model_oss/util /automl_vision/util

WORKDIR /automl_vision

ENV PYTHONPATH "${PYTHONPATH}:/automl_vision/util"

ENTRYPOINT ["flask", "--app", "movinet.serving.serving_main", "run"]
CMD ["--host=0.0.0.0", "--port=8501"]
