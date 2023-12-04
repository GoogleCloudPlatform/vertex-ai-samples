FROM gcr.io/automl-migration-test/movinet-base:latest

# Copy license.
RUN wget https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/main/LICENSE

RUN wget https://raw.githubusercontent.com/tensorflow/models/954dd73bffd43174bd3ca26a4a34abebe4147570/official/projects/movinet/tools/export_saved_model.py \
  -O /usr/local/lib/python3.9/dist-packages/official/projects/movinet/tools/export_saved_model.py

WORKDIR /automl_vision

ENV PYTHONPATH "${PYTHONPATH}:/automl_vision/util"

ENTRYPOINT ["python3", "-m", "official.projects.movinet.tools.export_saved_model"]
