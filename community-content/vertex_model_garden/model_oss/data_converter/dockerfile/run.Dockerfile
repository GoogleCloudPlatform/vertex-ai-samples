FROM gcr.io/automl-migration-test/automl-vision-data-converter-base:latest

# Copy license.
RUN wget https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/main/LICENSE

COPY model_oss/data_converter /automl_vision/data_converter
COPY model_oss/util /automl_vision/util

WORKDIR /automl_vision
ENV PYTHONPATH "${PYTHONPATH}:/automl_vision"

# Run pylint to validate code.
COPY .pylintrc /automl_vision/.pylintrc
RUN find . -type f -name "*.py" | xargs pylint --rcfile=./.pylintrc --errors-only

ENTRYPOINT ["python3","data_converter/data_converter_main.py"]

CMD ["--input_file_path=YOUR_INPUT_FILE",\
"--input_file_type=csv",\
"--objective=iod",\
"--output_dir=YOUR_OUTPUT_DIR",\
"--num_shard=10,10,10",\
"--split_ratio=0.8,0.1,0.1"]
