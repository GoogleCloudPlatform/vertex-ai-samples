
# Use pytorch GPU base image
# FROM gcr.io/cloud-aiplatform/training/pytorch-gpu.1-7
FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-10:latest

# set working directory
WORKDIR /app

# Install required packages
RUN pip install google-cloud-storage transformers datasets tqdm cloudml-hypertune

# Copies the trainer code to the docker image.
COPY ./trainer/__init__.py /app/trainer/__init__.py
COPY ./trainer/experiment.py /app/trainer/experiment.py
COPY ./trainer/utils.py /app/trainer/utils.py
COPY ./trainer/metadata.py /app/trainer/metadata.py
COPY ./trainer/model.py /app/trainer/model.py
COPY ./trainer/task.py /app/trainer/task.py

# Set up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.task"]
