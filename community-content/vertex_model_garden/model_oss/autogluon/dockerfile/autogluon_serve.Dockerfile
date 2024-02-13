# Dockerfile for serving dockers with AutoGluon.
#
# To build:
# docker build -f model_oss/autogluon/dockerfile/serve.Dockerfile . -t ${YOUR_IMAGE_TAG}
#
# To push to gcr:
# docker tag ${YOUR_IMAGE_TAG} gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}
# docker push gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}

FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

USER root

# AutoGluon might require libgomp for some dependencies.
RUN apt-get update && apt-get -y upgrade && apt-get install -y --no-install-recommends \
        curl \
        wget \
        vim \
        libgomp1

# Install AutoGluon and other dependencies.
RUN pip install --upgrade pip
RUN pip install autogluon==1.0.0
RUN pip install flask==3.0.0

# Dependencies needed to work with GCS.
RUN pip install absl-py==2.0.0
RUN pip install google-cloud-storage==2.7.0

# Copy scripts into the container.
COPY model_oss/autogluon /autogluon
COPY model_oss/util /autogluon/util
WORKDIR /autogluon

# Copy license.
RUN wget https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/main/LICENSE
RUN wget https://github.com/pallets/flask/blob/main/LICENSE.rst

# Expose the port the app runs on.
EXPOSE 8501

# Set the working directory to a specific path for consistency.
WORKDIR /autogluon

# Change to a non-root user for security purposes.
RUN useradd -m autogluonuser
USER autogluonuser

# Run Flask application.
CMD ["python", "serve.py"]
