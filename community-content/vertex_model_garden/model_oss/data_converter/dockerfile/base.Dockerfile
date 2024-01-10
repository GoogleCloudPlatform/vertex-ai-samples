FROM python:3.9

ENV DEBIAN_FRONTEND=noninteractive

# Install basic libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        curl \
        wget \
        sudo \
        gnupg \
        python3-opencv \
        lsb-release \
        ca-certificates \
        build-essential \
        git \
        vim \
        screen \
        libportaudio2 \
        libusb-1.0-0-dev \
        openjdk-17-jre

# Add gcsfuse distribution URL as a package source and import its public key.
RUN echo "deb https://packages.cloud.google.com/apt gcsfuse-`lsb_release -c -s` main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

# Install gcsfuse.
RUN apt-get update && apt-get install -y --no-install-recommends gcsfuse

# Install google cloud SDK.
RUN wget -q https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-359.0.0-linux-x86_64.tar.gz
RUN tar xzf google-cloud-sdk-359.0.0-linux-x86_64.tar.gz
RUN ./google-cloud-sdk/install.sh -q
# Make sure gsutil will use the default service account.
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg


# Install required libs.
RUN pip install --upgrade pip
RUN pip install pyyaml==5.4.1
RUN pip install pycocotools==2.0.6
RUN pip install opencv-python-headless==4.7.0.72
RUN pip install numpy==1.24.2
RUN pip install pandas==1.5.3
RUN pip install Pillow==9.4.0
RUN pip install apache-beam[gcp]==2.45.0
RUN pip install object-detection==0.0.3
RUN pip install google-cloud-storage==1.42.3
RUN pip install gcsfs==2021.10.1
RUN pip install pylint==2.17.2
