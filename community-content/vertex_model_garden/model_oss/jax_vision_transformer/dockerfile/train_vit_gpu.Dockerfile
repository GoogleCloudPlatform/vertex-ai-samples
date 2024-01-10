# This Dockerfile runs the JAX based Vision transformer training on GPU.
# See https://github.com/google-research/vision_transformer#running-on-cloud
# for more details.
# Here is an example to build this dockerfile:
# PROJECT="your gcp project"
# IMAGE_TAG="trainn_vit_gpu:${USER}-test"
# docker build -f model_oss/jax_vision_transformer/dockerfile/train_vit_gpu.Dockerfile . -t "${IMAGE_TAG}"
# docker tag "${IMAGE_TAG}" "gcr.io/${PROJECT}/${IMAGE_TAG}"
# docker push "gcr.io/${PROJECT}/${IMAGE_TAG}"

FROM tensorflow/tensorflow:2.12.0-gpu

ENV DEBIAN_FRONTEND=noninteractive

# Install basic libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        curl \
        wget \
        sudo \
        gnupg \
        libsm6 \
        libxext6 \
        libxrender-dev \
        lsb-release \
        ca-certificates \
        build-essential \
        git

# Copy Apache license.
RUN wget https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/main/LICENSE

# Get 'vision_transformer' repository from github.
RUN git clone https://github.com/google-research/vision_transformer
# Ser current directory to the downloaded 'vision_transformer' repository.
WORKDIR ./vision_transformer
# Using git reset command to pin it down to a specific version.
RUN git reset --hard e66b4732d44504251197a3da3f5949f3f3ce9ca6

# Install required libs
RUN pip install --upgrade pip
# The following pip installs are pinned down versions of those inside
# vit_jax/requirements.txt file.
# NOTE: Using `no-deps` flag to avoid overwriting of
# dependent library versions. For example,
# both `chex` and `jax` can overwrite each others
# `jax-lib` version.
RUN pip install --no-deps absl-py==1.4.0
RUN pip install --no-deps aqtp==0.0.10
RUN pip install --no-deps array-record==0.2.0
RUN pip install --no-deps astunparse==1.6.3
RUN pip install --no-deps cached-property==1.5.2
RUN pip install --no-deps cachetools==5.3.0
RUN pip install --no-deps certifi==2019.11.28
RUN pip install --no-deps chardet==3.0.4
RUN pip install --no-deps chex==0.1.7
RUN pip install --no-deps click==8.1.3
RUN pip install --no-deps cloudpickle==2.2.1
RUN pip install --no-deps clu==0.0.9
RUN pip install --no-deps contextlib2==21.6.0
RUN pip install --no-deps dacite==1.8.1
RUN pip install --no-deps dbus-python==1.2.16
RUN pip install --no-deps decorator==5.1.1
RUN pip install --no-deps dm-tree==0.1.8
RUN pip install --no-deps einops==0.6.1
RUN pip install --no-deps etils==1.3.0
RUN pip install --no-deps flatbuffers==23.3.3
RUN pip install --no-deps flax==0.6.10
RUN pip install --no-deps git+https://github.com/google/flaxformer@9adaa4467cf17703949b9f537c3566b99de1b416
RUN pip install --no-deps gast==0.4.0
RUN pip install --no-deps google-auth==2.16.2
RUN pip install --no-deps google-auth-oauthlib==0.4.6
RUN pip install --no-deps google-pasta==0.2.0
RUN pip install --no-deps googleapis-common-protos==1.59.0
RUN pip install --no-deps grpcio==1.51.3
RUN pip install --no-deps h5py==3.8.0
RUN pip install --no-deps idna==2.8
RUN pip install --no-deps importlib-metadata==6.1.0
RUN pip install --no-deps importlib-resources==5.12.0
RUN pip install --no-deps keras==2.12.0
RUN pip install --no-deps libclang==16.0.0
RUN pip install --no-deps Markdown==3.4.3
RUN pip install --no-deps markdown-it-py==2.2.0
RUN pip install --no-deps MarkupSafe==2.1.2
RUN pip install --no-deps mdurl==0.1.2
RUN pip install --no-deps ml-collections==0.1.1
RUN pip install --no-deps msgpack==1.0.5
RUN pip install --no-deps nest-asyncio==1.5.6
RUN pip install --no-deps numpy==1.23.5
RUN pip install --no-deps oauthlib==3.2.2
RUN pip install --no-deps opt-einsum==3.3.0
RUN pip install --no-deps optax==0.1.5
RUN pip install --no-deps orbax-checkpoint==0.1.6
RUN pip install --no-deps packaging==23.0
RUN pip install --no-deps pandas==2.0.1
RUN pip install --no-deps pip==23.1.2
RUN pip install --no-deps promise==2.3
RUN pip install --no-deps protobuf==4.22.1
RUN pip install --no-deps psutil==5.9.5
RUN pip install --no-deps pyasn1==0.4.8
RUN pip install --no-deps pyasn1-modules==0.2.8
RUN pip install --no-deps Pygments==2.15.1
RUN pip install --no-deps PyGObject==3.36.0
RUN pip install --no-deps python-apt==2.0.1+ubuntu0.20.4.1
RUN pip install --no-deps python-dateutil==2.8.2
RUN pip install --no-deps pytz==2023.3
RUN pip install --no-deps PyYAML==6.0
RUN pip install --no-deps requests==2.22.0
RUN pip install --no-deps requests-oauthlib==1.3.1
RUN pip install --no-deps requests-unixsocket==0.2.0
RUN pip install --no-deps rich==13.3.5
RUN pip install --no-deps rsa==4.9
RUN pip install --no-deps scipy==1.10.1
RUN pip install --no-deps setuptools==67.6.0
RUN pip install --no-deps six==1.14.0
RUN pip install --no-deps tensorboard==2.12.0
RUN pip install --no-deps tensorboard-data-server==0.7.0
RUN pip install --no-deps tensorboard-plugin-wit==1.8.1
RUN pip install --no-deps tensorflow==2.12.0
RUN pip install --no-deps tensorflow-cpu==2.12.0
RUN pip install --no-deps tensorflow-datasets==4.9.2
RUN pip install --no-deps tensorflow-estimator==2.12.0
RUN pip install --no-deps tensorflow-hub==0.13.0
RUN pip install --no-deps tensorflow-io-gcs-filesystem==0.31.0
RUN pip install --no-deps tensorflow-metadata==1.13.1
RUN pip install --no-deps tensorflow-probability==0.20.0
RUN pip install --no-deps tensorflow-text==2.12.1
RUN pip install --no-deps tensorstore==0.1.36
RUN pip install --no-deps termcolor==2.2.0
RUN pip install --no-deps toml==0.10.2
RUN pip install --no-deps toolz==0.12.0
RUN pip install --no-deps tqdm==4.65.0
RUN pip install --no-deps typing_extensions==4.5.0
RUN pip install --no-deps tzdata==2023.3
RUN pip install --no-deps urllib3==1.25.8
RUN pip install --no-deps Werkzeug==2.2.3
RUN pip install --no-deps wheel==0.40.0
RUN pip install --no-deps wrapt==1.14.1
RUN pip install --no-deps zipp==3.15.0
# Installing jax at the very end with GPU support.
# NOTE: Not using `no-deps` flag here because
# we need CUDA support.
RUN pip install jax[cuda11_cudnn82]==0.4.6 \
--find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

COPY ./model_oss/jax_vision_transformer/vit_config_without_data.py vit_jax/configs/vit.py

ENV PYTHONPATH ./vit_jax
ENTRYPOINT ["python", "-m", "vit_jax.main"]