# This Dockerfile trains the F-VLM model on GPU.
# Here is an example to build this dockerfile:
# PROJECT="your gcp project"
# IMAGE_TAG="jax-f-vlm-train:${USER}-test"
# docker build -f model_oss/fvlm/dockerfile/jax_fvlm_train_gpu.Dockerfile . -t "${IMAGE_TAG}"
# docker tag "${IMAGE_TAG}" "gcr.io/${PROJECT}/${IMAGE_TAG}"
# docker push "gcr.io/${PROJECT}/${IMAGE_TAG}"

# See https://cloud.google.com/tensorflow-enterprise/docs/overview for details.
FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-12.py310:m110

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

# Install required libs
RUN pip install --upgrade pip
# The following pip installs are pinned down versions satisfying
# fvlm/requirements.txt file.
# Get F-VLM repository by using git sparse-checkout to avoid downloading entire
# google-research repository.
# Using the commit 6712c224985c694001ba8ee68697bbf4dcb32edb on Jan 4th, 2024.
ARG COMMIT_ID=6712c224985c694001ba8ee68697bbf4dcb32edb
RUN git clone -c \
remote.origin.fetch=+${COMMIT_ID}:refs/remotes/origin/${COMMIT_ID} \
https://github.com/google-research/google-research --no-checkout --progress \
--depth 1
WORKDIR ./google-research
RUN git sparse-checkout init --cone
RUN git sparse-checkout set fvlm
RUN git checkout ${COMMIT_ID}

# Note: The following libraries are pinned down versions of:
# https://github.com/google-research/google-research/blob/master/fvlm/requirements.txt
RUN pip install --no-cache-dir tensorflow==2.12.0
RUN pip install --no-cache-dir tensorflow-datasets==4.9.2
RUN pip install --no-cache-dir numpy==1.23.5
RUN pip install --no-cache-dir torch==2.0.1
RUN pip install --no-cache-dir torchvision==0.15.2
RUN pip install --no-cache-dir opencv-python==4.7.0.72
RUN pip install --no-cache-dir tqdm==4.65.0
RUN pip install --no-cache-dir git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33
RUN pip install --no-cache-dir Pillow==9.5.0
RUN pip install --no-cache-dir orbax-checkpoint==0.3.3
RUN pip install --no-cache-dir gin-config==0.5.0
RUN pip install --no-cache-dir pycocotools==2.0.6
RUN pip install --no-cache-dir contextlib2==21.6.0
RUN pip install --no-cache-dir ml-collections==0.1.1
RUN pip install --no-cache-dir chex==0.1.7
RUN pip install --no-cache-dir optax==0.1.5
# Dependencies already included. Use no-deps to not update numpy.
RUN pip install --no-cache-dir --no-deps flax==0.7.2
RUN pip install --no-cache-dir --no-deps clu==0.0.9
# Installing jax at the very end with GPU support.
# NOTE: Not using `no-deps` flag here because we need CUDA support.
RUN pip install --no-cache-dir jax[cuda11_cudnn86]==0.4.9 \
--find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

WORKDIR ./fvlm
ENV PYTHONPATH ./

ENTRYPOINT ["python", "train_and_eval.py"]
