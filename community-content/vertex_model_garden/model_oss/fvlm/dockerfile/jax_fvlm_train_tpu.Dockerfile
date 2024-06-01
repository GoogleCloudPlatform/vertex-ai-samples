# This Dockerfile trains the F-VLM model on TPU.
# Here is an example to build this dockerfile:
# PROJECT="your gcp project"
# IMAGE_TAG="jax-f-vlm-train-tpu:${USER}-test"
# docker build -f model_oss/fvlm/dockerfile/jax_fvlm_train_tpu.Dockerfile . -t "${IMAGE_TAG}"
# docker tag "${IMAGE_TAG}" "gcr.io/${PROJECT}/${IMAGE_TAG}"
# docker push "gcr.io/${PROJECT}/${IMAGE_TAG}"

FROM python:3.11

# Get libtpu shared library. See go/what-is-libtpu.
RUN curl -L https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/libtpu/1.6.0/libtpu.so -o /lib/libtpu.so

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
        git \
        libgl1

# Copy Apache license.
RUN wget https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/main/LICENSE

# Install required libs
RUN pip install --upgrade pip

# Get F-VLM repository by using git sparse-checkout to avoid downloading entire
# google-research repository.
# Using the commit 05ece4b1c97285b48b51fa44321ccb2cb347406a on Dec 11th, 2023.
ARG COMMIT_ID=05ece4b1c97285b48b51fa44321ccb2cb347406a
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
RUN pip install --no-cache-dir ml_dtypes==0.3.1
RUN pip install --no-cache-dir tensorstore==0.1.51
RUN pip install --no-cache-dir MarkupSafe==2.1.3
RUN pip install --no-cache-dir Pillow==9.5.0
RUN pip install --no-cache-dir PyYAML==6.0.1
RUN pip install --no-cache-dir absl_py==1.4.0
RUN pip install --no-cache-dir array_record==0.4.1
RUN pip install --no-cache-dir astunparse==1.6.3
RUN pip install --no-cache-dir cachetools==5.3.1
RUN pip install --no-cache-dir certifi==2023.7.22
RUN pip install --no-cache-dir charset_normalizer==3.3.0
RUN pip install --no-cache-dir chex==0.1.83
RUN pip install --no-cache-dir click==8.1.7
RUN pip install --no-cache-dir clip==0.2.0
RUN pip install --no-cache-dir clu==0.0.9
RUN pip install --no-cache-dir contourpy==1.1.1
RUN pip install --no-cache-dir cycler==0.12.1
RUN pip install --no-cache-dir dm_tree==0.1.8
RUN pip install --no-cache-dir etils==1.5.1
RUN pip install --no-cache-dir filelock==3.12.4
RUN pip install --no-cache-dir flatbuffers==23.5.26
RUN pip install --no-cache-dir flax==0.7.4
RUN pip install --no-cache-dir fonttools==4.43.1
RUN pip install --no-cache-dir fsspec==2023.9.2
RUN pip install --no-cache-dir ftfy==6.1.1
RUN pip install --no-cache-dir gast==0.5.4
RUN pip install --no-cache-dir gin_config==0.5.0
RUN pip install --no-cache-dir google_auth==2.23.3
RUN pip install --no-cache-dir google_auth_oauthlib==1.0.0
RUN pip install --no-cache-dir google_pasta==0.2.0
RUN pip install --no-cache-dir googleapis_common_protos==1.61.0
RUN pip install --no-cache-dir grpcio==1.59.0
RUN pip install --no-cache-dir h5py==3.10.0
RUN pip install --no-cache-dir importlib_resources==6.1.0
RUN pip install --no-cache-dir 'jax[tpu]==0.4.18' \
-f https://storage.googleapis.com/jax-releases/libtpu_releases.html
RUN pip install --no-cache-dir jaxlib==0.4.18
RUN pip install --no-cache-dir jinja2==3.1.2
RUN pip install --no-cache-dir keras==2.14.0
RUN pip install --no-cache-dir kiwisolver==1.4.5
RUN pip install --no-cache-dir libclang==16.0.6
RUN pip install --no-cache-dir markdown==3.5
RUN pip install --no-cache-dir matplotlib==3.8.0
RUN pip install --no-cache-dir mpmath==1.3.0
RUN pip install --no-cache-dir networkx==3.1
RUN pip install --no-cache-dir numpy==1.26.0
RUN pip install --no-cache-dir nvidia_cublas_cu12==12.1.3.1
RUN pip install --no-cache-dir nvidia_cuda_cupti_cu12==12.1.105
RUN pip install --no-cache-dir nvidia_cuda_nvrtc_cu12==12.1.105
RUN pip install --no-cache-dir nvidia_cuda_runtime_cu12==12.1.105
RUN pip install --no-cache-dir nvidia_cudnn_cu12==8.9.2.26
RUN pip install --no-cache-dir nvidia_cufft_cu12==11.0.2.54
RUN pip install --no-cache-dir nvidia_curand_cu12==10.3.2.106
RUN pip install --no-cache-dir nvidia_cusolver_cu12==11.4.5.107
RUN pip install --no-cache-dir nvidia_cusparse_cu12==12.1.0.106
RUN pip install --no-cache-dir nvidia_nccl_cu12==2.18.1
RUN pip install --no-cache-dir nvidia_nvjitlink_cu12==12.2.140
RUN pip install --no-cache-dir nvidia_nvtx_cu12==12.1.105
RUN pip install --no-cache-dir opencv_python==4.8.1.78
RUN pip install --no-cache-dir orbax_checkpoint==0.4.1
RUN pip install --no-cache-dir promise==2.3
RUN pip install --no-cache-dir protobuf==3.20.3
RUN pip install --no-cache-dir psutil==5.9.5
RUN pip install --no-cache-dir pyasn1==0.5.0
RUN pip install --no-cache-dir pycocotools==2.0.7
RUN pip install --no-cache-dir pygments==2.16.1
RUN pip install --no-cache-dir regex==2023.10.3
RUN pip install --no-cache-dir rich==13.6.0
RUN pip install --no-cache-dir scipy==1.11.3
RUN pip install --no-cache-dir sympy==1.12
RUN pip install --no-cache-dir tensorboard==2.14.1
RUN pip install --no-cache-dir tensorboard_data_server==0.7.1
RUN pip install --no-cache-dir tensorflow==2.14.0
RUN pip install --no-cache-dir tensorflow_datasets==4.9.3
RUN pip install --no-cache-dir torch==2.1.0
RUN pip install --no-cache-dir torchvision==0.16.0
RUN pip install --no-cache-dir urllib3==2.0.6
RUN pip install --no-cache-dir wcwidth==0.2.8
RUN pip install --no-cache-dir werkzeug==3.0.0
RUN pip install --no-cache-dir wheel==0.41.2
RUN pip install --no-cache-dir tensorflow_text==2.14.0

WORKDIR ./fvlm
ENV PYTHONPATH ./

ENTRYPOINT ["python", "train_and_eval.py"]
