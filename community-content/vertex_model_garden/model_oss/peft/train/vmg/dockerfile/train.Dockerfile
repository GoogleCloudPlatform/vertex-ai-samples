# Dockerfile for PEFT Training.
#
# To build:
# docker build -f model_oss/peft/dockerfile/train.Dockerfile . -t ${YOUR_IMAGE_TAG}
#
# To push to gcr:
# docker tag ${YOUR_IMAGE_TAG} gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}
# docker push gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}

# Picked from https://cloud.google.com/deep-learning-containers/docs/choosing-container#pytorch
FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-cu121.2-2.py310:m123
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y curl git wget software-properties-common vim libaio-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists*

# Copy license.
RUN wget https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/main/LICENSE


# Install libraries.
ENV PIP_ROOT_USER_ACTION=ignore
RUN pip install --upgrade pip

# Remove packages that are not needed and are causing conflicts.
# dataproc_jupyter_plugin was installed as a part of pytorch-cu121.2-2.py310
# container which we don't need. It depends on ibis-framework and bigframes.
# The package and its dependencies request lower versions of pyarrow/pydantic
# than deepspeed/datasets. So, dataproc_jupyter_plugin conflicts with
# deepspeed/datasets.
RUN pip uninstall -y dataproc_jupyter_plugin ibis-framework bigframes

# Prefer to install with requirement file as much as possible for reasons
# described in b/355034754.
COPY model_oss/peft/train/vmg/dockerfile/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# flash-attn cannot be installed with the requirement file approach above
# because of the `no-build-isolation` requirement.
#
# It is OK to install it after other packages FOR NOW because it only has
# limited dependencies. And there's no concern about it overwriting previously
# installed packages.
# https://github.com/Dao-AILab/flash-attention/blob/v2.6.3/setup.py#L523
RUN pip install flash-attn==2.6.3 --no-build-isolation

# Install `diffusers` library as editable and in root folder (/) on purpose.
RUN git clone --depth 1 --branch v0.25.1 https://github.com/huggingface/diffusers.git
# Remove `diffusers` (NOTE that the dependency libraries are kept).
RUN pip uninstall -y diffusers
# Using `--no-deps` option to make sure previously installed packages are not
# overwritten.
RUN pip install --no-deps -e /diffusers

# Make sure there's no inconsistent pip libraries.
RUN pip check

# Install merge related packages in a separate env.
COPY model_oss/peft/train/vmg/dockerfile/merge_env.yaml /tmp/merge_env.yaml
RUN conda env create -n merge --yes --file /tmp/merge_env.yaml
RUN conda init

# Switch to diffusers examples folder.
WORKDIR /diffusers/examples

RUN mkdir -p ./vertex_vision_model_garden_peft/
COPY model_oss/peft/train/vmg/configs/* ./vertex_vision_model_garden_peft/
COPY model_oss/peft/train/vmg/*.py ./vertex_vision_model_garden_peft/train/vmg/
COPY model_oss/peft/train/vmg/templates /diffusers/examples/util/templates
COPY model_oss/peft/train/util/*.py /diffusers/examples/util/
COPY model_oss/util/* /diffusers/examples/util/
COPY model_oss/notebook_util/dataset_validation_util.py /diffusers/examples/util
COPY model_oss/peft/train/vmg/tests/*.py ./vertex_vision_model_garden_peft/tests/
COPY model_oss/peft/train/test_utils/test_util.py ./vertex_vision_model_garden_peft/tests/
COPY model_oss/peft/train/test_utils/command_builder.py ./vertex_vision_model_garden_peft/tests/

RUN chmod a+rwX -R /diffusers/examples/
ENV PYTHONPATH /diffusers/examples/
# Must disable torch XLA, otherwise runtime uses CPU even if GPU exists.
ENV USE_TORCH_XLA 0

ENTRYPOINT ["python3", "./vertex_vision_model_garden_peft/train/vmg/train_entrypoint.py"]
