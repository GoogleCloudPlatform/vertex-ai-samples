# Dockerfile wrapping NeMo.
#
# To workaround base nemo docker image using too many layers, we use Multi-stage
# build to first collect the additional files we'll need.
FROM alpine:latest AS prep_files
WORKDIR /workspace
RUN mkdir -p configs vdt vdt/util
COPY scripts/*.py vdt/
COPY scripts/util/*.py vdt/util/
COPY configs/* configs/
COPY docker/patches/24.09/* vdt/patches/
RUN chmod a+rwX -R vdt
# Copy license.
RUN wget https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/main/LICENSE

# Available tags
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags
# It installs NeMo source code in /opt/NeMo folder, with tag=r2.0.0
FROM nvcr.io/nvidia/nemo:24.09

RUN apt-get update && apt-get install -y sudo zsh tmux && \
    rm -rf /var/lib/apt/lists*

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | \
    tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update -y && apt-get install google-cloud-sdk -y && \
    rm -rf /var/lib/apt/lists*

# Install libraries with pip
ENV PIP_ROOT_USER_ACTION=ignore

# We expect this will be run in the root directory of the vertex-dist-recipes repo
ARG HOST_SRC_DIR="."

# The pre-installed NeMo introduces a lot of deps conflicts.
# We uninstall the confilicting libs and reinstall some of them as needed.
COPY ${HOST_SRC_DIR}/docker/uninstall.txt /tmp/uninstall.txt
RUN cat /tmp/uninstall.txt | grep -v '#' | xargs pip uninstall -y
COPY ${HOST_SRC_DIR}/docker/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Make sure there's no inconsistent pip libraries.
RUN pip check

WORKDIR /workspace

# Copy configs
COPY ${HOST_SRC_DIR}/configs/* /opt/NeMo/examples/nlp/language_modeling/conf/

# Copy all additional files we need from `prep_files` image.
COPY --from=prep_files /workspace/ .

# Install for `src/utils/training_metrics/process_training_results.py` to report
# throughput and MFU numbers.
RUN git clone https://github.com/AI-Hypercomputer/gpu-recipes.git

# This hack is needed for multi-node training while not using a sharing file system.
RUN patch --verbose -l -d /opt/megatron-lm/megatron/core/datasets -p1 -i /workspace/vdt/patches/local_rank.patch; \
    git -C /workspace/gpu-recipes apply /workspace/vdt/patches/throughput_calc.patch; \
    git -C /opt/NeMo apply /workspace/vdt/patches/nemo2hf.patch; \
    git -C /opt/NeMo apply /workspace/vdt/patches/sigabort.patch;
# git -C /opt/NeMo apply /workspace/vdt/patches/gpu_stats.patch;

# Do not put an entrypoint here. Specify the entrypoint in the docker run script.
