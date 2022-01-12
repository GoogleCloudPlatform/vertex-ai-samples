# TODO Need license file

ARG CUDA_MAJOR=11
ARG CUDA_MINOR=0

FROM gcr.io/deeplearning-platform-release/base-cu110

ARG CUDA_MAJOR
ARG CUDA_MINOR

SHELL ["/bin/bash", "-c"]

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
      build-essential \
      cmake \
      cuda-command-line-tools-${CUDA_MAJOR}-${CUDA_MINOR} \
      git \
      hmmer \
      kalign \
      tzdata \
      wget \
    && rm -rf /var/lib/apt/lists/*

# Compile HHsuite from source.
RUN git clone --branch v3.3.0 https://github.com/soedinglab/hh-suite.git /tmp/hh-suite \
    && mkdir /tmp/hh-suite/build \
    && pushd /tmp/hh-suite/build \
    && cmake -DCMAKE_INSTALL_PREFIX=/opt/hhsuite .. \
    && make -j 4 && make install \
    && ln -s /opt/hhsuite/bin/* /usr/bin \
    && popd \
    && rm -rf /tmp/hh-suite

ENV PATH="/opt/conda/bin:$PATH"
RUN conda update -qy conda \
    && conda install -y -c conda-forge \
      openmm=7.5.1 \
      cudatoolkit==${CUDA_VERSION} \
      pdbfixer \
      pip \
      python=3.7

COPY . /app/alphafold

# Install pip packages.
RUN pip3 install --upgrade pip \
    && pip3 install -r /app/alphafold/requirements.txt \
    && pip3 install py3Dmol tqdm \
    && pip3 install --upgrade jax==0.2.14 jaxlib==0.1.69+cuda${CUDA_MAJOR}${CUDA_MINOR} -f \
      https://storage.googleapis.com/jax-releases/jax_releases.html
   
# Downgrade because of new base image.
RUN pip3 install jax==0.2.14 py3Dmol tqdm

# Install alphafold.
WORKDIR /app/alphafold
RUN python setup.py install

# Apply OpenMM patch.
WORKDIR /opt/conda/lib/python3.7/site-packages
RUN patch -p0 < /app/alphafold/docker/openmm.patch

# Creating a tmp location for jackhmmr; not mounting through to host though.
RUN sudo mkdir -m 777 --parents /tmp/ramdisk

# We need to run `ldconfig` first to ensure GPUs are visible, due to some quirk
# with Debian. See https://github.com/NVIDIA/nvidia-docker/issues/1399 for
# details.
# ENTRYPOINT does not support easily running multiple commands, so instead we
# write a shell script to wrap them up.
WORKDIR /app/alphafold
RUN echo '#!/bin/bash\nldconfig\n\'

