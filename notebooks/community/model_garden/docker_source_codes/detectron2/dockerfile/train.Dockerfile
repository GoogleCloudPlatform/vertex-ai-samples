# Dockerfile for Detectron2 training.
#
# To build:
# docker build -f model_oss/detectron2/dockerfile/train.Dockerfile . -t ${YOUR_IMAGE_TAG}
#
# To push to gcr:
# docker tag ${YOUR_IMAGE_TAG} gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}
# docker push gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}

FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04
# Using an older system (18.04) to avoid opencv incompatibility (issue#3524).

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
    python3.7 python3.7-dev python3.7-distutils \
    python3-opencv ca-certificates git wget sudo ninja-build \
    curl wget vim

# Make python3 available for python3.7.
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2
RUN update-alternatives --config python3
# Make python available for python3.7.
RUN ln -sv /usr/bin/python3.7 /usr/bin/python

# Create a non-root user.
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/pip/get-pip.py && \
    python3.7 get-pip.py --user && \
    rm get-pip.py

# Important! Otherwise, it uses existing numpy from host-modules
# which throws error.
RUN pip install --user numpy==1.20.3

# Install dependencies:
# See https://pytorch.org/ for other options if you use
# a different version of CUDA.
RUN pip install --user tensorboard==2.11.0
# cmake from apt-get is too old.
RUN pip install --user cmake==3.25.2
RUN pip install --user torch==1.10.0+cu111 torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --user setuptools==59.5.0
RUN pip install --user opencv-python==4.7.0.72
RUN pip install --user cloudml-hypertune==0.1.0.dev6
RUN pip install --user fvcore==0.1.5.post20221221
# Install detectron2.
RUN git clone -b v0.6 https://github.com/facebookresearch/detectron2 detectron2_repo
# Set FORCE_CUDA because during `docker build` cuda is not accessible.
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda
# architectures and take a lot more time,
# because inside `docker build`, there is no way to tell
# which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
RUN pip install --user -e detectron2_repo

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"

# Copy license.
RUN wget https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/main/LICENSE

# Copy model-garden detectron2 files to '/home/appuser/trainer' folder.
ADD ./model_oss/detectron2 /home/appuser/trainer

################ Copy plain_train_net.py to task.py and
# then modify it using sed commands. ###################
# Src: https://github.com/facebookresearch/detectron2/blob/v0.6/tools/plain_train_net.py
RUN sudo cp /home/appuser/detectron2_repo/tools/plain_train_net.py /home/appuser/trainer/task.py
# Make additional changes to task.py.
# Note(lavrai): Start adding SED commands from end of file towards the top
# so that the line numbers do not keep changing for the source file.
# For entry-point:
RUN sudo sed -i "214 d" /home/appuser/trainer/task.py
RUN sudo sed -i "213 a\    default_arg_parser = default_argument_parser()" /home/appuser/trainer/task.py
RUN sudo sed -i "214 a\    extended_parser = trainer_utils.extend_parser_arguments(default_arg_parser)" /home/appuser/trainer/task.py
RUN sudo sed -i "215 a\    args = extended_parser.parse_args()" /home/appuser/trainer/task.py
# For main() function:
RUN sudo sed -i "192 a\    trainer_utils.register_dataset(args)" /home/appuser/trainer/task.py
# For setup() function:
RUN sudo sed -i "184 a\    cfg.SOLVER.BASE_LR = args.lr" /home/appuser/trainer/task.py
RUN sudo sed -i "185 a\    cfg.OUTPUT_DIR = args.output_dir" /home/appuser/trainer/task.py
RUN sudo sed -i "186 a\    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file_copy)" /home/appuser/trainer/task.py
RUN sudo sed -i "182 a\    config_file_copy = args.config_file" /home/appuser/trainer/task.py
RUN sudo sed -i "183 a\    args.config_file = model_zoo.get_config_file(args.config_file)" /home/appuser/trainer/task.py
# For new import:
RUN sudo sed -i "27 a\from detectron2 import model_zoo" /home/appuser/trainer/task.py
RUN sudo sed -i "21 a\import trainer_utils" /home/appuser/trainer/task.py

ENV PYTHONPATH /home/appuser/trainer

ENTRYPOINT ["python", "-m", "trainer.task"]