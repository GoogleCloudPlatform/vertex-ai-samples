# Dockerfile for basic training dockers with timm.
#
# To build:
# docker build -f model_oss/timm/dockerfile/train.Dockerfile . -t ${YOUR_IMAGE_TAG}
#
# To push to gcr:
# docker tag ${YOUR_IMAGE_TAG} gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}
# docker push gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}

# Base on pytorch-cuda image.
FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

# Install tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        wget \
        vim

# Copy license.
RUN wget https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/main/LICENSE

# Download timm source code with pinned version.
RUN wget -q https://github.com/rwightman/pytorch-image-models/archive/refs/tags/v0.6.12.tar.gz
RUN tar xzf v0.6.12.tar.gz

# Install libraries.
RUN pip install cloudml-hypertune==0.1.0.dev6

# Switch to timm repo.
WORKDIR /workspace/pytorch-image-models-0.6.12

# NOTE: use 'sed' to modify the timm source code to
#       make timm CheckpointSaver can work with gcsfuse.
RUN sed -i "1 i\import shutil" timm/utils/checkpoint_saver.py
RUN sed -i "s#os.link#shutil.copyfile#g" timm/utils/checkpoint_saver.py
RUN sed -i "s#os.unlink#os.remove#g" timm/utils/checkpoint_saver.py

# NOTE: use 'sed' to modify the timm source code to
#       add hp training support to timm trainer.
RUN sed -i "693 a\        if saver is not None: hpt = hypertune.HyperTune(); hpt.report_hyperparameter_tuning_metric(hyperparameter_metric_tag='top1_accuracy', metric_value=best_metric, global_step=best_epoch)" train.py
RUN sed -i "1 i\import hypertune" train.py

# Install timm from source code.
RUN pip install -e .

# https://pytorch.org/docs/stable/elastic/run.html
ENTRYPOINT ["torchrun"]