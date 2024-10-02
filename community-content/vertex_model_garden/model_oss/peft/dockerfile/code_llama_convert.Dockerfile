# Base on pytorch-cuda image.
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

# Install tools.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -y --no-install-recommends curl
RUN apt-get install -y --no-install-recommends wget
RUN apt-get install -y --no-install-recommends git

# Install libraries.
ENV PIP_ROOT_USER_ACTION=ignore
RUN python3 -m pip install --upgrade pip
RUN pip install tokenizers==0.13.3
RUN pip install accelerate==0.21.0
RUN pip install sentencepiece==0.1.99
RUN pip install datasets==2.14.4
RUN pip install protobuf==4.24.1

# Install transformers
RUN git clone https://github.com/huggingface/transformers.git
WORKDIR transformers
# Pin the commit to add-code-llama 08/25/2023
RUN git reset --hard 015f8e110d270a0ad42de4ae5b98198d69eb1964
RUN pip install -e .

ENTRYPOINT ["python","src/transformers/models/llama/convert_llama_weights_to_hf.py"]
