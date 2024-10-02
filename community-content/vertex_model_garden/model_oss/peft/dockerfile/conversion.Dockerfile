# Dockerfile for Language Model Conversion.
#
# To build:
# docker build -f model_oss/peft/dockerfile/conversion.Dockerfile . -t ${YOUR_IMAGE_TAG}
#
# To push to gcr:
# docker tag ${YOUR_IMAGE_TAG} gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}
# docker push gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}

FROM tensorflow/build:2.14-python3.8

RUN git clone https://github.com/facebookresearch/llama-recipes.git && \
    cd llama-recipes && \
    pip install -r requirements.txt && \
    pip freeze | grep transformers && \
    git clone https://github.com/huggingface/transformers.git && \
    cd transformers && \
    pip install protobuf

WORKDIR /llama-recipes/transformers

ENTRYPOINT ["python","src/transformers/models/llama/convert_llama_weights_to_hf.py"]
