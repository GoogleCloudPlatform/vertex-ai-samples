FROM pytorch/torchserve:latest-cpu

RUN pip3 install transformers

COPY ./custom_text_handler.py /home/model-server/
COPY ./index_to_name.json /home/model-server/
COPY ./models/finetuned-bert-classifier/ /home/model-server/

USER root
RUN printf "\nservice_envelope=json" >> /home/model-server/config.properties
RUN printf "\ninference_address=http://0.0.0.0:7080" >> /home/model-server/config.properties
RUN printf "\nmanagement_address=http://0.0.0.0:7081" >> /home/model-server/config.properties
USER model-server

EXPOSE 7080
EXPOSE 7081

RUN torch-model-archiver -f \
  --model-name=finetune-bert-classifier \
  --version=1.0 \
  --serialized-file=/home/model-server/pytorch_model.bin \
  --handler=/home/model-server/custom_text_handler.py \
  --extra-files "/home/model-server/config.json,/home/model-server/tokenizer.json,/home/model-server/training_args.bin,/home/model-server/tokenizer_config.json,/home/model-server/special_tokens_map.json,/home/model-server/vocab.txt,/home/model-server/index_to_name.json" \
  --export-path=/home/model-server/model-store

CMD ["torchserve", \
     "--start", \
     "--ts-config=/home/model-server/config.properties", \
     "--models", \
     "finetune-bert-classifier=finetune-bert-classifier.mar", \
     "--model-store", \
     "/home/model-server/model-store"]
