FROM pytorch/torchserve:0.3.0-cpu

COPY . /home/model-server/
WORKDIR /home/model-server/

USER model-server
RUN torch-model-archiver \
  --model-name=antandbee \
  --version=1.0 \
  --model-file=./model.py \
  --serialized-file=./model/antandbee.pth \
  --handler=./handler.py \
  --extra-files=./index_to_name.json \
  --export-path=./model-store \
  -f

CMD ["torchserve", \
     "--model-store ./model-store", \
     "--ts-config ./config.properties", \
     "--models antandbee=antandbee.mar"]


