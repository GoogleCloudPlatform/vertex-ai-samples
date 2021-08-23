FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-4

COPY . /trainer

WORKDIR /trainer

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "-m", "task"]