# To use this image, run this command with the desired notebook args from the top-level vertex-ai-samples directory:
# 1. To lint all changed notebooks:
# docker run -v ${PWD}:/setup/app gcr.io/python-docs-samples-tests/notebook_linter:latest
# 2. To lint specific notebooks:
# docker run -v ${PWD}:/setup/app gcr.io/python-docs-samples-tests/notebook_linter:latest notebooks/1.ipynb notebooks/2.ipynb

FROM python:3.11

WORKDIR setup

COPY ./requirements.txt .
COPY ./run_linter.sh .

# Install dependencies.
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

WORKDIR app

ENTRYPOINT ["/setup/run_linter.sh"]
