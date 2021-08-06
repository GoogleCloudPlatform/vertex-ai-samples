# PyTorch - Python Package Training

## Overview

The directory provides code to fine tune a transformer model ([BERT-base](https://huggingface.co/bert-base-cased)) from Huggingface Transformers Library for sentiment analysis task.  [BERT](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html) (Bidirectional Encoder Representations from Transformers) is a transformers model pre-trained on a large corpus of unlabeled text in a self-supervised fashion. In this sample, we use [IMDB sentiment classification dataset](https://huggingface.co/datasets/imdb) for the task. We show you packaging a PyTorch training model to submit it to Vertex AI using pre-built PyTorch containers and handling Python dependencies through Python build scripts (`setup.py`). 

The code directory structure and packaging is based on the sample [here](https://github.com/GoogleCloudPlatform/ai-platform-samples/blob/master/training/pytorch/structured/).

## Prerequisites
* Setup your project by following the instructions from [documentation](https://cloud.google.com/vertex-ai/docs/start/cloud-environment)
* Change directories to this sample.

## Directory Structure

* `trainer` directory: all Python modules to train the model.
* `scripts` directory: command-line scripts to train the model on Vertex AI.
* `setup.py`: `setup.py` scripts specifies Python dependencies required for the training job. Vertex Training uses pip to install the package on the training instances allocated for the job.

### Trainer Modules
| File Name | Purpose |
| :-------- | :------ |
| [metadata.py](trainer/metadata.py) | Defines: metadata for classification task such as predefined model dataset name, target labels. |
| [utils.py](trainer/utils.py) | Includes: utility functions such as data input functions to read data, save model to GCS bucket. |
| [model.py](trainer/model.py) | Includes: function to create model with a sequence classification head from a pretrained model. |
| [experiment.py](trainer/experiment.py) | Runs the model training and evaluation experiment, and exports the final model. |
| [task.py](trainer/task.py) | Includes: 1) Initialize and parse task arguments (hyper parameters), and 2) Entry point to the trainer. |

### Scripts

* [train-cloud.sh](scripts/train-cloud.sh) This script submits a training job to Vertex AI

## How to run
For local testing, run:
```
source ./scripts/train-local.sh
```

For cloud training, once the prerequisites are satisfied, update the
`BUCKET_NAME` environment variable in `scripts/train-cloud.sh`. You may then
run the following script to submit an AI Platform Training job:
```
source ./scripts/train-cloud.sh
```

## Run on GPU
The provided trainer code runs on a GPU if one is available including data loading and model creation.

To run the trainer code on a different GPU configuration or latest PyTorch pre-built container image, make the following changes to the trainer script.
* Update the PyTorch image URI to one of [PyTorch pre-built containers](https://cloud.google.com/ai-platform/training/docs/getting-started-pytorch#pytorch_containers)
* Update the scale tier to one that includes a GPU, e.g. `BASIC_GPU`.

Then, run the script to submit an AI Platform Training job:
```
source ./scripts/train-cloud.sh
```

### Versions
This script uses the pre-built PyTorch containers for PyTorch 1.7.
* `gcr.io/cloud-ml-public/training/pytorch-gpu.1-7`

