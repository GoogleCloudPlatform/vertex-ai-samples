# PyTorch Custom Containers GPU Template

## Overview

The directory provides code to fine tune a transformer model ([BERT-base](https://huggingface.co/bert-base-cased)) from Huggingface Transformers Library for sentiment analysis task.  [BERT](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html) (Bidirectional Encoder Representations from Transformers) is a transformers model pre-trained on a large corpus of unlabeled text in a self-supervised fashion. In this sample, we use [IMDB sentiment classification dataset](https://huggingface.co/datasets/imdb) for the task. We show you packaging a PyTorch training model to submit it to AI Platform using pre-built PyTorch containers and handling Python dependencies using [AI Platform Training custom containers](https://cloud.google.com/ai-platform/training/docs/custom-containers-training). 

The code directory structure and packaging is based on the sample [here](https://github.com/GoogleCloudPlatform/ai-platform-samples/blob/master/training/pytorch/structured/).

## Prerequisites

* Setup your project by following the instructions in the [setup](../../../../../setup/) directory.
* [Setup docker with Cloud Container Registry](https://cloud.google.com/container-registry/docs/pushing-and-pulling)
* Change the directory to this sample and run

`Note:` These instructions are used for local testing. When you submit a training job, no code will be executed on your local machine.
  

## Directory Structure

* `trainer` directory: with all the python modules to adapt to your data
* `scripts` directory: command-line scripts to train the model locally or on AI Platform
* `Dockerfile`: defines the docker image that include Python dependencies required for running the training job and the training Python modules itself.

### Trainer Modules
| File Name | Purpose |
| :-------- | :------ |
| [metadata.py](trainer/metadata.py) | Defines: metadata for classification task such as predefined model dataset name, target labels. |
| [utils.py](trainer/utils.py) | Includes: utility functions such as data input functions to read data, save model to GCS bucket. |
| [model.py](trainer/model.py) | Includes: function to create model with a sequence classification head from a pretrained model. |
| [experiment.py](trainer/experiment.py) | Runs the model training and evaluation experiment, and exports the final model. |
| [task.py](trainer/task.py) | Includes: 1) Initialise and parse task arguments (hyper parameters), and 2) Entry point to the trainer. |

### Scripts

* [train-local.sh](scripts/train-local) This script builds and tests Docker image locally and trains the model locally. It generates a SavedModel in local folder on the Docker Image.
* [train-cloud.sh](scripts/train-cloud.sh) This script builds your Docker image locally, pushes the image to Container Registry and submits a custom container training job to AI Platform.

Please read the [documentation](https://cloud.google.com/ai-platform/training/docs/custom-containers-training) on AI Platform Training with Custom Containers for more details.

## How to run

Once the prerequisites are satisfied, you may:

1. For local testing, run: 
    ```
    source ./scripts/train-local.sh
    ```
2. For cloud testing, run:
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
* `gcr.io/cloud-aiplatform/training/pytorch-gpu.1-7`
