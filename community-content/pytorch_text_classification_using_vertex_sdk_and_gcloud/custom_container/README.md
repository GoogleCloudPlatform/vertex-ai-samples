# PyTorch Custom Containers GPU Template

## Overview

The directory provides code to fine tune a transformer model ([BERT-base](https://huggingface.co/bert-base-cased)) from Huggingface Transformers Library for sentiment analysis task.  [BERT](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html) (Bidirectional Encoder Representations from Transformers) is a transformers model pre-trained on a large corpus of unlabeled text in a self-supervised fashion. In this sample, we use [IMDB sentiment classification dataset](https://huggingface.co/datasets/imdb) for the task. We show you packaging a PyTorch training model to submit it to Vertex AI using pre-built PyTorch containers and handling Python dependencies using [Vertex Training custom containers](https://cloud.google.com/vertex-ai/docs/training/create-custom-container?hl=hr). 

## Prerequisites

* Setup your project by following the instructions from [documentation](https://cloud.google.com/vertex-ai/docs/start/cloud-environment)
* [Setup docker with Cloud Container Registry](https://cloud.google.com/container-registry/docs/pushing-and-pulling)
* Change the directory to this sample and run

`Note:` These instructions are used for local testing. When you submit a training job, no code will be executed on your local machine.
  

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

* [train-cloud.sh](scripts/train-cloud.sh) This script builds your Docker image locally, pushes the image to Container Registry and submits a custom container training job to Vertex AI.

Please read the [documentation](https://cloud.google.com/vertex-ai/docs/training/containers-overview?hl=hr) on Vertex Training with Custom Containers for more details.

## How to run

Once the prerequisites are satisfied, you may:

1. For local testing, run (refer [notebook](../pytorch-text-classification-vertex-ai-train-tune-deploy.ipynb) for instructions): 
    ```
    CUSTOM_TRAIN_IMAGE_URI='gcr.io/{PROJECT_ID}/pytorch_gpu_train_{APP_NAME}'
    cd ./custom_container/ && docker build -f Dockerfile -t $CUSTOM_TRAIN_IMAGE_URI ../python_package
    docker run --gpus all -it --rm $CUSTOM_TRAIN_IMAGE_URI
    ```
2. For cloud testing, run:
    ```
    source ./scripts/train-cloud.sh
    ```

## Run on GPU
The provided trainer code runs on a GPU if one is available including data loading and model creation.

To run the trainer code on a different GPU configuration or latest PyTorch pre-built container image, make the following changes to the trainer script.
* Update the PyTorch image URI to one of [PyTorch pre-built containers](https://cloud.google.com/vertex-ai/docs/training/pre-built-containers#available_container_images)
* Update the [`worker-pool-spec`](https://cloud.google.com/vertex-ai/docs/training/configure-compute?hl=hr) in the gcloud command that includes a GPU

Then, run the script to submit a Custom Job on Vertex Training job:
```
source ./scripts/train-cloud.sh
```

### Versions
This script uses the pre-built PyTorch containers for PyTorch 1.7.
* `us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-7:latest`
