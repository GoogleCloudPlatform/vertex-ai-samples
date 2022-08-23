# PyTorch on Google Cloud: Text Classification

**Deploying PyTorch models using Vertex Prediction pre-built PyTorch images is currently an Experimental feature. Pre-GA products and features might have limited support, and changes to pre-GA products and features might not be compatible with other pre-GA versions. The Experimental release is covered by the Pre-GA Offerings Terms of your Google Cloud Platform [Terms of Service](https://cloud.google.com/terms).**

In the PyTorch on Google Cloud series of blog posts, we aim to share how to build, train, deploy and orchestrate PyTorch models at scale and how to create reproducible machine learning pipelines on Google Cloud with [Vertex AI](https://cloud.google.com/vertex-ai).

This tutorial on text classification shows how to train a PyTorch based text classification model by fine tuning a pre-trained Huggingface Transformers model and deploy the model on [Vertex AI](https://cloud.google.com/vertex-ai/docs/start/client-libraries#python) using Vertex SDK and [`gcloud ai`](https://cloud.google.com/sdk/gcloud/reference/beta/ai).

## Notebooks

| <h4>Notebook</h4> | <h4>Description</h4>                       |
| :-------- | :------- |
| [pytorch-text-classification-vertex-ai-train-tune-deploy.ipynb](./pytorch-text-classification-vertex-ai-train-tune-deploy.ipynb) | Notebook to show training, hyper-parameter tuning and deploying a PyTorch model on Vertex AI |

## Folders


| <h4>Folder Name</h4> | <h4>Description</h4>                       |
| :-------- | :------- |
| [`python_package`](./python_package) | Folder with scripts to train and tune the text classification model using PyTorch and Hugging Face Transformers. In the [notebook](./pytorch-text-classification-vertex-ai-train-tune-deploy.ipynb), this folder is used for submitting a training job on Vertex AI using pre-built PyTorch containers. |
| [`custom_container`](./custom_container) | Folder with reference to training scripts in [`python_package`](./python_package)folder including a `Dockerfile` to build a custom container. In the [notebook](./pytorch-text-classification-vertex-ai-train-tune-deploy.ipynb), this folder is used for submitting a training job and hyper-parameter tuning job on Vertex AI using custom containers. |
| [`predictor`](./predictor) | Folder with custom prediction handler to deploy a PyTorch model to Vertex Prediction. In the [notebook](./pytorch-text-classification-vertex-ai-train-tune-deploy.ipynb), this folder is used for deploying a PyTorch model on Vertex AI using Vertex Prediction pre-built PyTorch images |
