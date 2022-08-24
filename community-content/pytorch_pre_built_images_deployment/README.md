# PyTorch Deployment on Google Cloud: Text Classification

**Deploying PyTorch models using Vertex Prediction pre-built PyTorch images is currently an Experimental feature. Pre-GA products and features might have limited support, and changes to pre-GA products and features might not be compatible with other pre-GA versions. The Experimental release is covered by the Pre-GA Offerings Terms of your Google Cloud Platform [Terms of Service](https://cloud.google.com/terms).**

In the PyTorch on Google Cloud series of blog posts, we aim to share how to deploy PyTorch models at scale on [Vertex AI](https://cloud.google.com/vertex-ai).

This tutorial on text classification shows how to deploy a PyTorch based text classification model on [Vertex AI](https://cloud.google.com/vertex-ai/docs/start/client-libraries#python) using Vertex SDK and [`gcloud ai`](https://cloud.google.com/sdk/gcloud/reference/beta/ai).

## Notebooks

| <h4>Notebook</h4> | <h4>Description</h4>                       |
| :-------- | :------- |
| [pytorch-text-classification-vertex-ai-deploy.ipynb](./pytorch-text-classification-vertex-ai-deploy.ipynb) | Notebook to show deploying a PyTorch model on Vertex AI |

## Folders


| <h4>Folder Name</h4> | <h4>Description</h4>                       |
| :-------- | :------- |
| [`predictor`](./predictor) | Folder with custom prediction handler to deploy a PyTorch model to Vertex Prediction. In the [notebook](./pytorch-text-classification-vertex-ai-deploy.ipynb), this folder is used for deploying a PyTorch model on Vertex AI using Vertex Prediction pre-built PyTorch images |
