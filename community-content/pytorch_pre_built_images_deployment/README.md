# PyTorch Deployment on Google Cloud: Text Classification

**This is an Experimental release**, covered by the Pre-GA Offerings Terms of your Google Cloud Platform [Terms of Service](https://cloud.google.com/terms).

Experiments are focused on validating a prototype and are not guaranteed to be released. They are not intended for production use or covered by any SLA, support obligation, or deprecation policy and might be subject to backward-incompatible changes.

**Kindly drop us a note before you run any scale tests.**

**Do not hesitate to contact cloudml-feedback@google.com if you have any questions or run into any issues.**

The projects need to be added to the allowlist in order to deploy PyTorch models using Vertex AI Prediction pre-built PyTorch images. If you are interested in the feature, please send an email to cloudml-feedback@google.com to provide your project numbers and project ids.

## Overview

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
