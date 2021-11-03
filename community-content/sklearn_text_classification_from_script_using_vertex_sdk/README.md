# Train and deploy a sklearn model with Vertex AI

This repository shows how to train and deploy a text classifier using sklearn and Vertex AI.

The main Vertex AI features are:
- Vertex AI Custom Training
- Vertex AI Model
- Vertex AI Endpoint

Further GCP services are:
- Google Cloud Logging
- Google Cloud Storage

## Repository

    ├── README.md
    ├── create_job.ipynb    # <-- creates the training job and deploys it to an endpoint
    ├── requirements.txt    # <-- requirements for deploying the job
    └── task.py             # <-- contains the training application

## Training job overview

The training job performs the following steps:

1. Downloads the `NewsAggregator` dataset from the UCI Machine Learning Repository
2. Trains and evaluates a classifier using scikit-learn
3. Exports model and evaluation artifacts to GCS
4. Deploys the model as a `Vertex AI Endpoint`
