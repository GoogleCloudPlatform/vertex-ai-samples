[PyTorch Image Classification Multi-Node Distributed Data Parallel Training on GPU using Vertex AI Training with Custom Container](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/training/multi_node_ddp_nccl_vertex_training/multi_node_ddp_nccl_vertex_training_with_custom_container.ipynb)

Learn how to create a distributed PyTorch training job using Vertex AI SDK for Python and custom containers.

The steps performed include:

- Setting up your GCP project : Setting up the PROJECT_ID, REGION & SERVICE_ACCOUNT
- Creating a cloud storage bucket
- Building Custom Container using Artifact Registry and Docker
- Create a Vertex AI Tensorboard Instance to store your Vertex AI experiment
- Run a Vertex AI SDK CustomContainerTrainingJob