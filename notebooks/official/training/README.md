
[Distributed Vertex AI Hyperparameter Tuning](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/training/distributed_hyperparameter_tuning.ipynb)

```
In this notebook, you create a custom trained model from a Python script in a Docker container.

The steps performed include:

- Training using a Python package.
- Report accuracy when hyperparameter tuning.
- Save the model artifacts to Cloud Storage using GCSFuse.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Hyperparameter Tuning](https://cloud.google.com/vertex-ai/docs/training/hyperparameter-tuning-overview).


[Get started with Vertex AI Distributed Training](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/training/get_started_with_vertex_distributed_training.ipynb)

```
Learn how to use `Vertex AI Distributed Training` when training with `Vertex AI`.

The steps performed include:

- `MirroredStrategy`: Train on a single VM with multiple GPUs.
- `MultiWorkerMirroredStrategy`: Train on multiple VMs with automatic setup of replicas.
- `MultiWorkerMirroredStrategy`: Train on multiple VMs with fine grain control of replicas.
- `ReductionServer`: Train on multiple VMS and sync updates across VMS with `Vertex AI Reduction Server`.
- `TPUTraining`: Train with multiple Cloud TPUs.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Distributed Training](https://cloud.google.com/vertex-ai/docs/training/distributed-training).


[Run hyperparameter tuning for a TensorFlow model](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/training/hyperparameter_tuning_tensorflow.ipynb)

```
Learn how to run a Vertex AI Hyperparameter Tuning job for a TensorFlow model.

The steps performed include:

* Modify training application code for automated hyperparameter tuning.
* Containerize training application code.
* Configure and launch a hyperparameter tuning job with the Vertex AI Python SDK.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Hyperparameter Tuning](https://cloud.google.com/vertex-ai/docs/training/hyperparameter-tuning-overview).


[Vertex AI Hyperparameter Tuning for XGBoost](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/training/hyperparameter_tuning_xgboost.ipynb)

```
Learn how to use `Vertex AI Hyperparameter Tuning` for training a XGBoost custom model.

The steps performed include:

- Training using a Python package.
- Report accuracy when hyperparameter tuning.
- Save the model artifacts to Cloud Storage using GCSFuse.
- Create a `Vertex AI Model` resource.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Hyperparameter Tuning](https://cloud.google.com/vertex-ai/docs/training/hyperparameter-tuning-overview).


[PyTorch image classification multi-node distributed data parallel training on cpu using Vertex training with custom container](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/training/multi_node_ddp_gloo_vertex_training_with_custom_container.ipynb)

```
Learn how to create a distributed PyTorch training job using Vertex AI SDK for Python and custom containers.

The steps performed include:

- Setting up your GCP project : Setting up the PROJECT_ID, REGION & SERVICE_ACCOUNT
- Creating a cloud storage bucket
- Building Custom Container using Artifact Registry and Docker
- Create a Vertex AI TensorBoard instance to store your Vertex AI experiment
- Run a Vertex AI SDK CustomContainerTrainingJob

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Training](https://cloud.google.com/vertex-ai/docs/training/custom-training).


[PyTorch image classification multi-node NCCL distributed data parallel training on cpu using Vertex training with custom container](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/training/multi_node_ddp_nccl_vertex_training_with_custom_container.ipynb)

```
Learn how to create a distributed PyTorch training job using Vertex AI SDK for Python and custom containers.

The steps performed include:

- Building Custom Container using Artifact Registry and Docker
- Create a Vertex AI tensorboard instance to store your Vertex AI experiment
- Run a Vertex AI SDK CustomContainerTrainingJob

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Training](https://cloud.google.com/vertex-ai/docs/training/custom-training).


[Training, tuning and deploying a PyTorch text sentiment classification model on Vertex AI](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/training/pytorch-text-sentiment-classification-custom-train-deploy.ipynb)

```
Learn to build, train, tune and deploy a PyTorch model on Vertex AI.

The steps performed include:

- Create training package for the text classification model.
- Train the model with custom training on Vertex AI.
- Check the created model artifacts.
- Create a custom container for predictions.
- Deploy the trained model to a Vertex AI Endpoint using the custom container for predictions.
- Send online prediction requests to the deployed model and validate.
- Clean up the resources created in this notebook.

```

&nbsp;&nbsp;&nbsp;Learn more about [Custom training](https://cloud.google.com/vertex-ai/docs/training/custom-training).


[Train PyTorch model on Vertex AI with data from Cloud Storage](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/training/pytorch_gcs_data_training.ipynb)

```
Learn how to create a training job using PyTorch and a dataset stored on Cloud Storage.

The steps performed include:

- Write a custom training script that creates your train & test datasets and trains the model.
- Run a Vertex AI SDK `CustomTrainingJob`

```

&nbsp;&nbsp;&nbsp;Learn more about [PyTorch integration in Vertex AI](https://cloud.google.com/vertex-ai/docs/start/pytorch).


[Vertex AI SDK 2.0 Vertex AI Remote Hyperparameter Tuning for OSS ML frameworks](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/training/sdk2_remote_hyperparameter_tuning.ipynb)

```
Learn to use `Vertex AI SDK 2.

The steps performed include:

- Download and split the dataset
- Perform transformations as a Vertex AI remote training.
- For scikit-learn, PyTorch, TensorFlow, PyTorch Lightning, Tabnet
    - Tune the model remotely.
    - Get the best model.

```


[Vertex AI SDK 2.0 Vertex AI Remote Training for OSS ML frameworks](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/training/sdk2_remote_training.ipynb)

```
Learn to use `Vertex AI SDK 2.

The steps performed include:

- Download and split the dataset
- Perform transformations as a Vertex AI remote training.
- For scikit-learn, PyTorch, TensorFlow, PyTorch Lightning
    - Train the model remotely.
    - Uptrain the pretrained model remotely.
    - Evaluate both the pretrained and uptrained model.

```


[Distributed XGBoost training with Dask](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/training/xgboost_data_parallel_training_on_cpu_using_dask.ipynb)

```
Learn how to create a distributed training job using XGBoost with Dask.

The steps performed include:

- Configure the `PROJECT_ID` and `REGION` variables for your Google Cloud project.
- Create a Cloud Storage bucket to store your model artifacts.
- Build a custom Docker container that hosts your training code and push the container image to Artifact Registry.
- Run a Vertex AI SDK CustomContainerTrainingJob

```

&nbsp;&nbsp;&nbsp;Learn more about [Custom training](https://cloud.google.com/vertex-ai/docs/training/custom-training).

