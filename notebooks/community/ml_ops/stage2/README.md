# Stage 2: Experimentation

## Purpose

Progressively develop data requirements, model architecture, training procedures and evaluation metrics to transition model experimentation into a production environment.


## Recommendations  

The second stage in MLOps is experimenting in developing one or more baseline models. This stage may be done entirely by data scientists. Typically, there is a lot of try this and try that, with variance in the amount of experimental results that are not tracked (dropped on the floor). Flexibility is a key here, where the expectation is DIY (custom) and automatic methods are seamlessly interchangeable. We recommend:

- Use a data pipeline for feeding data to a model.
- Construct the data pipeline using Dataflow.
- Train a baseline model using Vertex AutoML or Vertex NAS
- Improve on the baseline with custom models/training using Vertex Training.
- For custom training, retrieve model architecture from internal or external model hub
- For domain specific custom models, retrieve domain specific initialized weights from model garden.
- For non-domain specific custom training, use warmup procedure to stabilize the weight initialization (or reuse previous cached warmup initialization).
- For custom training, use pre-training after warmup initialization for synthetic data.
- After warmup/pretrain is complete, do hyperparameter tuning with Vertex Vizier.
- Do full training with Vertex Training and Vertex Tensorboard.
- Hyperparameter tuning and full training is done with the same training script.
- Track training experiments with Vertex Experiments.
- For experiments that are preserved, track the artifacts, hyperparameters and evaluations in Vertex ML Metadata
- Use the What-if-Tool (WIT) to explore how the trained model would make predictions in different scenarios.


<img src='stage2v3.png'>
<br/>
<br/>
<br/>
<img src='stage2.2v1.png'>

## Notebooks

### Get Started

[Get Started with Vertex Experiments and Vertex ML Metadata](get_started_vertex_experiments.ipynb)

```
The steps performed include:

- Use Python logging to log training configuration/results locally.
- Use Google Cloud Logging to log training configuration/results in cloud storage.
- Create a Vertex AI `Experiment` resource.
- Instantiate an experiment run.
- Log parameters for the run.
- Log metrics for the run.
- Display the logged experiment run.
```

[Get Started with Vertex TensorBoard](get_started_vertex_tensorboard.ipynb)

```
The steps performed include:

- Create a TensorBoard callback when training a model.
- Using Tensorboard with locally trained model.
- Using Vertex AI TensorBoard with Vertex AI Training.
```

[Get Started with Custom Training Packages (Tensorflow)](get_started_vertex_training.ipynb)

```
The steps performed include:

- Training using a single Python script.
- Training using a Python package.
- Training using a custom training image.
- Laying out a training package.
```

[Get Started with Custom Training Packages (Scikit-Learn)](get_started_vertex_training_sklearn.ipynb)

```
The steps performed include:

- Training using a Python package.
- Report accuracy when hyperparameter tuning.
- Save the model artifacts to Cloud Storage using GCSFuse.
- Create a `Vertex AI Model` resource.
```

[Get Started with Custom Training Packages (XGBoost)](get_started_vertex_training_xgboost.ipynb)

```
The steps performed include:

- Training using a Python package.
- Report accuracy when hyperparameter tuning.
- Save the model artifacts to Cloud Storage using GCSFuse.
- Create a `Vertex AI Model` resource.
```

[Get Started with Custom Training Packages (Pytorch)](get_started_vertex_training_pytorch.ipynb)

```
The steps performed include:

- Single node training using a Python package.
- Report accuracy when hyperparameter tuning.
- Save the model artifacts to Cloud Storage using GCSFuse.
- Create a `Vertex AI Model` resource.
```

[Get Started with Custom Training Packages (R)](get_started_vertex_training_r.ipynb)

```
The steps performed include:

- Locally train an R model in a notebook using %%R magic commands
- Create a deployment image with trained R model and serving functions.
- Test the deployment image locally.
- Create a `Vertex AI Model` resource for the deployment image with embedded R model.
- Deploy the deployment image with embedded R model to a `Vertex AI Endpoint` resource.
- Test the deployment image with embedded R model.
- Create a R-to-Python training package.
- Create a training image for training the model.
- Train a R model using `Vertex AI Trainingh` service with the R-to-Python training package.
```

[Get Started with Custom Training Packages (R) and Deployment in R environment](get_started_vertex_training_r_using_r_kernel.ipynb)
```
The steps performed include:

- Create a custom R training script
- Create a custom R serving script
- Create a custom R deployment (serving) container.
- Train the model using `Vertex AI` custom training.
- Create an `Endpoint` resource.
- Deploy the `Model` resource (trained R model) to the `Endpoint` resource.
- Make an online prediction.
```

[Get Started with Custom Training Packages (LightGBM)](get_started_vertex_training_lightgbm.ipynb)

```
The steps performed include:

- Training using a Python package.
- Save the model artifacts to Cloud Storage using GCSFuse.
- Construct a FastAPI prediction server.
- Construct a Dockerfile deployment image.
- Test the deployment image locally.
- Create a `Vertex AI Model` resource.
```

[Get Started with Distributed Training](get_started_vertex_distributed_training.ipynb)

```
The steps performed include:

- `MirroredStrategy`: Train on a single VM with multiple GPUs.
- `MultiWorkerMirroredStrategy`: Train on multiple VMs with automatic setup of replicas.
- `MultiWorkerMirroredStrategy`: Train on multiple VMs with fine grain control of replicas.
- `ReductionServer`: Train on multiple VMS and sync updates across VMS with `Vertex AI Reduction Server`.
- `TPUTraining`: Train with multiple Cloud TPUs.
```

[Get Started with Vizier Hyperparameter Tuning](get_started_vertex_vizier.ipynb)

```
The steps performed include:

- Hyperparameter tuning with Random algorithm.
- Hyperparameter tuning with Vizier (Bayesian) algorithm.
```

[Get Started with AutoML Training](get_started_automl_training.ipynb)

```
The steps performed include:

- Train an image model.
- Export the image model as an edge model.
- Train a tabular model.
- Export the tabular model as a cloud model.
- Train a text model.
```

[Get Started with BQML Training](get_started_bqml_training.ipynb)

```
The steps performed include:

- Create a local BigQuery table in your project
- Train a BQML model
- Evaluate the BQML model
- Export the BQML model as a cloud model
- Upload the exported model as a `Vertex AI Model` resource
- Hyperparameter tune a BQML model with `Vertex AI Vizier`
- Automatically register a BQML model to `Vertex AI Model Registry`
```

[Get Started with Vertex Feature Store](get_started_vertex_feature_store.ipynb)

```
The steps performed include:

- Creating a Vertex AI `Featurestore` resource.
    - Creating `EntityType` resources for the `Featurestore` resource.
    - Creating `Feature` resources for each `EntityType` resource.
- Import feature values (entity data items) into `Featurestore` resource from Cloud Storage.
- Import feature values (entity data items) into `Featurestore` resource from pandas DataFrame.
- Perform online serving from a `Featurestore` resource.
- Perform batch serving from a `Featurestore` resource.
```

[Get Started with Google CMEK Training](get_started_with_cmek_training.ipynb)

```
The steps performed include:

- Creating a customer managed encryption key.
- Creating an image dataset with CMEK encryption.
- Train an AutoML model with CMEK encryption.
```

[Get Started with TensorFlow Hub models](get_started_with_tfhub_models.ipynb)

```
The steps performed include:

- Download a TensorFlow Hub prebuilt model.
- Add the task component as a classifier for the CIFAR-10 dataset.
- Fine tune locally the model with transfer learning training.
- Construct a custom training script:
    - Get training data from TensorFlow Datasets
    - Get model architecture from TensorFlow Hub
    - Train then model
    - Save model artifacts and upload as Vertex AI Model resource.
```

[Get Started with Vertex AI TabNet builtin algorithm](get_started_with_tabnet.ipynb)
```
The steps performed include:

- Get the training data.
- Configure training parameters for the Vertex AI TabNet container.
- Train the model using Vertex AI Training using CSV data.
- Upload the model as a Vertex AI Model resource.
- Deploy the Vertex AI Model resource to a Vertex AI Endpoint resource.
- Make a prediction with the deployed model.
- Hyperparameter tuning the Vertex AI TabNet model.
- Train the model using Vertex AI Training using BigQuery table.
```

[Get Started with Vision API and AutoML](get_started_with_visionapi_and_automl.ipynb)
```
The steps performed include:

- Preprocess training files using `Vision AI` APIs to extract the text from PDF files.
- Create a custom import file that includes annotation data based on the sample `BigQuery` dataset.
- Create a `Vertex AI Dataset` resource.
- Train the model.
- View the model evaluation.
- Deploy the `Vertex AI Model` resource to a serving `Endpoint` resource.
- Make a prediction.
- Undeploy the `Model`.
```


### E2E Stage Example

[Stage 2: Experimentation](mlops_experimentation.ipynb)

```
The steps performed include:

- Review the `Dataset` resource created during stage 1.
- Train an AutoML tabular binary classifier model in the background.
- Build the experimental model architecture.
- Construct a custom training package for the `Dataset` resource.
- Test the custom training package locally.
- Test the custom training package in the cloud with Vertex AI Training.
- Hyperparameter tune the model training with Vertex AI Vizier.
- Train the custom model with Vertex AI Training.
- Add a serving function for online/batch prediction to the custom model.
- Test the custom model with the serving function.
- Evaluate the custom model using Vertex AI Batch Prediction
- Wait for the AutoML training job to complete.
- Evaluate the AutoML model using Vertex AI Batch Prediction with the same evaluation slices as the custom model.
- Set the evaluation results of the AutoML model as the baseline.
- If the evaluation of the custom model is below baseline, continue to experiment with the custom model.
- If the evaluation of the custom model is above baseline, save the model as the first best model.
```
