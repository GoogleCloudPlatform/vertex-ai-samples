
[Training and deploying a sales forecasting model using FBProphet and Vertex AI](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/custom/SDK_FBProphet_Forecasting_Online.ipynb)

```
The objective of this notebook is to create, deploy and serve a custom forecasting model on Vertex AI.

The steps performed include:
- Train a model locally that forecasts sales for the given number of days.
- Train another model that uses both sales and weather data for sales prediction.
- Save both the models.
- Build a FastAPI server to handle the predictions for the chosen model.
- Build a custom container image of the serving application with the model artifacts.
- Upload the model to Vertex AI Model Registry.
- Deploy the model to a Vertex AI Endpoint.
- Send online prediction requests to the deployed model.
- Clean up the resources created in this session.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Training](https://cloud.google.com/vertex-ai/docs/training/custom-training).

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Prediction](https://cloud.google.com/vertex-ai/docs/predictions/get-predictions).


[Custom training and batch prediction](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/custom/sdk-custom-image-classification-batch.ipynb)

```
Learn to use `Vertex AI Training` to create a custom trained model and use `Vertex AI Batch Prediction` to do a batch prediction on the trained model.

The steps performed include:

- Create a `Vertex AI` custom job for training a TensorFlow model.
- Upload the trained model artifacts as a `Model` resource.
- Make a batch prediction.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Training](https://cloud.google.com/vertex-ai/docs/training/custom-training).

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Batch Prediction](https://cloud.google.com/vertex-ai/docs/tabular-data/classification-regression/get-batch-predictions).


[Profile model training performance using Profiler](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/custom/custom_training_tensorboard_profiler.ipynb)

```
Learn how to enable Vertex AI TensorBoard Profiler for custom training jobs.

The steps performed include:

- Setup a service account and a Cloud Storage bucket
- Create a TensorBoard instance
- Create and run a custom training job
- View the TensorBoard Profiler dashboard

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI TensorBoard Profiler](https://cloud.google.com/vertex-ai/docs/experiments/tensorboard-profiler).


[Training a TensorFlow model on BigQuery data](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/custom/custom-tabular-bq-managed-dataset.ipynb)

```
Learn how to create a custom-trained model from a Python script in a Docker container using the Vertex AI SDK for Python, and then get a prediction from the deployed model by sending data.

The steps performed include:

- Create a Vertex AI custom `TrainingPipeline` for training a model.
- Train a TensorFlow model.
- Deploy the `Model` resource to a serving `Endpoint` resource.
- Make a prediction.
- Undeploy the `Model` resource.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Training](https://cloud.google.com/vertex-ai/docs/training/custom-training).


[Custom training and online prediction](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/custom/sdk-custom-image-classification-online.ipynb)

```
Learn to use `Vertex AI Training` to create a custom-trained model from a Python script in a Docker container, and learn to use `Vertex AI Prediction` to do a prediction on the deployed model by sending data.

The steps performed include:

- Create a `Vertex AI` custom job for training a TensorFlow model.
- Upload the trained model artifacts to a `Model` resource.
- Create a serving `Endpoint` resource.
- Deploy the `Model` resource to a serving `Endpoint` resource.
- Make a prediction.
- Undeploy the `Model` resource.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Training](https://cloud.google.com/vertex-ai/docs/training/custom-training).

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Prediction](https://cloud.google.com/vertex-ai/docs/predictions/get-predictions).


[Deploying Iris-detection model using FastAPI and Vertex AI custom container serving](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/custom/SDK_Custom_Container_Prediction.ipynb)

```
Learn how to create, deploy and serve a custom classification model on Vertex AI.

The steps performed include:

- Train a model that uses flower's measurements as input to predict the class of iris.
- Save the model and its serialized pre-processor.
- Build a FastAPI server to handle predictions and health checks.
- Build a custom container with model artifacts.
- Upload and deploy custom container to Vertex AI Endpoints.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Training](https://cloud.google.com/vertex-ai/docs/training/custom-training).

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Prediction](https://cloud.google.com/vertex-ai/docs/predictions/get-predictions).

