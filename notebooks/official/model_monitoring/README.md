
[Vertex AI Batch Prediction with Model Monitoring](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/model_monitoring/batch_prediction_model_monitoring.ipynb)

```
Learn to use the `Vertex AI Model Monitoring` service to detect drift and anomalies in batch prediction.

The steps performed include:

- Upload a pre-trained model as a Vertex AI Model resource.
- Generate batch prediction requests.
- Interpret the statistics, visualizations, other data reported by the model monitoring feature.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Model Monitoring for batch predictions](https://cloud.google.com/vertex-ai/docs/model-monitoring/model-monitoring-batch-predictions).


[Vertex AI Model Monitoring for AutoML tabular models](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/model_monitoring/get_started_with_model_monitoring_automl.ipynb)

```
Learn to use the `Vertex AI Model Monitoring` service to detect feature skew and drift in the input predict requests, for AutoML tabular models.

The steps performed include:

- Train an `AutoML` model.
- Deploy the `Model` resource to the `Endpoint` resource.
- Configure the `Endpoint` resource for model monitoring.
- Generate synthetic prediction requests for skew.
- Wait for email alert notification.
- Generate synthetic prediction requests for drift.
- Wait for email alert notification.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Model Monitoring](https://cloud.google.com/vertex-ai/docs/model-monitoring).


[Vertex AI Model Monitoring for batch prediction in AutoML image models](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/model_monitoring/get_started_with_model_monitoring_automl_image_batch.ipynb)

```
Learn how to use `Vertex AI Model Monitoring` with `Vertex AI Batch Prediction` with an AutoML image classification model to detect an out of distribution image.

The steps performed include:

1. Train an AutoML image classification model.
2. Submit a batch prediction containing both in and out of distribution images.
3. Use Model Monitoring to calculate anomaly score on each image.
4. Identify the images in the batch prediction request that are out of distribution.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Model Monitoring](https://cloud.google.com/vertex-ai/docs/model-monitoring).


[Vertex AI Model Monitoring for custom tabular models](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/model_monitoring/get_started_with_model_monitoring_custom.ipynb)

```
Learn to use the `Vertex AI Model Monitoring` service to detect feature skew and drift in the input predict requests, for custom tabular models.

The steps performed include:

- Download a pre-trained custom tabular model.
- Upload the pre-trained model as a `Model` resource.
- Deploy the `Model` resource to the `Endpoint` resource.
- Configure the `Endpoint` resource for model monitoring.
- Generate synthetic prediction requests for skew.
- Wait for email alert notification.
- Generate synthetic prediction requests for drift.
- Wait for email alert notification.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Model Monitoring](https://cloud.google.com/vertex-ai/docs/model-monitoring).


[Vertex AI Model Monitoring for custom tabular models with TensorFlow Serving container](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/model_monitoring/get_started_with_model_monitoring_custom_tf_serving.ipynb)

```
Learn to use the `Vertex AI Model Monitoring` service to detect feature skew and drift in the input predict requests, for custom tabular models, using a custom deployment container.

The steps performed include:

- Download a pre-trained custom tabular model.
- Upload the pre-trained model as a `Model` resource.
- Deploying the `Model` resource to an `Endpoint` resource with `TensorFlow Serving` serving binary.
- Configure the `Endpoint` resource for model monitoring.
- Generate synthetic prediction requests for skew.
- Wait for email alert notification.
- Generate synthetic prediction requests for drift.
- Wait for email alert notification.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Model Monitoring](https://cloud.google.com/vertex-ai/docs/model-monitoring).


[Vertex AI Model Monitoring for setup for tabular models](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/model_monitoring/get_started_with_model_monitoring_setup.ipynb)

```
Learn to setup the `Vertex AI Model Monitoring` service to detect feature skew and drift in the input predict requests.

The steps performed include:

- Download a pre-trained custom tabular model.
- Upload the pre-trained model as a `Model` resource.
- Deploy the `Model` resource to the `Endpoint` resource.
- Configure the `Endpoint` resource for model monitoring.
    - Skew and drift detection for feature inputs.
    - Skew and drift detection for feature attributions.
- Automatic generation of the `input schema` by sending 1000 prediction request.
- List, pause, resume and delete monitoring jobs.
- Restart monitoring job with predefined `input schema`.
- View logged monitored data.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Model Monitoring](https://cloud.google.com/vertex-ai/docs/model-monitoring).


[Vertex AI Model Monitoring for XGBoost models](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/model_monitoring/get_started_with_model_monitoring_xgboost.ipynb)

```
Learn to use the `Vertex AI Model Monitoring` service to detect feature skew and drift in the input predict requests for XGBoost models.

The steps performed include:

- Download a pre-trained XGBoost model.
- Upload the pre-trained model as a `Model` resource.
- Deploy the `Model` resource to the `Endpoint` resource.
- Configure the `Endpoint` resource for model monitoring:
  - drift detection only -- no access to training data.
  - predefine the input schema to map feature alias names to the unnamed array input to the model.
- Generate synthetic prediction requests for drift.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Model Monitoring](https://cloud.google.com/vertex-ai/docs/model-monitoring).


[Vertex AI Model Monitoring with Explainable AI Feature Attributions](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/model_monitoring/model_monitoring.ipynb)

```
Learn to use the `Vertex AI Model Monitoring` service to detect drift and anomalies in prediction requests from a deployed `Vertex AI Model` resource.

The steps performed include:

- Upload a pre-trained model as a `Vertex AI Model` resource.
- Create an `Vertex AI Endpoint` resource.
- Deploy the `Model` resource to the `Endpoint` resource.
- Configure the `Endpoint` resource for model monitoring.
- Initialize the baseline distribution for model monitoring.
- Generate synthetic prediction requests.
- Understand how to interpret the statistics, visualizations, other data reported by the model monitoring feature.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Model Monitoring](https://cloud.google.com/vertex-ai/docs/model-monitoring).

