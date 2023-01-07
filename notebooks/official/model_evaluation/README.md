
[Evaluating batch prediction results from an AutoML Tabular classification model](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/model_evaluation/automl_tabular_classification_model_evaluation.ipynb)

```
Learn how to train a Vertex AI AutoML Tabular classification model and learn how to evaluate it through a Vertex AI pipeline job using `google_cloud_pipeline_components`:

The steps performed include:

- Create a Vertex AI `Dataset`.
- Train an Automl Tabular classification model on the `Dataset` resource.
- Import the trained `AutoML model resource` into the pipeline.
- Run a `Batch Prediction` job.
- Evaluate the AutoML model using the `Classification Evaluation component`.
- Import the classification metrics to the AutoML model resource.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Model Evaluation](https://cloud.google.com/vertex-ai/docs/evaluation/introduction).

&nbsp;&nbsp;&nbsp;Learn more about [AutoML Tabular](https://cloud.google.com/vertex-ai/docs/start/automl-users#tables).


[Evaluating batch prediction results from AutoML Video classification model](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/model_evaluation/automl_video_classification_model_evaluation.ipynb)

```
Learn how to train a Vertex AI AutoML Video classification model and learn how to evaluate it through a Vertex AI pipeline job using `google_cloud_pipeline_components`:

The steps performed include:

- Create a `Vertex AI Dataset`.
- Train a Automl Video Classification model on the `Vertex AI Dataset` resource.
- Import the trained `AutoML Vertex AI Model resource` into the pipeline.
- Run a batch prediction job inside the pipeline.
- Evaulate the AutoML model using the classification evaluation component.
- Import the classification metrics to the AutoML Vertex AI Model resource.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Model Evaluation](https://cloud.google.com/vertex-ai/docs/evaluation/introduction).

&nbsp;&nbsp;&nbsp;Learn more about [AutoML Video](https://cloud.google.com/vertex-ai/docs/video-data/classification/prepare-data).


[Evaluating batch prediction results from AutoML Tabular regression model](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/model_evaluation/automl_tabular_regression_model_evaluation.ipynb)

```
Learn how to evaluate a Vertex AI model resource through a Vertex AI pipeline job using `google_cloud_pipeline_components`:

The steps performed include:

- Create a Vertex AI Dataset
- Configure a `AutoMLTabularTrainingJob`
- Run the `AutoMLTabularTrainingJob` which returns a model
- Import a pre-trained `AutoML model resource` into the pipeline
- Run a `batch prediction` job in the pipeline
- Evaulate the AutoML model using the `regression evaluation component`
- Import the Regression Metrics to the AutoML model resource

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Model Evaluation](https://cloud.google.com/vertex-ai/docs/evaluation/introduction).

&nbsp;&nbsp;&nbsp;Learn more about [AutoML Tabular](https://cloud.google.com/vertex-ai/docs/start/automl-users#tables).


[Evaluating batch prediction results from custom tabular regression model](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/model_evaluation/custom_tabular_regression_model_evaluation.ipynb)

```
Learn how to evaluate a Vertex AI model resource through a Vertex AI pipeline job using `google_cloud_pipeline_components`:

The steps performed include:

- Create a Vertex AI `CustomTrainingJob` for training a model.
- Run the `CustomTrainingJob` 
- Retrieve and load the model artifacts.
- View the model evaluation.
- Upload the model as a Vertex AI Model resource.
- Import a pre-trained `Vertex AI model resource` into the pipeline.
- Run a `batch prediction` job in the pipeline.
- Evaulate the model using the `regression evaluation component`.
- Import the Regression Metrics to the Vertex AI model resource.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Model Evaluation](https://cloud.google.com/vertex-ai/docs/evaluation/introduction).

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Training](https://cloud.google.com/vertex-ai/docs/training/custom-training).


[AutoML text classification pipelines using google-cloud-pipeline-components](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/model_evaluation/automl_text_classification_model_evaluation.ipynb)

```
Learn how to use `Vertex AI Pipelines` and `Google Cloud Pipeline Components` to build an `AutoML` text classification model.

The steps performed include:

- Create a Vertex AI `Dataset`.
- Train a Automl Tabular Classification model on the `Dataset` resource.
- Import the trained `AutoML model resource` into the pipeline.
- Run a `Batch Prediction` job.
- Evaulate the AutoML model using the `Classification Evaluation Component`.
- Import the classification metrics to the AutoML model resource.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Model Evaluation](https://cloud.google.com/vertex-ai/docs/evaluation/introduction).

&nbsp;&nbsp;&nbsp;Learn more about [AutoML Text](https://cloud.google.com/vertex-ai/docs/text-data/classification/prepare-data).


[Evaluating BatchPrediction results from a Custom Tabular classification model](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/model_evaluation/custom_tabular_classification_model_evaluation.ipynb)

```
In this tutorial, you train a scikit-learn RandomForest model, save it in Vertex AI Model Registry and learn how to evaluate it through a Vertex AI pipeline job using `google_cloud_pipeline_components`.

The steps performed include:

- Fetch the dataset from the public source.
- Preprocess the data locally and save test data in BigQuery.
- Train a RandomForest classification model locally using scikit-learn Python package.
- Create a custom container in Artifact Registry for predictions.
- Upload the model in Vertex AI Model Registry.
- Create and run a Vertex AI Pipeline that:
    - Imports the trained model into the pipeline.
    - Runs a `Batch Prediction` job on the test data in BigQuery.
    - Evaulates the model using the evaluation component from google-cloud-pipeline-components Python SDK.
    - Imports the classification metrics in to the model resource in Vertex AI Model Registry.
- Print and visualize the classification evaluation metrics.
- Clean up the resources created in this notebook.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Model Evaluation](https://cloud.google.com/vertex-ai/docs/evaluation/introduction).

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Training](https://cloud.google.com/vertex-ai/docs/training/custom-training).

