# Google Cloud Vertex AI Official Notebooks

The official notebooks are a collection of curated and non-curated notebooks authored by Google Cloud staff members. The curated notebooks are linked to in the [Vertex AI online web documentation](https://cloud.google.com/vertex-ai/docs/tutorials/jupyter-notebooks).

The official notebooks are organized by Google Cloud Vertex AI services.

## Manifest of Curated Notebooks

### AutoML  Text data 


[Create, train, and deploy an AutoML text classification model](official/automl/automl-text-classification.ipynb)

Learn how to use `AutoML` to train a text classification model.

The steps performed include:

* Create a `Vertex AI Dataset`.
* Train an `AutoML` text classification `Model` resource.
* Obtain the evaluation metrics for the `Model` resource.
* Create an `Endpoint` resource.
* Deploy the `Model` resource to the `Endpoint` resource.
* Make an online prediction
* Make a batch prediction

### AutoML  Tabular data 


[AutoML tabular forecasting model for batch prediction](official/automl/sdk_automl_tabular_forecasting_batch.ipynb)

Learn how to create an `AutoML` tabular forecasting model from a Python script, and then do a batch prediction using the Vertex AI SDK.

The steps performed include:

- Create a `Vertex AI Dataset` resource.
- Train an `AutoML` tabular forecasting `Model` resource.
- Obtain the evaluation metrics for the `Model` resource.
- Make a batch prediction.

### BigQuery ML  Vertex AI Model Registry  Batch prediction 


[Deploy BiqQuery ML Model on Vertex AI Model Registry and make predictions](official/model-registry/bqml-vertexai-model-registry.ipynb)

Learn how to use `Vertex AI Model Registry` with `BigQuery ML` and make batch predictions:

The steps performed include:

- Train a model with `BigQuery ML`
- Upload the model to `Vertex AI Model Registry` 
- Create a `Vertex AI Endpoint` resource
- Deploy the `Model` resource to the `Endpoint` resource
- Make `prediction` requests to the model endpoint
- Run `batch prediction` job on the `Model` resource 


### BigQuery ML  Vertex AI Model Registry  Online prediction 


[Online prediction with BigQuery ML](official/bigquery_ml/bqml-online-prediction.ipynb)

Learn how to train and deploy a churn prediction model for real-time inference, with the data in BigQuery and model trained using BigQuery ML, registered to Vertex AI Model Registry, and deployed to an endpoint on Vertex AI for online predictions.

The steps performed include:

- Using Python & SQL to query the public data in BigQuery
- Preparing the data for modeling
- Training a classification model using BigQuery ML and registering it to Vertex AI Model Registry
- Inspecting the model on Vertex AI Model Registry
- Deploying the model to an endpoint on Vertex AI
- Making sample online predictions to the model endpoint


### Custom Training 


[Custom training and batch prediction](official/custom/sdk-custom-image-classification-batch.ipynb)

Learn to use `Vertex AI Training` to create a custom trained model and use `Vertex AI Batch Prediction` to do a batch prediction on the trained model.

The steps performed include:

- Create a `Vertex AI` custom job for training a TensorFlow model.
- Upload the trained model artifacts as a `Model` resource.
- Make a batch prediction.

[Custom training and online prediction](official/custom/sdk-custom-image-classification-online.ipynb)

Learn to use `Vertex AI Training` to create a custom-trained model from a Python script in a Docker container, and learn to use `Vertex AI Prediction` to do a prediction on the deployed model by sending data.

The steps performed include:

- Create a `Vertex AI` custom job for training a TensorFlow model.
- Upload the trained model artifacts to a `Model` resource.
- Create a serving `Endpoint` resource.
- Deploy the `Model` resource to a serving `Endpoint` resource.
- Make a prediction.
- Undeploy the `Model` resource.

### Tabular Data 


[Compare Vertex AI Forecasting and BigQuery ML ARIMA_PLUS](official/automl/automl_forecasting_bqml_arima_plus_comparison.ipynb)

Learn how to create an BQML ARIMA_PLUS model using a training [Vertex AI Pipeline](https://cloud.

The steps performed are:

- Train the BQML ARIMA_PLUS model.
- View BQML model evaluation.
- Make a batch prediction with the BQML model.
- Create a Vertex AI `Dataset` resource.
- Train the Vertex AI Forecasting model.
- View the Model evaluation.
- Make a batch prediction with the Model.


### Vertex AI Experiments 


[Compare pipeline runs with Vertex AI Experiments](official/experiments/comparing_pipeline_runs.ipynb)

Learn how to use `Vertex AI Experiments` to log a pipeline job and compare different pipeline jobs.



[Build Vertex AI Experiment lineage for custom training](official/experiments/build_model_experimentation_lineage_with_prebuild_code.ipynb)

Learn how to integrate preprocessing code in a Vertex AI experiments.



[Track parameters and metrics for locally trained models](official/experiments/comparing_local_trained_models.ipynb)

Learn how to use Vertex AI Experiments to compare and evaluate model experiments.

The steps performed include:

- log the model parameters
- log the loss and metrics on every epoch to TensorBoard
- log the evaluation metrics


### Vertex AI Feature Store 


[Online and Batch predictions using Vertex AI Feature Store](official/feature_store/sdk-feature-store.ipynb)

Learn how to use `Vertex AI Feature Store` to import feature data, and to access the feature data for both online serving and offline tasks, such as training.

The steps performed include:

- Create featurestore, entity type, and feature resources.
- Import feature data into `Vertex AI Feature Store` resource.
- Serve online prediction requests using the imported features.
- Access imported features in offline jobs, such as training jobs.

### Matching Engine 


[Create Vertex AI Matching Engine index](official/matching_engine/sdk_matching_engine_for_indexing.ipynb)

Learn how to create Approximate Nearest Neighbor (ANN) Index, query against indexes, and validate the performance of the index.

The steps performed include:

* Create ANN Index and Brute Force Index
* Create an IndexEndpoint with VPC Network
* Deploy ANN Index and Brute Force Index
* Perform online query
* Compute recall


### Model Monitoring 


[Vertex AI Model Monitoring with Explainable AI Feature Attributions](official/model_monitoring/model_monitoring.ipynb)

Learn to use the `Vertex AI Model Monitoring` service to detect drift and anomalies in prediction requests from a deployed `Vertex AI Model` resource.

The steps performed include:

- Upload a pre-trained model as a `Vertex AI Model` resource.
- Create an `Vertex AI Endpoint` resource.
- Deploy the `Model` resource to the `Endpoint` resource.
- Configure the `Endpoint` resource for model monitoring.
- Initialize the baseline distribution for model monitoring.
- Generate synthetic prediction requests.
- Understand how to interpret the statistics, visualizations, other data reported by the model monitoring feature.

### Vertex AI Pipelines 


[Lightweight Python function-based components, and component I/O](official/pipelines/lightweight_functions_component_io_kfp.ipynb)

Learn to use the KFP SDK to build lightweight Python function-based components, and then you learn to use `Vertex AI Pipelines` to execute the pipeline.

The steps performed include:

- Build Python function-based KFP components.
- Construct a KFP pipeline.
- Pass *Artifacts* and *parameters* between components, both by path reference and by value.
- Use the `kfp.dsl.importer` method.
- Compile the KFP pipeline.
- Execute the KFP pipeline using `Vertex AI Pipelines`

### Vertex AI Pipelines  Tabular data 


[AutoML Tabular pipelines using google-cloud-pipeline-components](official/pipelines/automl_tabular_classification_beans.ipynb)

Learn to use `Vertex AI Pipelines` and `Google Cloud Pipeline Components` to build an `AutoML` tabular classification model.

The steps performed include:

- Create a KFP pipeline:
    - Create a `Dataset` resource.
    - Train an AutoML tabular classification `Model` resource.
    - Create an `Endpoint` resource.
    - Deploys the `Model` resource to the `Endpoint` resource.
- Compile the KFP pipeline.
- Execute the KFP pipeline using `Vertex AI Pipelines`



[AutoML tabular regression pipelines using google-cloud-pipeline-components](official/pipelines/google_cloud_pipeline_components_automl_tabular.ipynb)

Learn to use `Vertex AI Pipelines` and `Google Cloud Pipeline Components` to build an `AutoML` tabular regression model.

The steps performed include:

- Create a KFP pipeline:
    - Create a `Dataset` resource.
    - Train an AutoML tabular regression `Model` resource.
    - Create an `Endpoint` resource.
    - Deploys the `Model` resource to the `Endpoint` resource.
- Compile the KFP pipeline.
- Execute the KFP pipeline using `Vertex AI Pipelines`



### Vertex AI Pipelines 


[Custom training with pre-built Google Cloud Pipeline Components](official/pipelines/custom_model_training_and_batch_prediction.ipynb)

Learn to use `Vertex AI Pipelines` and `Google Cloud Pipeline Components` to build a custom model.

The steps performed include:

- Create a KFP pipeline:
    - Train a custom model.
    - Upload the trained model as a `Model` resource.
    - Create an `Endpoint` resource.
    - Deploy the `Model` resource to the `Endpoint` resource.
    - Make a batch prediction request.



[Pipeline control structures using the KFP SDK](official/pipelines/control_flow_kfp.ipynb)

Learn how to use the KFP SDK to build pipelines that use loops and conditionals, including nested examples.

The steps performed include:

- Create a KFP pipeline:
    - Use control flow components
- Compile the KFP pipeline.
- Execute the KFP pipeline using `Vertex AI Pipelines`

[Metrics visualization and run comparison using the KFP SDK](official/pipelines/metrics_viz_run_compare_kfp.ipynb)

Learn how to use the KFP SDK to build pipelines that generate evaluation metrics.

The steps performed include:

- Create KFP components:
    - Generate ROC curve and confusion matrix visualizations for classification results
    - Write metrics
- Create KFP pipelines.
- Execute KFP pipelines
- Compare metrics across pipeline runs

[Pipelines introduction for KFP](official/pipelines/pipelines_intro_kfp.ipynb)

Learn how to use the KFP SDK to build pipelines that generate evaluation metrics.

The steps performed include:

- Define and compile a `Vertex AI` pipeline.
- Specify which service account to use for a pipeline run.

### Vertex Explainable AI  Tabular data 


[AutoML training tabular binary classification model for batch explanation](official/explainable_ai/sdk_automl_tabular_binary_classification_batch_explain.ipynb)

Learn to use `AutoML` to create a tabular binary classification model from a Python script, and then learn to use `Vertex AI Batch Prediction` to make predictions with explanations.

The steps performed include:

- Create a `Vertex Dataset` resource.
- Train an `AutoML` tabular binary classification model.
- View the model evaluation metrics for the trained model.
- Make a batch prediction request with explainability.


* Prediction Service: Does an on-demand prediction for the entire set of instances (i.e., one or more data items) and returns the results in real-time.

* Batch Prediction Service: Does a queued (batch) prediction for the entire set of instances in the background and stores the results in a Cloud Storage bucket when ready.

### Vertex Explainable AI  Image data 


[Custom training image classification model for batch prediction with explainabilty](official/explainable_ai/sdk_custom_image_classification_batch_explain.ipynb)

Learn to use `Vertex AI Training and Explainable AI` to create a custom image classification model with explanations, and then you learn to use `Vertex AI Batch Prediction` to make a batch prediction request with explanations.

The steps performed include:

- Create a `Vertex AI` custom job for training a TensorFlow model.
- View the model evaluation for the trained model.
- Set explanation parameters for when the model is deployed.
- Upload the trained model artifacts and explanation parameters as a `Model` resource.
- Make a batch prediction with explanations.

### Vertex ML Metadata 


[Track parameters and metrics for custom training jobs](official/ml_metadata/sdk-metric-parameter-tracking-for-custom-jobs.ipynb)

Learn how to use Vertex AI SDK for Python to:

The steps performed include:
- Track training parameters and prediction metrics for a custom training job.
- Extract and perform analysis for all parameters and metrics within an Experiment.



 
