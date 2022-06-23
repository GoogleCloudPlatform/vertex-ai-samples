# Google Cloud Vertex AI Official Notebooks

The official notebooks are a collection of curated and non-curated notebooks authored by Google Cloud staff members. The curated notebooks are linked to in the [Vertex AI online web documentation](https://cloud.google.com/vertex-ai/docs/tutorials/jupyter-notebooks).

The official notebooks are organized by Google Cloud Vertex AI services.

## Manifest of Curated Notebooks

### AutoML

[AutoML text classification model](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/automl/automl-text-classification.ipynb)

<blockquote>
In this tutorial, you learn how to use `AutoML` to train a text classification model.

This tutorial uses the following Google Cloud ML services:

- `AutoML Training`
- `Vertex AI Model resource`

The steps performed include:

- Create a `Vertex AI Dataset`
- Train an `AutoML` text classification `Model` resource.
- Obtain the evaluation metrics for the `Model` resource.
- Create an `Endpoint` resource.
- Deploy the `Model` resource to the `Endpoint` resource.
- Make an online prediction.
- Make a batch prediction.
</blockquote>

[AutoML tabular forecasting model](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/automl/sdk_automl_tabular_forecasting_batch.ipynb)

<blockquote>
In this tutorial, you create an `AutoML` tabular forecasting model from a Python script, and then do a batch prediction using the Vertex AI SDK. 

This tutorial uses the following Google Cloud ML services:

- `AutoML Training`
- `Vertex AI Batch Prediction`
- `Vertex AI Model` resource

The steps performed include:

- Create a `Vertex AI Dataset` resource.
- Train an `AutoML` tabular forecasting `Model` resource.
- Obtain the evaluation metrics for the `Model` resource.
- Make a batch prediction.
</blockquote>

### Vertex AI Training

[Custom image classification model training and batch prediction](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/custom/sdk-custom-image-classification-batch.ipynb)

<blockquote>
In this tutorial, you learn to use `Vertex AI Training` to create a custom trained model and use `Vertex AI Batch Prediction` to do a batch prediction on the trained model.

This tutorial uses the following Google Cloud ML services:

- `Vertex AI Training`
- `Vertex AI Batch Prediction`
- `Vertex AI Model` resource

The steps performed include:

- Create a `Vertex AI` custom job for training a TensorFlow model.
- Upload the trained model artifacts as a `Model` resource.
- Make a batch prediction.
</blockquote>

[Custom image classification model training and online prediction](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/custom/sdk-custom-image-classification-online.ipynb)

<blockquote>
In this tutorial, you learn to use `Vertex AI Training` to create a custom-trained model from a Python script in a Docker container, and learn to use `Vertex AI Prediction` to do a prediction on the deployed model by sending data. 

This tutorial uses the following Google Cloud ML services:

- `Vertex AI Training`
- `Vertex AI Prediction`
- `Vertex AI Model` resource
- `Vertex AI Endpoint` resource

The steps performed include:

- Create a `Vertex AI` custom job for training a TensorFlow model.
- Upload the trained model artifacts to a `Model` resource.
- Create a serving `Endpoint` resource.
- Deploy the Model resource to a serving `Endpoint` resource.
- Make a prediction.
- Undeploy the `Model` resource.
</blockquote>

### Vertex Explainable AI

[AutoML tabular binary classification model with batch explanations](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/explainable_ai/sdk_automl_tabular_binary_classification_batch_explain.ipynb)

<blockquote>
In this tutorial, you learn to use `AutoML` to create a tabular binary classification model from a Python script, and then learn to use `Vertex AI Batch Prediction` to make predictions with explanations.

This tutorial uses the following Google Cloud ML services:

- `Vertex AI AutoML`
- `Vertex AI Batch Prediction`
- `Vertex Explainable AI`
- `Vertex AI Model` resource

The steps performed include:

- Create a `Vertex Dataset` resource.
- Train an `AutoML` tabular binary classification model.
- View the model evaluation metrics for the trained model.
- Make a batch prediction request with explainability.
</blockquote>

[AutoML tabular binary classification model with online explanations](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/explainable_ai/sdk_automl_tabular_classification_online_explain.ipynb)

<blockquote>
In this tutorial, you learn to use `AutoML` to create a tabular binary classification model from a Python script, and then learn to use `Vertex AI Online Prediction` to make online predictions with explanations. 

This tutorial uses the following Google Cloud ML services:

- `Vertex AI AutoML`
- `Vertex AI Prediction`
- `Vertex Explainable AI`
- `Vertex AI Model` resource
- `Vertex AI Endpoint` resource

The steps performed include:

- Create a `Vertex AI Dataset` resource.
- Train an `AutoML` tabular binary classification model.
- View the model evaluation metrics for the trained model.
- Create a serving `Endpoint` resource.
- Deploy the `Model` resource to a serving `Endpoint` resource.
- Make an online prediction request with explainability.
- Undeploy the `Model` resource.
</blockquote>

[Custom tabular regression model with batch explanations](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/explainable_ai/sdk_custom_tabular_regression_batch_explain.ipynb)

<blockquote>
In this tutorial, you learn to use `Vertex AI Training` and `Explainable AI` to create a custom image classification model with explanations, and then you learn to use `Vertex AI Batch Prediction` to make a batch prediction request with explanations. 

This tutorial uses the following Google Cloud ML services:

- `Vertex AI Training`
- `Vertex AI Batch Prediction`
- `Vertex Explainable AI`
- `Vertex AI Mode`l resource

The steps performed include:

- Create a `Vertex AI` custom job for training a TensorFlow model.
- View the model evaluation for the trained model.
- Set explanation parameters for when the model is deployed.
- Upload the trained model artifacts and explanations as a `Model` resource.
- Make a batch prediction with explanations.
</blockquote>
    
[Custom tabular regression model with online explanations](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/explainable_ai/sdk_custom_tabular_regression_online_explain.ipynb)

<blockquote>
In this tutorial, you learn to use `Vertex AI Training` and `Explainable AI` to create a custom image classification model with explanations, and then you learn to use `Vertex AI Prediction` to make an online prediction request with explanations.

This tutorial uses the following Google Cloud ML services:

- `Vertex AI Training`
- `Vertex AI Prediction`
- `Vertex Explainable AI`
- `Vertex AI Model` resource
- `Vertex AI Endpoint` resource

The steps performed include:

- Create a `Vertex AI` custom job for training a TensorFlow model.
- View the model evaluation for the trained model.
- Set explanation parameters for when the model is deployed.
- Upload the trained model artifacts and explanations as a `Model` resource.
- Create a serving `Endpoint` resource.
- Deploy the `Model` resource to a serving `Endpoint` resource.
- Make a prediction with explanation.
- Undeploy the `Model` resource.
</blockquote>
    
[Custom image classification model with batch explanations](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/explainable_ai/sdk_custom_image_classification_batch_explain.ipynb)

<blockquote>
In this tutorial, you learn to use `Vertex AI Training and Explainable AI` to create a custom image classification model with explanations, and then you learn to use `Vertex AI Batch Prediction` to make a batch prediction request with explanations. 

This tutorial uses the following Google Cloud ML services:

- `Vertex AI Training`
- `Vertex AI Batch Prediction`
- `Vertex Explainable AI`
- `Vertex AI Model` resource

The steps performed include:

- Create a `Vertex AI` custom job for training a TensorFlow model.
- View the model evaluation for the trained model.
- Set explanation parameters for when the model is deployed.
- Upload the trained model artifacts and explanation parameters as a `Model` resource.
- Make a batch prediction with explanations.
</blockquote>

[Custom image classification model with online explanations](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/explainable_ai/sdk_custom_image_classification_online_explain.ipynb)

<blockquote>
In this tutorial, you learn to use `Vertex AI Training and Explainable AI` to create a custom image classification model with explanations, and then you learn to use `Vertex AI Prediction` to make an online prediction request with explanations. 

This tutorial uses the following Google Cloud ML services:

- `Vertex AI Training`
- `Vertex AI Online Prediction`
- `Vertex Explainable AI`
- `Vertex AI Model` resource
- `Vertex AI Endpoint` resource

The steps performed include:

- Create a `Vertex AI` custom job for training a TensorFlow model.
- View the model evaluation for the trained model.
- Set explanation parameters for when the model is deployed.
- Upload the trained model artifacts and explanations as a `Model` resource.
- Create a serving `Endpoint` resource.
- Deploy the `Model` resource to a serving `Endpoint` resource.
- Make a prediction with explanation.
- Undeploy the `Model` resource.
</blockquote>

### Vertex Feature Store

[Managing features in a feature store](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/feature_store/gapic-feature-store.ipynb)

<blockquote>
In this notebook, you will learn how to use `Vertex AI Feature Store` to import feature data, and to access the feature data for both online serving and offline tasks, such as training.

This tutorial uses the following Google Cloud ML services:

- `Vertex AI Feature Store`
    
The steps performed include:

- Create featurestore, entity type, and feature resources.
- Import feature data into `Vertex AI Feature Store` resource.
- Serve online prediction requests using the imported features.
- Access imported features in offline jobs, such as training jobs.

</blockquote>

### Vertex Model Monitoring

[Monitoring drift detection in online serving](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/model_monitoring/model_monitoring.ipynb)

<blockquote>
In this notebook, you learn to use the `Vertex AI Model Monitoring` service to detect drift and anomalies in prediction requests from a deployed `Vertex AI Model` resource.

This tutorial uses the following Google Cloud ML services:

- `Vertex AI Model Monitoring`
- `Vertex AI Prediction`
- `Vertex AI Model` resource
- `Vertex AI Endpoint` resource

The steps performed include:

- Upload a pre-trained model as a `Vertex AI Model` resource.
- Create an `Vertex AI Endpoint` resource.
- Deploy the `Model` resource to the `Endpoint` resource.
- Configure the `Endpoint` resource for model monitoring.
- Generate synthetic prediction requests.
- Understand how to interpret the statistics, visualizations, other data reported by the model monitoring feature.
</blockquote>
    
### Vertex ML Metadata

[Tracking hyperparameters and metrics in custom training job](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/ml_metadata/sdk-metric-parameter-tracking-for-custom-jobs.ipynb)

<blockquote>
In this notebook, you learn how to use `Vertex ML Metadata` to track training parameters and evaluation metrics.

This tutorial uses the following Google Cloud ML services:

- `Vertex ML Metadata`
- `Vertex AI Experiments`

The steps performed include:

- Track parameters and metrics for a `Vertex AI` custom trained model.
- Extract and perform analysis for all parameters and metrics within an Experiment.
</blockquote>

[Tracking hyperparameters and metrics in locally trained job](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/ml_metadata/sdk-metric-parameter-tracking-for-locally-trained-models.ipynb)

<blockquote>
In this notebook, you learn how to use `Vertex ML Metadata` to track training parameters and evaluation metrics.

This tutorial uses the following Google Cloud ML services:

- `Vertex ML Metadata`
- `Vertex AI Experiments`

The steps performed include:

- Track parameters and metrics for a locally trained model.
- Extract and perform analysis for all parameters and metrics within an Experiment.
</blockquote>
    
### Vertex AI Pipelines

[Creating Python function KFP components](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/pipelines/lightweight_functions_component_io_kfp.ipynb)

<blockquote>
In this tutorial, you learn to use the KFP SDK to build lightweight Python function-based components, and then you learn to use `Vertex AI Pipelines` to execute the pipeline.

This tutorial uses the following Google Cloud ML services:

- `Vertex AI Pipelines`

The steps performed include:

- Build Python function-based KFP components.
- Construct a KFP pipeline.
    - Pass Artifacts and parameters between components, both by path reference and by value.
    - Use the kfp.dsl.importer method.
- Compile the KFP pipeline.
- Execute the KFP pipeline using `Vertex AI Pipelines`
</blockquote>
    
[AutoML image classification model pipeline](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/pipelines/google_cloud_pipeline_components_automl_images.ipynb)

<blockquote>
In this tutorial, you learn to use `Vertex AI Pipelines` and `Google Cloud Pipeline Components` to build an `AutoML` image classification model.

This tutorial uses the following Google Cloud ML services:

- `Vertex AI Pipelines`
- `Google Cloud Pipeline Components`
- `Vertex AutoML`
- `Vertex AI Model` resource
- `Vertex AI Endpoint` resource

The steps performed include:

- Create a KFP pipeline:
    - Create a `Vertex AI Dataset` resource.
    - Train an `AutoML` image classification `Model` resource.
    - Create an `Endpoint` resource.
    - Deploys the `Model` resource to the `Endpoint` resource.
- Compile the KFP pipeline.
- Execute the KFP pipeline using `Vertex AI Pipelines`
</blockquote>
    
[AutoML tabular classification model pipeline](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/pipelines/automl_tabular_classification_beans.ipynb)

<blockquote>
In this tutorial, you learn to use `Vertex AI Pipelines` and `Google Cloud Pipeline Components` to build an AutoML tabular classification model.

This tutorial uses the following Google Cloud ML services:

- `Vertex AI Pipelines`
- `Google Cloud Pipeline Components`
- `Vertex AutoML`
- `Vertex AI Model` resource
- `Vertex AI Endpoint` resource

The steps performed include:

- Create a KFP pipeline:
    - Create a `Dataset` resource.
    - Train an `AutoML` tabular classification `Model` resource.
    - Create an `Endpoint` resource.
    - Deploys the `Model` resource to the `Endpoint` resource.
- Compile the KFP pipeline.
- Execute the KFP pipeline using `Vertex AI Pipelines`
</blockquote>
    
[AutoML tabular regression model pipeline](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/pipelines/google_cloud_pipeline_components_automl_tabular.ipynb)

<blockquote>
In this tutorial, you learn to use `Vertex AI Pipelines` and `Google Cloud Pipeline Components` to build an `AutoML` tabular regression model.

This tutorial uses the following Google Cloud ML services:

- `Vertex AI Pipelines`
- `Google Cloud Pipeline Components`
- `Vertex AutoML`
- `Vertex AI Model` resource
- `Vertex AI Endpoint` resource

The steps performed include:

- Create a KFP pipeline:
    - Create a `Dataset` resource.
    - Train an `AutoML` tabular regression `Model` resource.
    - Create an `Endpoint` resource.
    - Deploys the `Model` resource to the `Endpoint` resource.
- Compile the KFP pipeline.
- Execute the KFP pipeline using `Vertex AI Pipelines`
</blockquote>

[AutoML text classification model pipeline](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/pipelines/google_cloud_pipeline_components_automl_text.ipynb)

<blockquote>
In this tutorial, you learn to use `Vertex AI Pipelines` and `Google Cloud Pipeline Components` to build an `AutoML` text classification model.

This tutorial uses the following Google Cloud ML services:

- `Vertex AI Pipelines`
- `Google Cloud Pipeline Components`
- `Vertex AutoML`
- `Vertex AI Model` resource
"- `Vertex AI Endpoint` resource

The steps performed include:

- Create a KFP pipeline:
    - Create a `Dataset` resource.
    - Train an AutoML text classification `Model` resource.
    - Create an `Endpoint` resource.
    - Deploys the `Model` resource to the `Endpoint` resource.
- Compile the KFP pipeline.
- Execute the KFP pipeline using `Vertex AI Pipelines`
</blockquote>

[Custom training and batch prediction using prebuilt components pipeline](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/pipelines/custom_model_training_and_batch_prediction.ipynb)

<blockquote>
In this tutorial, you learn to use `Vertex AI Pipelines` and `Google Cloud Pipeline Components` to build a custom model.

This tutorial uses the following Google Cloud ML services:

- `Vertex AI Pipelines`
- `Google Cloud Pipeline Components`
- `Vertex AI Training`
- `Vertex AI Model` resource
- `Vertex AI Endpoint` resource

The steps performed include:

- Create a KFP pipeline:
    - Train a custom model.
    - Upload the trained model as a `Model` resource.
    - Create an `Endpoint` resource.
- Deploy the `Model` resource to the `Endpoint` resource.
- Make a batch prediction request.
</blockquote>

[Custom training using prebuilt and custom components pipeline](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/pipelines/google_cloud_pipeline_components_model_train_upload_deploy.ipynb)

<blockquote>
In this tutorial, you learn to use `Vertex AI Pipelines` and `Google Cloud Pipeline Components` to build and deploy a custom model.

This tutorial uses the following Google Cloud ML services:

- `Vertex AI Pipelines`
- `Google Cloud Pipeline Components`
- `Vertex AI Training`
- `Vertex AI Model` resource
- `Vertex AI Endpoint` resource

The steps performed include:

- Create a KFP pipeline:
    - Train a custom model.
    - Uploads the trained model as a `Model` resource.
    - Creates an `Endpoint` resource.
    - Deploys the `Model` resource to the `Endpoint` resource.
- Compile the KFP pipeline.
- Execute the KFP pipeline using `Vertex AI Pipelines`
</blockquote>
    
[Introduction to control flow in pipelines](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/pipelines/control_flow_kfp.ipynb)

<blockquote>
In this tutorial, you use the KFP SDK to build pipelines that use loops and conditionals, including nested examples.

This tutorial uses the following Google Cloud ML services:

- `Vertex AI Pipelines`

The steps performed include:

- Create a KFP pipeline:
    - Use control flow components
- Compile the KFP pipeline.
- Execute the KFP pipeline using `Vertex AI Pipelines`
</blockquote>

[Introduction to KFP components and pipelines](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/pipelines/pipelines_intro_kfp.ipynb)

<blockquote>
In this tutorial, you use the KFP SDK to build pipelines.

This tutorial uses the following Google Cloud ML services:

- `Vertex AI Pipelines`

The steps performed include:

- Define and compile a `Vertex AI` pipeline.
- Schedule a recurring pipeline run.
- Specify which service account to use for a pipeline run.
</blockquote>

### Vertex AI Vizier

[Using Vizier for multi-objective study](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/vizier/gapic-vizier-multi-objective-optimization.ipynb)




 
