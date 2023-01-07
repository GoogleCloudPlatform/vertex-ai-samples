
[AutoML image classification pipelines using google-cloud-pipeline-components](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/pipelines/google_cloud_pipeline_components_automl_images.ipynb)

```
Learn how to use `Vertex AI Pipelines` and `Google Cloud Pipeline Components` to build an `AutoML` image classification model.

The steps performed include:

- Create a KFP pipeline:
    - Create a `Dataset` resource.
    - Train an AutoML image classification `Model` resource.
    - Create an `Endpoint` resource.
    - Deploys the `Model` resource to the `Endpoint` resource.
- Compile the KFP pipeline.
- Execute the KFP pipeline using `Vertex AI Pipelines`

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction).

&nbsp;&nbsp;&nbsp;Learn more about [AutoML components](https://cloud.google.com/vertex-ai/docs/pipelines/vertex-automl-component).


[Metrics visualization and run comparison using the KFP SDK](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/pipelines/metrics_viz_run_compare_kfp.ipynb)

```
Learn how to use the KFP SDK to build pipelines that generate evaluation metrics.

The steps performed include:

- Create KFP components:
    - Generate ROC curve and confusion matrix visualizations for classification results
    - Write metrics
- Create KFP pipelines.
- Execute KFP pipelines
- Compare metrics across pipeline runs

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction).


[Lightweight Python function-based components, and component I/O](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/pipelines/lightweight_functions_component_io_kfp.ipynb)

```
Learn to use the KFP SDK to build lightweight Python function-based components, and then you learn to use `Vertex AI Pipelines` to execute the pipeline.

The steps performed include:

- Build Python function-based KFP components.
- Construct a KFP pipeline.
- Pass *Artifacts* and *parameters* between components, both by path reference and by value.
- Use the `kfp.dsl.importer` method.
- Compile the KFP pipeline.
- Execute the KFP pipeline using `Vertex AI Pipelines`

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction).


[Custom training with pre-built Google Cloud Pipeline Components](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/pipelines/custom_model_training_and_batch_prediction.ipynb)

```
Learn to use `Vertex AI Pipelines` and `Google Cloud Pipeline Components` to build a custom model.

The steps performed include:

- Create a KFP pipeline:
    - Train a custom model.
    - Upload the trained model as a `Model` resource.
    - Create an `Endpoint` resource.
    - Deploy the `Model` resource to the `Endpoint` resource.
    - Make a batch prediction request.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction).

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Training components](https://cloud.google.com/vertex-ai/docs/training/create-training-pipeline).


[AutoML Tabular pipelines using google-cloud-pipeline-components](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/pipelines/automl_tabular_classification_beans.ipynb)

```
Learn to use `Vertex AI Pipelines` and `Google Cloud Pipeline Components` to build an `AutoML` tabular classification model.

The steps performed include:

- Create a KFP pipeline:
    - Create a `Dataset` resource.
    - Train an AutoML tabular classification `Model` resource.
    - Create an `Endpoint` resource.
    - Deploys the `Model` resource to the `Endpoint` resource.
- Compile the KFP pipeline.
- Execute the KFP pipeline using `Vertex AI Pipelines`

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction).

&nbsp;&nbsp;&nbsp;Learn more about [AutoML components](https://cloud.google.com/vertex-ai/docs/pipelines/vertex-automl-component).


[Training an acquisition-prediction model using Swivel, BigQuery ML and Vertex AI Pipelines](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/pipelines/google_cloud_pipeline_components_bqml_text.ipynb)

```
Learn how to build a simple BigQuery ML pipeline using Vertex AI pipelines in order to calculate text embeddings of content from articles and classify them
into the *corporate acquisitions* category.

The steps performed include:

- Creating a component for Dataflow job that ingests data to BigQuery.
- Creating a component for preprocessing steps to run on the data in BigQuery.
- Creating a component for training a logistic regression model using BigQuery ML.
- Building and configuring a Kubeflow DSL pipeline with all the created components.
- Compiling and running the pipeline in Vertex AI Pipelines.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction).

&nbsp;&nbsp;&nbsp;Learn more about [BigQuery ML components](https://cloud.google.com/vertex-ai/docs/pipelines/bigqueryml-component).


[Pipelines introduction for KFP](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/pipelines/pipelines_intro_kfp.ipynb)

```
Learn how to use the KFP SDK to build pipelines that generate evaluation metrics.

The steps performed include:

- Define and compile a `Vertex AI` pipeline.
- Specify which service account to use for a pipeline run.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction).


[AutoML tabular regression pipelines using google-cloud-pipeline-components](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/pipelines/google_cloud_pipeline_components_automl_tabular.ipynb)

```
Learn to use `Vertex AI Pipelines` and `Google Cloud Pipeline Components` to build an `AutoML` tabular regression model.

The steps performed include:

- Create a KFP pipeline:
    - Create a `Dataset` resource.
    - Train an AutoML tabular regression `Model` resource.
    - Create an `Endpoint` resource.
    - Deploys the `Model` resource to the `Endpoint` resource.
- Compile the KFP pipeline.
- Execute the KFP pipeline using `Vertex AI Pipelines`

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction).

&nbsp;&nbsp;&nbsp;Learn more about [AutoML components](https://cloud.google.com/vertex-ai/docs/pipelines/vertex-automl-component).


[Model upload, predict, and evaluate using google-cloud-pipeline-components](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/pipelines/google_cloud_pipeline_components_model_upload_predict_evaluate.ipynb)

```
Learn how to evaluate a custom model using a pipeline with components from `google_cloud_pipeline_components` and a custom pipeline component you build.

The steps performed include:

- Upload a pre-trained model as a `Model` resource.
- Run a `BatchPredictionJob` on the `Model` resource with ground truth data.
- Generate evaluation `Metrics` artifact about the `Model` resource.
- Compare the evaluation metrics to a threshold.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction).

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Model components](https://cloud.google.com/vertex-ai/docs/pipelines/model-endpoint-component).


[Pipeline control structures using the KFP SDK](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/pipelines/control_flow_kfp.ipynb)

```
Learn how to use the KFP SDK to build pipelines that use loops and conditionals, including nested examples.

The steps performed include:

- Create a KFP pipeline:
    - Use control flow components
- Compile the KFP pipeline.
- Execute the KFP pipeline using `Vertex AI Pipelines`

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction).


[Loan eligibility prediction using `google-cloud-pipeline-components` and Spark ML](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/pipelines/google_cloud_pipeline_components_dataproc_tabular.ipynb)

```
Learn how to build a Vertex AI pipeline and train a random-forest model using Spark ML for loan-eligibility classification problem.

The steps performed include:

*   Use the `DataprocPySparkBatchOp` to preprocess data.
*   Create a Vertex AI dataset resource on the training data.
*   Train a random forest model using PySpark.
*   Build a Vertex AI pipeline and run the training job.
*   Use the Spark serving image in order to deploy a Spark model on Vertex AI Endpoint.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction).

&nbsp;&nbsp;&nbsp;Learn more about [Dataproc components](https://cloud.google.com/vertex-ai/docs/pipelines/dataproc-component).


[Model train, upload, and deploy using Google Cloud Pipeline Components](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/pipelines/google_cloud_pipeline_components_model_train_upload_deploy.ipynb)

```
Learn how to use `Vertex AI Pipelines` and `Google Cloud Pipeline Components` to build and deploy a custom model.

The steps performed include:

- Create a KFP pipeline:
    - Train a custom model.
    - Uploads the trained model as a `Model` resource.
    - Creates an `Endpoint` resource.
    - Deploys the `Model` resource to the `Endpoint` resource.
- Compile the KFP pipeline.
- Execute the KFP pipeline using `Vertex AI Pipelines`

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction).

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Training components](https://cloud.google.com/vertex-ai/docs/pipelines/customjob-component).


[AutoML text classification pipelines using google-cloud-pipeline-components](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/pipelines/google_cloud_pipeline_components_automl_text.ipynb)

```
Learn to use `Vertex AI Pipelines` and `Google Cloud Pipeline Components` to build an `AutoML` text classification model.

The steps performed include:

- Create a KFP pipeline:
    - Create a `Dataset` resource.
    - Train an AutoML text classification `Model` resource.
    - Create an `Endpoint` resource.
    - Deploys the `Model` resource to the `Endpoint` resource.
- Compile the KFP pipeline.
- Execute the KFP pipeline using `Vertex AI Pipelines`

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction).

&nbsp;&nbsp;&nbsp;Learn more about [AutoML components](https://cloud.google.com/vertex-ai/docs/pipelines/vertex-automl-component).


[Training and batch prediction with BigQuery source and destinantion for a custom tabular classification model](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/pipelines/custom_tabular_train_batch_pred_bq_pipeline.ipynb)

```
In this tutorial, you train a scikit-learn tabular classification model and create batch prediction job for it through a Vertex AI pipeline using `google_cloud_pipeline_components`.

The steps performed include:

- Create a dataset in BigQuery.
- Set some data aside from the source dataset for batch prediction.
- Create a custom python package for training application.
- Upload the python package to Cloud Storage.
- Create a Vertex AI Pipeline that:
    - creates a Vertex AI Dataset from the source dataset.
    - trains a scikit-learn RandomForest classification model on the dataset.
    - uploads the trained model to Vertex AI Model Registry.
    - runs a batch prediction job with the model on the test data.
- Check the prediction results from the destination table in BigQuery.
- Clean up the resources created in this notebook.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction).

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Batch Prediction components](https://cloud.google.com/vertex-ai/docs/pipelines/batchprediction-component).

