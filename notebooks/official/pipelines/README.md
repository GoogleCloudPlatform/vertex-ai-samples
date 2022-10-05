
[AutoML image classification pipelines using google-cloud-pipeline-components](official/pipelines/google_cloud_pipeline_components_automl_images.ipynb)

Learn how to use `Vertex AI Pipelines` and `Google Cloud Pipeline Components` to build an `AutoML` image classification model.

The steps performed include:

- Create a KFP pipeline:
    - Create a `Dataset` resource.
    - Train an AutoML image classification `Model` resource.
    - Create an `Endpoint` resource.
    - Deploys the `Model` resource to the `Endpoint` resource.
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

[Lightweight Python function-based components, and component I/O](official/pipelines/lightweight_functions_component_io_kfp.ipynb)

Learn to use the KFP SDK to build lightweight Python function-based components, and then you learn to use `Vertex AI Pipelines` to execute the pipeline.

The steps performed include:

- Build Python function-based KFP components.
- Construct a KFP pipeline.
- Pass *Artifacts* and *parameters* between components, both by path reference and by value.
- Use the `kfp.dsl.importer` method.
- Compile the KFP pipeline.
- Execute the KFP pipeline using `Vertex AI Pipelines`

[Custom training with pre-built Google Cloud Pipeline Components](official/pipelines/custom_model_training_and_batch_prediction.ipynb)

Learn to use `Vertex AI Pipelines` and `Google Cloud Pipeline Components` to build a custom model.

The steps performed include:

- Create a KFP pipeline:
    - Train a custom model.
    - Upload the trained model as a `Model` resource.
    - Create an `Endpoint` resource.
    - Deploy the `Model` resource to the `Endpoint` resource.
    - Make a batch prediction request.



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



[Training an acquisition-prediction model using Swivel, BigQuery ML and Vertex AI Pipelines](official/pipelines/google_cloud_pipeline_components_bqml_text.ipynb)

Learn how to build a simple BigQuery ML pipeline using Vertex AI pipelines in order to calculate text embeddings of content from articles and classify them
into the *corporate acquisitions* category.

The steps performed include:

- Creating a component for Dataflow job that ingests data to BigQuery.
- Creating a component for preprocessing steps to run on the data in BigQuery.
- Creating a component for training a logistic regression model using BigQuery ML.
- Building and configuring a Kubeflow DSL pipeline with all the created components.
- Compiling and running the pipeline in Vertex AI Pipelines.

[Pipelines introduction for KFP](official/pipelines/pipelines_intro_kfp.ipynb)

Learn how to use the KFP SDK to build pipelines that generate evaluation metrics.

The steps performed include:

- Define and compile a `Vertex AI` pipeline.
- Specify which service account to use for a pipeline run.

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



[Model upload, predict, and evaluate using google-cloud-pipeline-components](official/pipelines/google_cloud_pipeline_components_model_upload_predict_evaluate.ipynb)

Learn how to evaluate a custom model using a pipeline with components from `google_cloud_pipeline_components` and a custom pipeline component you build.

The steps performed include:

- Upload a pre-trained model as a `Model` resource.
- Run a `BatchPredictionJob` on the `Model` resource with ground truth data.
- Generate evaluation `Metrics` artifact about the `Model` resource.
- Compare the evaluation metrics to a threshold.


[Pipeline control structures using the KFP SDK](official/pipelines/control_flow_kfp.ipynb)

Learn how to use the KFP SDK to build pipelines that use loops and conditionals, including nested examples.

The steps performed include:

- Create a KFP pipeline:
    - Use control flow components
- Compile the KFP pipeline.
- Execute the KFP pipeline using `Vertex AI Pipelines`

[Loan eligibility prediction using `google-cloud-pipeline-components` and Spark ML](official/pipelines/google_cloud_pipeline_components_dataproc_tabular.ipynb)

Learn how to build a Vertex AI pipeline and train a Random-forest model using Spark ML for loan-eligibility classification problem.

The steps performed include:

*   Use the `DataprocPySparkBatchOp` to preprocess data.
*   Create a Vertex AI dataset resource on the training data.
*   Train a random forest model using Pyspark.
*   Build a Vertex AI pipeline and run the training job.
*   Use the Spark serving image in order to deploy a Spark model on Vertex AI Endpoint.

[AutoML text classification pipelines using google-cloud-pipeline-components](official/pipelines/google_cloud_pipeline_components_automl_text.ipynb)

Learn to use `Vertex AI Pipelines` and `Google Cloud Pipeline Components` to build an `AutoML` text classification model.

The steps performed include:

- Create a KFP pipeline:
    - Create a `Dataset` resource.
    - Train an AutoML text classification `Model` resource.
    - Create an `Endpoint` resource.
    - Deploys the `Model` resource to the `Endpoint` resource.
- Compile the KFP pipeline.
- Execute the KFP pipeline using `Vertex AI Pipelines`


