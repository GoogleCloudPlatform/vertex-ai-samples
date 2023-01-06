# Stage 3: Formalization

## Purpose

Automate the production pipeline for continuous integration into ML operations, and where the pipeline continuously develops higher performing candidate models.


## Recommendations  

The third stage in MLOps is formalization to develop an automated pipeline process to generate candidate models. This stage may be done entirely by ML engineers, with the assistance of data scientists. We recommend:

- Encapsulate data and training procedures into automated pipelines using Vertex AI Pipelines.
- Use KFP for pipeline DAG generation, with the exception of a vast amount of unstructured data, we recommend TFX.
- Formalization is constructed as two independent, version controlled, pipelines: 1) data pipeline, 2) training pipeline.
- The model feeder in the data pipeline is a generator, which can take feedback from the training pipeline, for dynamically changing:
  - The batch size.
  - The subpopulation distribution to draw batches from.
  - The number of batches to prefetch.
  - Load balancing across mirrored data sources and shards – if not handled further upstream.
- Use Vertex AI ML Metadata for retrieving hyperparameters.
- Use Google Container Registry for custom training containers.
- Use Vertex AI Tensorboard for visualizing and monitoring training progress.
- Data and training scripts are versioned controlled.
- If the dataset statistics change (beyond a threshold), retune the hyperparameters.
- If domain-specific weight initialization changes, fine-tune the hyperparameters.
- If the model architecture changes, redo the formalization stage.
- Use early stop procedure in training script to detect failure to achieve training objective.
- Store the results of the trained model evaluation in Vertex AI ML Metadata.

<img src='stage3v3.png'>

## Notebooks

### Get Started


[Get started with Vertex AI Model Registry](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/ml_ops/stage3/get_started_with_model_registry.ipynb)

```
Learn how to use `Vertex AI Model Registry` to create and register multiple versions of a model.

The steps performed include:

- Create and register a first version of a model to `Vertex AI Model Registry`.
- Create and register a second version of a model to `Vertex AI Model Registry`.
- Updating the model version which is the default (blessed).
- Deleting a model version.
- Retraining the next model version.

```


[Get started with Dataflow pipeline components](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/ml_ops/stage3/get_started_with_dataflow_pipeline_components.ipynb)

```
Learn how to use prebuilt `Google Cloud Pipeline Components` for `Dataflow`.

The steps performed include:

- Build an Apache Beam data pipeline.
- Encapsulate the Apache Beam data pipeline with a Dataflow component in a Vertex AI pipeline.
- Execute a Vertex AI pipeline.

```


[Get started with Apache Airflow and Vertex AI Pipelines](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/ml_ops/stage3/get_started_with_airflow_and_vertex_pipelines.ipynb)

```
Learn how to use Apache Airflow with `Vertex AI Pipelines`.

The steps performed include:

- Create Cloud Composer environment.
- Upload Airflow DAG to Composer environment that performs data processing -- i.e., creates a BigQuery table from a CSV file.
- Create a `Vertex AI Pipeline` that triggers the Airflow DAG.
- Execute the `Vertex AI Pipeline`.

```


[Get started with Kubeflow Pipelines](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/ml_ops/stage3/get_started_with_kubeflow_pipelines.ipynb)

```
Learn how to use `Kubeflow Pipelines`(KFP).

The steps performed include:

- Building KFP lightweight Python function components.
- Assembling and compiling KFP components into a pipeline.
- Executing a KFP pipeline using Vertex AI Pipelines.
- Loading component and pipeline definitions from a source code repository.
- Building sequential, parallel, multiple output components.
- Building control flow into pipelines.

```


[Get started with Vertex AI custom training pipeline components](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/ml_ops/stage3/get_started_with_custom_training_pipeline_components.ipynb)

```
Learn how to use prebuilt `Google Cloud Pipeline Components` for `Vertex AI Training`.

The steps performed include:

- Construct a pipeline for:
    - Training a Vertex AI custom trained model.
    - Test the serving binary with a batch prediction job.
    - Deploying a Vertex AI custom trained model.
- Execute a Vertex AI pipeline.
- Construct a pipeline for:
     - Construct a custom training component.
     - Convert custom training component to CustomTrainingJobOp.
     - Training a Vertex AI custom trained model using the converted component.
    - Deploying a Vertex AI custom trained model.
- Execute a Vertex AI pipeline.

```


[Get started with Dataproc Serverless pipeline components](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/ml_ops/stage3/get_started_with_dataproc_serverless_pipeline_components.ipynb)

```
Learn how to use prebuilt `Google Cloud Pipeline Components` for `Dataproc Serverless` service.

The steps performed include:

- `DataprocPySparkBatchOp` for running PySpark batch workloads.
- `DataprocSparkBatchOp` for running Spark batch workloads.
- `DataprocSparkSqlBatchOp` for running Spark SQL batch workloads.
- `DataprocSparkRBatchOp` for running SparkR batch workloads.

```


[Get started with Vertex AI Hyperparameter Tuning pipeline components](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/ml_ops/stage3/get_started_with_hpt_pipeline_components.ipynb)

```
Learn how to use prebuilt `Google Cloud Pipeline Components` for `Vertex AI Hyperparameter Tuning`.

The steps performed include:

- Construct a pipeline for:
    - Hyperparameter tune/train a custom model.
    - Retrieve the tuned hyperparameter values and metrics to optimize.
    - If the metrics exceed a specified threshold.
      - Get the location of the model artifacts for the best tuned model.
      - Upload the model artifacts to a `Vertex AI Model` resource.
- Execute a Vertex AI pipeline.

```


[Get started with machine management for Vertex AI Pipelines](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/ml_ops/stage3/get_started_with_machine_management.ipynb)

```
Learn how to convert a self-contained custom training component into a `Vertex AI CustomJob`, whereby:

The steps performed in this tutorial include:

- Create a custom component with a self-contained training job.
- Execute pipeline using component-level settings for machine resources
- Convert the self-contained training component into a `Vertex AI CustomJob`.
- Execute pipeline using customjob-level settings for machine resources

```


[Get started with TFX pipelines](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/ml_ops/stage3/get_started_with_tfx_pipeline.ipynb)

```
Learn how to use TensorFlow Extended (TFX) with `Vertex AI Pipelines`.

The steps performed include:

- Create a TFX e2e pipeline.
- Execute the pipeline locally.
- Execute the pipeline on Google Cloud using `Vertex AI Training`
- Execute the pipeline using `Vertex AI Pipelines`.

```


[Orchestrating a workflow to train and deploy an scikit-learn model using Vertex AI Pipelines with online prediction](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/ml_ops/stage3/get_started_vertex_pipelines_sklearn_with_prediction.ipynb)

```
Learn how to use prebuilt components in `Vertex AI Pipelines` for training and deploying a scikit-Learn custom model, and then using `Vertex AI Prediction` to make an online prediction.

The steps performed include:

- Construct a scikit-learn training package.
- Construct a pipeline to train and deploy a scikit-learn model.
- Execute the pipeline.
- Make an online prediction.

```


[Get started with BigQuery ML pipeline components](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/ml_ops/stage3/get_started_with_bqml_pipeline_components.ipynb)

```
Learn how to use prebuilt `Google Cloud Pipeline Components` for `BigQuery ML`.

The steps performed include:

- Construct a pipeline for:
    - Training BigQuery ML model.
    - Evaluating the BigQuery ML model.
    - Exporting the BigQuery ML model.
    - Importing the BigQuery ML model to a Vertex AI model.
    - Deploy the Vertex AI model.
- Execute a Vertex AI pipeline.
- Make a prediction with the deployed Vertex AI model.

```


[Orchestrating a workflow to train and deploy an XGBoost model using Vertex AI Pipelines with Vertex AI Experiments](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/ml_ops/stage3/get_started_vertex_pipelines_xgboost_with_experiments.ipynb)

```
Learn how to use prebuilt components in `Vertex AI Pipelines` for training and deploying a XGBoost custom model, and using `Vertex AI Experiments` to log the corresponding training parameters and metrics, from within the training package.

The steps performed include:

- Construct a XGBoost training package.
  - Add tracking the experiment
- Construct a pipeline to train and deploy a XGBoost model.
- Execute the pipeline.

```


[Get started with AutoML tabular pipeline workflows](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/ml_ops/stage3/get_started_with_automl_tabular_pipeline_workflow.ipynb)

```
Learn how to use `AutoML Tabular Pipeline Template` for training, exporting and tuning an AutoML tabular model.

The steps performed include:

- Define training specification.
    - Dataset specification
    - Hyperparameter overide specification
    - machine specifications
- Construct tabular workflow pipeline.
- Compile and execute pipeline.
- View evaluation metrics artifact.
- Export AutoML model as an OSS TF model.
- Create `Endpoint` resource.
- Deploy exported OSS TF model.
- Make a prediction.

```


[Get started with rapid prototyping with AutoML and BigQuery ML](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/ml_ops/stage3/get_started_with_rapid_prototyping_bqml_automl.ipynb)

```
Learn how to use `Vertex AI Predictions` for rapid prototyping a model.

The steps performed include:

- Creating a BigQuery and Vertex AI training dataset.
- Training a BigQuery ML and AutoML model.
- Extracting evaluation metrics from the BigQueryML and AutoML models.
- Selecting the best trained model.
- Deploying the best trained model.
- Testing the deployed model infrastructure.

```


[Get started with AutoML pipeline components](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/ml_ops/stage3/get_started_with_automl_pipeline_components.ipynb)

```
Learn how to use prebuilt `Google Cloud Pipeline Components` for `Vertex AI AutoML`.

The steps performed include:

- Construct a pipeline for:
    - Training a Vertex AI AutoML trained model.
    - Test the serving binary with a batch prediction job.
    - Deploying a Vertex AI AutoML trained model.
- Execute a Vertex AI pipeline.

```


[Orchestrating a workflow to train and deploy an XGBoost model using Vertex AI Pipelines with online prediction](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/ml_ops/stage3/get_started_vertex_pipelines_xgboost_with_prediction.ipynb)

```
Learn how to use prebuilt components in `Vertex AI Pipelines` for training and deploying a XGBoost custom model, and then using `Vertex AI Prediction` to make an online prediction.

The steps performed include:

- Construct a XGBoost training package.
- Construct a pipeline to train and deploy a XGBoost model.
- Execute the pipeline.
- Make an online prediction.

```


[Get started with BigQuery and TFDV pipeline components](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/ml_ops/stage3/get_started_with_bq_tfdv_pipeline_components.ipynb)

```
Learn how to use build lightweight Python components for BigQuery and TensorFlow Data Validation.

The steps performed include:

- Build and execute a pipeline component for creating a Vertex AI Tabular Dataset from a BigQuery table.
- Build and execute a pipeline component for generating TFDV statistics and schema from a Vertex AI Tabular Dataset.
- Execute a Vertex AI pipeline.

```

### E2E Stage Example

[Formalization](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/ml_ops/stage3/mlops_formalization.ipynb)

```
In this tutorial, you create a MLOps stage 3: formalization process.

The steps performed include:

- Obtain resources from the experimentation stage.
    - Baseline model.
    - Dataset schema/statistics for baseline model.
- Formalize a data preprocessing pipeline.
    - Extract columns/rows from BigQuery table to local BigQuery table.
    - Use TensorFlow Data Validation library to determine statistics, schema, and features.
    - Use Dataflow to preprocess the data.
    - Create a Vertex AI Dataset.
- Formalize a build model architecture pipeline.
    - Create the Vertex AI Model base model.
- Formalize a training pipeline.

```

