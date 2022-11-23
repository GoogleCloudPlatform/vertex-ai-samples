# Stage 4: Evaluation

## Purpose

Evaluate a candidate model for exceeding performance and business objectives to become the next blessed model.

## Recommendations  

The fourth stage in MLOps is evaluation where a candidate model is evaluated to be the next deployed “blessed” version of the model in production. The candidate model may be either the first candidate version or a subsequent update. In the former, the model is evaluated against a preset baseline. In the latter, the model is evaluated against the existing deployed “blessed” model. If the evaluation slices have not changed since the existing deployed “blessed” model was evaluated, it’s previous evaluation data is used for comparison. Otherwise, the existing deployed “blessed” model is evaluated on the existing evaluation slices.

In the case were a custom model and AutoML model are compared, the “apples-to-apples” evaluation is handled through a batch script, which does the following:

1. Fetches the custom evaluation slices.
2. Creates a batch prediction job for each custom evaluation slice.
3. Applies custom evaluation on the results from the batch prediction job.


Once this static evaluation is completed, one or more dynamic evaluations are performed before deploying a candidate model as the next blessed model. These include evaluating for:

1. Resource utilization and latency in a sandbox environment. 
2. A business objective using A/B testing.

This stage may be done entirely by MLOps. We recommend:

- Store and retrieve candidate models in Vertex Model Registry.
- Use Vertex ML Metadata to retain past and present evaluations, per candidate and blessed versions, as part of the model metadata.
- Tag the evaluations by evaluation slice and version of the slice.
- Evaluate with the most recent evaluation slices.
- Use Vertex Batch Prediction to perform a batch prediction request on each evaluation slice.
- Use a Python batch script to compute custom evaluations from the results of the batch prediction request.
- Use the Vertex pipeline resource developed during formalization to perform the above.
- Before blessing a candidate model, deploy the model to an sandbox environment that is identical (or comparable) to the production environment of the existing blessed model. Send copies of live product request traffic to the sandbox environment. Measure and comparable resource utilization and latency. Use Vertex ML - Metadata to retain past and present sandbox performance results.
- Collect a random sample of prediction requests/results from the sandbox environment. Use Vertex Explainable AI to inspect the reason the prediction was made. This is generally a manual inspection process. Look for things like the right prediction for the wrong reason, and biases.
- Use Vertex Endpoint resource with traffic split to perform A/B testing between the existing blessed model and candidate model against a business objective.  - - Use Vertex ML Metadata to retain past and present A/B testing results.




<img src='stage4v3.png'>

## Notebooks

### Get Started


[Get started with Vertex AI Model Registry](community/ml_ops/stage3/get_started_with_model_registry.ipynb)

In this tutorial, you learn how to use `Vertex AI Model Registry` to create and register multiple versions of a model.

The steps performed include:

- Create and register a first version of a model to `Vertex AI Model Registry`.
- Create and register a second version of a model to `Vertex AI Model Registry`.
- Updating the model version which is the default (blessed).
- Deleting a model version.
- Retraining the next model version.

[Get started with Dataflow pipeline components](community/ml_ops/stage3/get_started_with_dataflow_pipeline_components.ipynb)

In this tutorial, you learn how to use prebuilt `Google Cloud Pipeline Components` for `Dataflow`.

The steps performed include:

- Build an Apache Beam data pipeline.
- Encapsulate the Apache Beam data pipeline with a Dataflow component in a Vertex AI pipeline.
- Execute a Vertex AI pipeline.

[Get started with Apache Airflow and Vertex AI Pipelines](community/ml_ops/stage3/get_started_with_airflow_and_vertex_pipelines.ipynb)

In this tutorial, you learn how to use Apache Airflow with `Vertex AI Pipelines`.

The steps performed include:

- Create Cloud Composer environment.
- Upload Airflow DAG to Composer environment that performs data processing -- i.e., creates a BigQuery table from a CSV file.
- Create a `Vertex AI Pipeline` that triggers the Airflow DAG.
- Execute the `Vertex AI Pipeline`.

[Get started with Kubeflow Pipelines](community/ml_ops/stage3/get_started_with_kubeflow_pipelines.ipynb)

In this tutorial, you learn how to use `Kubeflow Pipelines`(KFP).

The steps performed include:

- Building KFP lightweight Python function components.
- Assembling and compiling KFP components into a pipeline.
- Executing a KFP pipeline using Vertex AI Pipelines.
- Loading component and pipeline definitions from a source code repository.
- Building sequential, parallel, multiple output components.
- Building control flow into pipelines.

[Get started with Vertex AI custom training pipeline components](community/ml_ops/stage3/get_started_with_custom_training_pipeline_components.ipynb)

In this tutorial, you learn how to use prebuilt `Google Cloud Pipeline Components` for `Vertex AI Training`.

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

[Get started with Dataproc Serverless pipeline components](community/ml_ops/stage3/get_started_with_dataproc_serverless_pipeline_components.ipynb)


In this tutorial, you learn how to use prebuilt `Google Cloud Pipeline Components` for `Dataproc Serverless` service. 


The steps performed include:

- `DataprocPySparkBatchOp` for running PySpark batch workloads.
- `DataprocSparkBatchOp` for running Spark batch workloads.
- `DataprocSparkSqlBatchOp` for running Spark SQL batch workloads.
- `DataprocSparkRBatchOp` for running SparkR batch workloads.

[Get started with Vertex AI Hyperparameter Tuning pipeline components](community/ml_ops/stage3/get_started_with_hpt_pipeline_components.ipynb)

In this tutorial, you learn how to use prebuilt `Google Cloud Pipeline Components` for `Vertex AI Hyperparameter Tuning`.

The steps performed include:

- Construct a pipeline for:
    - Hyperparameter tune/train a custom model.
    - Retrieve the tuned hyperparameter values and metrics to optimize.
    - If the metrics exceed a specified threshold.
      - Get the location of the model artifacts for the best tuned model.
      - Upload the model artifacts to a `Vertex AI Model` resource.
- Execute a Vertex AI pipeline.

[Get started with machine management for Vertex AI Pipelines](community/ml_ops/stage3/get_started_with_machine_management.ipynb)

In this tutorial, you convert a self-contained custom training component into a `Vertex AI CustomJob`, whereby:

    - The training job and artifacts are trackable.
    - Set machine resources, such as machine-type, cpu/gpu, memory, disk, etc.

The steps performed in this tutorial include:

- Create a custom component with a self-contained training job.
- Execute pipeline using component-level settings for machine resources
- Convert the self-contained training component into a `Vertex AI CustomJob`.
- Execute pipeline using customjob-level settings for machine resources 

[Get started with TFX pipelines](community/ml_ops/stage3/get_started_with_tfx_pipeline.ipynb)

In this tutorial, you learn how to use TensorFlow Extended (TFX) with `Vertex AI Pipelines`.

The steps performed include:

- Create a TFX e2e pipeline.
- Execute the pipeline locally.
- Execute the pipeline on Google Cloud using `Vertex AI Training`
- Execute the pipeline using `Vertex AI Pipelines`.

[Get started with BigQuery ML pipeline components](community/ml_ops/stage3/get_started_with_bqml_pipeline_components.ipynb)

In this tutorial, you learn how to use prebuilt `Google Cloud Pipeline Components` for `BigQuery ML`.

The steps performed include:

- Construct a pipeline for:
    - Training BigQuery ML model.
    - Evaluating the BigQuery ML model.
    - Exporting the BigQuery ML model.
    - Importing the BigQuery ML model to a Vertex AI model.
    - Deploy the Vertex AI model.
- Execute a Vertex AI pipeline.
- Make a prediction with the deployed Vertex AI model.

[Get started with AutoML tabular pipeline workflows](community/ml_ops/stage3/get_started_with_automl_tabular_pipeline_workflow.ipynb)

In this tutorial, you learn how to use `AutoML Tabular Pipeline Template` for training, exporting and tuning an AutoML tabular model.

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

[Get started with rapid prototyping with AutoML and BigQuery ML](community/ml_ops/stage3/get_started_with_rapid_prototyping_bqml_automl.ipynb)

In this tutorial, you learn how to use `Vertex AI Predictions` for rapid prototyping a model.

The steps performed include:

- Creating a BigQuery and Vertex AI training dataset.
- Training a BigQuery ML and AutoML model.
- Extracting evaluation metrics from the BigQueryML and AutoML models.
- Selecting the best trained model.
- Deploying the best trained model.
- Testing the deployed model infrastructure.

[Get started with AutoML pipeline components](community/ml_ops/stage3/get_started_with_automl_pipeline_components.ipynb)

In this tutorial, you learn how to use prebuilt `Google Cloud Pipeline Components` for `Vertex AI AutoML`.

The steps performed include:

- Construct a pipeline for:
    - Training a Vertex AI AutoML trained model.
    - Test the serving binary with a batch prediction job.
    - Deploying a Vertex AI AutoML trained model.
- Execute a Vertex AI pipeline.


[Get started with BigQuery and TFDV pipeline components](community/ml_ops/stage3/get_started_with_bq_tfdv_pipeline_components.ipynb)

In this tutorial, you learn how to use build lightweight Python components for BigQuery and TensorFlow Data Validation.

The steps performed include:

- Build and execute a pipeline component for creating a Vertex AI Tabular Dataset from a BigQuery table.
- Build and execute a pipeline component for generating TFDV statistics and schema from a Vertex AI Tabular Dataset.
- Execute a Vertex AI pipeline.

### E2E Stage Example

Stage 4: Evaluation
