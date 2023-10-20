
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

&nbsp;&nbsp;&nbsp;Learn more about [Classification for tabular data](https://cloud.google.com/vertex-ai/docs/tabular-data/classification-regression/overview).


[Challenger vs Blessed methodology for model deployment into production](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/pipelines/challenger_vs_blessed_deployment_method.ipynb)

```
Learn how to construct a Vertex AI pipeline, which trains a new challenger version of a model, evaluates the model and compares the evaluation to the existing blessed model in production, to determine whether the challenger model becomes the blessed model for replacement in production.

The steps performed include:

- Import a pretrained (blessed) model to the `Vertex AI Model Registry`.
- Import synthetic model evaluation metrics to the corresponding (blessed) model.
- Create a `Vertex AI Endpoint` resource
- Deploy the blessed model to the `Endpoint` resource.
- Create a Vertex AI Pipeline
    - Get the blessed model.
    - Import another instance (challenger) of the pretrained model.
    - Register the pretrained (challenger) model as a new version of the existing blessed model.
    - Create a synthetic model evaluation.
    - Import the synthetic model evaluation metrics to the corresponding challenger model.
    - Compare the evaluations and set the blessed or challenger as the default.
    - Deploy the new blessed model.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction).

&nbsp;&nbsp;&nbsp;Learn more about [Model evaluation in Vertex AI](https://cloud.google.com/vertex-ai/docs/evaluation/introduction).


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

&nbsp;&nbsp;&nbsp;Learn more about [Custom training components](https://cloud.google.com/vertex-ai/docs/training/create-training-pipeline).


[Training and batch prediction with BigQuery source and destination for a custom tabular classification model](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/pipelines/custom_tabular_train_batch_pred_bq_pipeline.ipynb)

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


[Get started with Vertex AI Hyperparameter Tuning pipeline components](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/pipelines/get_started_with_hpt_pipeline_components.ipynb)

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

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction).

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Hyperparameter Tuning](https://cloud.google.com/vertex-ai/docs/training/hyperparameter-tuning-overview).


[Get started with machine management for Vertex AI Pipelines](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/pipelines/get_started_with_machine_management.ipynb)

```
Learn how to convert a self-contained custom training component into a `Vertex AI CustomJob`, whereby:

The steps performed in this tutorial include:

- Create a custom component with a self-contained training job.
- Execute pipeline using component-level settings for machine resources
- Convert the self-contained training component into a `Vertex AI CustomJob`.
- Execute pipeline using customjob-level settings for machine resources

```


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

&nbsp;&nbsp;&nbsp;Learn more about [Regression for tabular data](https://cloud.google.com/vertex-ai/docs/tabular-data/classification-regression/overview).


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

&nbsp;&nbsp;&nbsp;Learn more about [Custom training components](https://cloud.google.com/vertex-ai/docs/pipelines/customjob-component).


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


[Vertex AI Pipelines with KFP 2.x](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/pipelines/kfp2_pipeline.ipynb)

```
Learn to use `Vertex AI Pipelines` and KFP 2.

The steps performed include:

- Create a KFP pipeline:
    - Create a `BigQuery Dataset` resource.
    - Export the dataset.
    - Train an XGBoost `Model` resource.
    - Create an `Endpoint` resource.
    - Deploys the `Model` resource to the `Endpoint` resource.
- Compile the KFP pipeline.
- Execute the KFP pipeline using `Vertex AI Pipelines`

```


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


[Multicontender vs Champion methodology for model deployment into production](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/pipelines/multicontender_vs_champion_deployment_method.ipynb)

```
Learn how to construct a Vertex AI pipeline, which evaluates new production data from a deployed  model against other versions  of the model, to determine if a contender model becomes the champion model for replacement in production.

The steps performed include:

- Import a pretrained (champion) model to the `Vertex AI Model Registry`.
- Import synthetic model training evaluation metrics to the corresponding (champion) model.
- Create a `Vertex AI Endpoint` resource
- Deploy the champion model to the `Endpoint` resource.
- Import additional (contender) versions of the deployed model.
- Import synthetic model training evaluation metrics to the corresponding (contender) models.
- Create a Vertex AI Pipeline
    - Get the champion model.
    - (Fake) Fine-tune champion model with production data
    - Import synthetic train+production evaluation metrics for the champion model.
    - Get the contender models.
    - (Fake) Fine-tune contender model with production data
    - Import synthetic train+production evaluation metrics for the contenders modesl.
    - Compare the evaluations of the contenders to the champion and set the new champion as the default.
    - Deploy the new champion model.

```


[Pipelines introduction for KFP](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/pipelines/pipelines_intro_kfp.ipynb)

```
Learn how to use the KFP SDK to build pipelines that generate evaluation metrics.

The steps performed include:

- Define and compile a `Vertex AI` pipeline.
- Specify which service account to use for a pipeline run.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction).


[BQML and AutoML - Experimenting with Vertex AI](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/pipelines/rapid_prototyping_bqml_automl.ipynb)

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

&nbsp;&nbsp;&nbsp;Learn more about [AutoML components](https://cloud.google.com/vertex-ai/docs/pipelines/vertex-automl-component).

&nbsp;&nbsp;&nbsp;Learn more about [BigQuery ML components](https://cloud.google.com/vertex-ai/docs/pipelines/bigqueryml-component).

