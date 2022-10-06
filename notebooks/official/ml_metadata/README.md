
[Track parameters and metrics for locally trained models](official/ml_metadata/sdk-metric-parameter-tracking-for-locally-trained-models.ipynb)

Learn how to use `Vertex ML Metadata` to track training parameters and evaluation metrics.

The steps performed include:

- Track parameters and metrics for a locally trained model.
- Extract and perform analysis for all parameters and metrics within an Experiment.

[Track parameters and metrics for custom training jobs](official/ml_metadata/sdk-metric-parameter-tracking-for-custom-jobs.ipynb)

Learn how to use Vertex AI SDK for Python to:

The steps performed include:
- Track training parameters and prediction metrics for a custom training job.
- Extract and perform analysis for all parameters and metrics within an Experiment.

[Track artifacts and metrics across Vertex AI Pipelines runs using Vertex ML Metadata](official/ml_metadata/vertex-pipelines-ml-metadata.ipynb)

Learn how to track artifacts and metrics with `Vertex ML Metadata` in `Vertex AI Pipeline` runs.

The steps performed include:

* Use the Kubeflow Pipelines SDK to build an ML pipeline that runs on Vertex AI
* The pipeline will create a dataset, train a scikit-learn model, and deploy the model to an endpoint
* Write custom pipeline components that generate artifacts and metadata
* Compare Vertex Pipelines runs, both in the Cloud console and programmatically
* Trace the lineage for pipeline-generated artifacts
* Query your pipeline run metadata
