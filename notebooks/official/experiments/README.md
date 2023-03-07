
[Build Vertex AI Experiment lineage for custom training](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/experiments/build_model_experimentation_lineage_with_prebuild_code.ipynb)

```
Learn how to integrate preprocessing code in a Vertex AI experiments.

The steps performed include:

- Execute module for preprocessing data
  - Create a dataset artifact
  - Log parameters
-  Execute module for training the model
  - Log parameters
  - Create model artifact
  - Assign tracking lineage to dataset, model and parameters

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Experiments](https://cloud.google.com/vertex-ai/docs/experiments/intro-vertex-ai-experiments).

&nbsp;&nbsp;&nbsp;Learn more about [Vertex ML Metadata](https://cloud.google.com/vertex-ai/docs/ml-metadata).


[Track parameters and metrics for locally trained models](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/experiments/comparing_local_trained_models.ipynb)

```
Learn how to use Vertex AI Experiments to compare and evaluate model experiments.

The steps performed include:

- log the model parameters
- log the loss and metrics on every epoch to TensorBoard
- log the evaluation metrics

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Experiments](https://cloud.google.com/vertex-ai/docs/experiments/intro-vertex-ai-experiments).


[Compare pipeline runs with Vertex AI Experiments](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/experiments/comparing_pipeline_runs.ipynb)

```
Learn how to use `Vertex AI Experiments` to log a pipeline job and compare different pipeline jobs.

The steps performed include:

* Formalize a training component
* Build a training pipeline
* Run several Pipeline jobs and log their results
* Compare different Pipeline jobs

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Experiments](https://cloud.google.com/vertex-ai/docs/experiments/intro-vertex-ai-experiments).

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction).


[Get started with Vertex AI Experiments](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/experiments/get_started_with_vertex_experiments.ipynb)

```
Learn how to use `Vertex AI Experiments` when training with `Vertex AI`.

The steps performed include:

- Local (notebook) Training
    - Create an experiment
    - Create a first run in the experiment
    - Log parameters and metrics
    - Create artifact lineage
    - Visualize the experiment results
    - Execute a second run
    - Compare the two runs in the experiment
- Cloud (`Vertex AI`) Training
    - Within the training script:
        - Create an experiment
        - Log parameters and metrics
        - Create artifact lineage
    - Create a `Vertex AI Training` custom job
    - Execute the custom job
    - Visualize the experiment results

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Experiments](https://cloud.google.com/vertex-ai/docs/experiments/intro-vertex-ai-experiments).

&nbsp;&nbsp;&nbsp;Learn more about [Vertex ML Metadata](https://cloud.google.com/vertex-ai/docs/ml-metadata).

&nbsp;&nbsp;&nbsp;Learn more about [Custom training](https://cloud.google.com/vertex-ai/docs/training/custom-training).

