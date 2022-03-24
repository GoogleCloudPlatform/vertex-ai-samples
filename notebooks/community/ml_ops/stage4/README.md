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




<img src='stage4.png'>

## Notebooks

### Get Started

[Get started with Google Artifact Registry](get_started_with_google_artifact_registry.ipynb)

```
The steps performed include:

- Creating a private Docker repository.
- Tagging a container image, specific to the private Docker repository.
- Pushing a container image to the private Docker repository.
- Pulling a container image from the private Docker repository.
- Deleting a private Docker repository.
```

Get started with Vertex Model Registry

[Get started with Vertex ML Metadata](get_started_with_vertex_ml_metadata.ipynb)

```
The steps performed include:

- Create a `Metadatastore` resource.
- Create (record)/List an `Artifact`, with artifacts and metadata.
- Create (record)/List an `Execution`.
- Create (record)/List a `Context`.
- Add `Artifact` to `Execution` as events.
- Add `Execution` and `Artifact` into the `Context`
- Delete `Artifact`, `Execution` and `Context`.
- Create and run a `Vertex AI Pipeline` ML workflow to train and deploy a scikit-learn model.
    - Create custom pipeline components that generate artifacts and metadata.
    - Compare Vertex AI Pipelines runs.
    - Trace the lineage for pipeline-generated artifacts.
    - Query your pipeline run metadata.
```

Get started with custom model evaluation

Get started with A/B Testing

[Get started with Vertex Explainable AI](get_started_with_vertex_xai.ipynb)

```
The steps performed include:

- Train an AutoML tabular model.
    - Do a batch prediction with explanations.
    - Do an online prediction with explanations.
- Train an custom TensorFlow tabular model.
    - Manually set configuration metadata.
    - Do a batch prediction with explanations.
    - Do an online prediction with explanations.
    - Automatically set configuration metadata.
- Train an custom TensorFlow image model.
    - Manually set configuration metadata.
    - Do a batch prediction with explanations.
    - Do an online prediction with explanations.
- Train an custom XGBoost tabular model.
    - Manually set configuration metadata.
    - Do an online prediction with explanations.
- Train an custom scikit-learn tabular model.
    - Manually set configuration metadata.
    - Do an online prediction with explanations.
```

### E2E Stage Example

Stage 4: Evaluation
