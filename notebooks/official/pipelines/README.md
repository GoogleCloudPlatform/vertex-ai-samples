# Vertex Pipeline examples

This directory holds [Vertex Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines) example notebooks.

- [pipelines_intro_kfp.ipynb](./pipelines_intro_kfp.ipynb) introduces some of the Vertex Pipelines features, using the [Kubeflow Pipelines (KFP) SDK](https://www.kubeflow.org/docs/components/pipelines/).

- [control_flow_kfp.ipynb](./control_flow_kfp.ipynb) shows how you can build pipelines that include conditionals and parallel 'for' loops using the KFP SDK.
- [lightweight_functions_component_io_kfp.ipynb](./lightweight_functions_component_io_kfp.ipynb) shows how to build lightweight Python function-based components, and in particular how to support component I/O using the KFP SDK.
- [metrics_viz_run_compare_kfp.ipynb](./metrics_viz_run_compare_kfp.ipynb) shows how to use the KFP SDK to build Vertex Pipelines that generate model metrics and metrics visualizations; and how to compare pipeline runs.

The following examples show how to use the components defined in [google_cloud_pipeline_components](https://github.com/kubeflow/pipelines/tree/master/components/google-cloud) to build pipelines that access [Vertex AI](https://cloud.google.com/vertex-ai/) services.

- [google-cloud-pipeline-components_automl_images.ipynb](./google-cloud-pipeline-components_automl_images.ipynb)
- [google-cloud-pipeline-components_automl_tabular.ipynb](./google-cloud-pipeline-components_automl_tabular.ipynb) (tabular regression model)
- [automl_tabular_classification_beans.ipynb](./automl_tabular_classification_beans.ipynb) (tabular classification model)
- [google-cloud-pipeline-components_automl_text.ipynb](.google-cloud-pipeline-components_automl_text.ipynb)
- (Experimental) [google_cloud_pipeline_components_model_train_upload_deploy.ipynb](./google_cloud_pipeline_components_model_train_upload_deploy.ipynb): includes an experimental component to run a custom training job directly by defining its worker specs
- (Experimental) [google_cloud_pipeline_components_model_upload_predict_evaluate.ipynb](./google_cloud_pipeline_components_model_upload_predict_evaluate.ipynb): includes an experimental evaluation component to generate evaluation metrics for a model given ground truth and predictions

**Note**: Currently, pipelines built using `kfp.v2`, such as these examples, will work only with Vertex Pipelines.
A 'compatibility mode', which will allow these pipelines to be run on OSS KFP as well, is coming soon.
