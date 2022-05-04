import execute_changed_notebooks_helper

notebooks = [
    "notebooks/official/pipelines/automl_tabular_classification_beans.ipynb"
    for _ in range(0, 1)
]

execute_changed_notebooks_helper.process_and_execute_notebooks(
    notebooks=notebooks,
    container_uri="gcr.io/cloud-devrel-public-resources/python-samples-testing-docker:latest",
    staging_bucket="gs://ivanmkc-test2/staging",
    artifacts_bucket="gs://ivanmkc-test2/artifacts",
    variable_project_id="python-docs-samples-tests",
    variable_region="us-central1",
    private_pool_id="projects/1012616486416/locations/us-central1/workerPools/notebook-ci",
    should_parallelize=False,
)
