from google.cloud import aiplatform

from google.cloud.aiplatform import utils

from typing import List, Optional, Sequence
import os

CONTAINER_URI = (
    "gcr.io/cloud-devrel-public-resources/python-samples-testing-docker:latest"
)


# def _package_and_upload_module(
#     self,
#     project: str,
#     destination_uri: str,
#     training_script_path: str,
#     requirements: str,
# ) -> str:
#     # Create packager
#     python_packager = utils.source_utils._TrainingScriptPythonPackager(
#         script_path=training_script_path, requirements=requirements
#     )

#     # Package and upload to GCS
#     package_gcs_uri = python_packager.package_and_copy_to_gcs(
#         gcs_staging_dir=destination_uri,
#         project=project,
#     )

#     print(f"Custom Training Python Package is uploaded to: {package_gcs_uri}")

#     return package_gcs_uri


def run_notebook_remote(
    script_path: str,
    container_uri: str,
    notebook_uri: str,
    requirements: Optional[Sequence[str]] = None,
):
    notebook_name = "notebook_execution"
    # job = aiplatform.CustomPythonPackageTrainingJob(
    #     display_name=notebook_name,
    #     python_package_gcs_uri=package_gcs_uri,
    #     python_module_name=python_module_name,
    #     container_uri=container_uri,
    # )

    job = aiplatform.CustomTrainingJob(
        display_name=notebook_name,
        script_path=script_path,
        container_uri=container_uri,
        requirements=requirements,
    )

    job.run(
        args=["--notebook_uri", notebook_uri],
        replica_count=1,
        sync=True,
    )


PYTHON_MODULE_NAME = f"{utils.source_utils._TrainingScriptPythonPackager._ROOT_MODULE}.{utils.source_utils._TrainingScriptPythonPackager._TASK_MODULE_NAME}"

project = "python-docs-samples-tests"
staging_bucket = "gs://ivanmkc-test2/notebooks"
destination_gcs_folder = staging_bucket + "/notebooks"

notebook_path = "notebooks/official/custom/custom-tabular-bq-managed-dataset.ipynb"

# package_gcs_uri = _package_and_upload_module(
#     project=project,
#     destination_uri=destination_uri,
#     training_script_path=".cloud-build/ExecuteChangedNotebooks.py",
#     requirements="",
# )

# Upload notebook
notebook_uri = utils._timestamped_copy_to_gcs(
    local_file_path=notebook_path, gcs_dir=destination_gcs_folder
)

aiplatform.init(project=project, staging_bucket=staging_bucket)

# Read requirements.txt
lines = []
with open(".cloud-build/requirements.txt") as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]

run_notebook_remote(
    script_path=".cloud-build/ExecuteChangedNotebooks.py",
    container_uri=CONTAINER_URI,
    notebook_uri=notebook_uri,
    requirements=lines,
)
