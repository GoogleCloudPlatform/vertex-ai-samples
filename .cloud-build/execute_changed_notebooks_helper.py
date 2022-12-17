#!/usr/bin/env python
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import concurrent
import dataclasses
import datetime
import functools
import json
import git
import operator
import os
import pathlib
import re
import subprocess
import utils
from typing import List, Optional
from utils import util

import execute_notebook_helper
import execute_notebook_remote
import nbformat
from google.cloud.devtools.cloudbuild_v1.types import BuildOperationMetadata
from ratemate import RateLimit
from tabulate import tabulate
from utils import NotebookProcessors, util

# A buffer so that workers finish before the orchestrating job
WORKER_TIMEOUT_BUFFER_IN_SECONDS: int = 60 * 60
PYTHON_VERSION = "3.9"  # Set default python version


def format_timedelta(delta: datetime.timedelta) -> str:
    """Formats a timedelta duration to [N days] %H:%M:%S format"""
    seconds = int(delta.total_seconds())

    secs_in_a_day = 86400
    secs_in_a_hour = 3600
    secs_in_a_min = 60

    days, seconds = divmod(seconds, secs_in_a_day)
    hours, seconds = divmod(seconds, secs_in_a_hour)
    minutes, seconds = divmod(seconds, secs_in_a_min)

    time_fmt = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    if days > 0:
        suffix = "s" if days > 1 else ""
        return f"{days} day{suffix} {time_fmt}"

    return time_fmt


@dataclasses.dataclass
class NotebookExecutionResult:
    name: str
    duration: datetime.timedelta
    is_pass: bool
    log_url: str
    output_uri: str
    build_id: str
    logs_bucket: str
    error_message: Optional[str]

    @property
    def output_uri_web(self) -> Optional[str]:
        if self.output_uri.startswith("gs://"):
            return f"https://storage.googleapis.com/{self.output_uri[5:]}"
        else:
            return None


def _process_notebook(
    notebook_path: str,
    variable_project_id: str,
    variable_region: str,
    variable_service_account: str,
    variable_vpc_network: Optional[str],
):
    # Read notebook
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    # Create preprocessors
    remove_no_execute_cells_preprocessor = NotebookProcessors.RemoveNoExecuteCells()
    update_variables_preprocessor = NotebookProcessors.UpdateVariablesPreprocessor(
        replacement_map={
            "PROJECT_ID": variable_project_id,
            "REGION": variable_region,
            "SERVICE_ACCOUNT": variable_service_account,
            "VPC_NETWORK": variable_vpc_network,
        },
    )
    unique_strings_preprocessor = NotebookProcessors.UniqueStringsPreprocessor()

    # Use no-execute preprocessor
    (
        nb,
        resources,
    ) = remove_no_execute_cells_preprocessor.preprocess(nb)

    (nb, resources) = update_variables_preprocessor.preprocess(nb, resources)
    (nb, resources) = unique_strings_preprocessor.preprocess(nb, resources)

    with open(notebook_path, mode="w", encoding="utf-8") as new_file:
        nbformat.write(nb, new_file)


def _get_notebook_python_version(notebook_path: str) -> str:
    """
    Get the python version for running the notebook if it is specified in
    the notebook.
    """
    python_version = PYTHON_VERSION

    # Load the notebook
    file = open(notebook_path)
    src = file.read()
    nb_json = json.loads(src)

    # Iterate over the cells in the ipynb
    for cell in nb_json["cells"]:
        if cell["cell_type"] == "markdown":
            markdown = str.join("", cell["source"])

            # Look for the python version specification pattern
            re_match = re.search(
                "python version = (\d\.\d)", markdown, flags=re.IGNORECASE
            )
            if re_match:
                # get the version number
                python_version = re_match.group(1)
                break

    return python_version


def _create_tag(filepath: str) -> str:
    tag = os.path.basename(os.path.normpath(filepath))
    tag = re.sub("[^0-9a-zA-Z_.-]+", "-", tag)

    if tag.startswith(".") or tag.startswith("-"):
        tag = tag[1:]

    return tag


rate_limit = RateLimit(max_count=50, per=60, greedy=True)


def process_and_execute_notebook(
    container_uri: str,
    staging_bucket: str,
    artifacts_bucket: str,
    variable_project_id: str,
    variable_region: str,
    variable_service_account: str,
    variable_vpc_network: Optional[str],
    private_pool_id: Optional[str],
    deadline: datetime.datetime,
    notebook: str,
    should_get_tail_logs: bool = False,
) -> NotebookExecutionResult:
    rate_limit.wait()  # wait before creating the task

    print(f"Running notebook: {notebook}")

    # Handle empty strings
    if not variable_vpc_network:
        variable_vpc_network = None

    if not private_pool_id:
        private_pool_id = None

    # Create paths
    notebook_output_uri = "/".join([artifacts_bucket, pathlib.Path(notebook).name])

    # Create tag from notebook
    tag = _create_tag(filepath=notebook)

    result = NotebookExecutionResult(
        name=tag,
        duration=datetime.timedelta(seconds=0),
        is_pass=False,
        output_uri=notebook_output_uri,
        log_url="",
        build_id="",
        logs_bucket="",
        error_message=None,
    )

    # TODO: Handle cases where multiple notebooks have the same name
    time_start = datetime.datetime.now()
    operation = None
    try:
        # Get the python version for running the notebook if specified
        notebook_exec_python_version = _get_notebook_python_version(
            notebook_path=notebook
        )
        print(f"Running notebook with python {notebook_exec_python_version}")

        # Pre-process notebook by substituting variable names
        _process_notebook(
            notebook_path=notebook,
            variable_project_id=variable_project_id,
            variable_region=variable_region,
            variable_service_account=variable_service_account,
            variable_vpc_network=variable_vpc_network,
        )

        # Upload the pre-processed code to a GCS bucket
        code_archive_uri = util.archive_code_and_upload(staging_bucket=staging_bucket)

        # Calculate timeout in seconds
        timeout_in_seconds = max(
            int((deadline - datetime.datetime.now()).total_seconds()), 1
        )

        operation = execute_notebook_remote.execute_notebook_remote(
            code_archive_uri=code_archive_uri,
            notebook_uri=notebook,
            notebook_output_uri=notebook_output_uri,
            container_uri=container_uri,
            tag=tag,
            private_pool_id=private_pool_id,
            private_pool_region=variable_region,
            timeout_in_seconds=timeout_in_seconds,
            python_version=notebook_exec_python_version,
        )

        operation_metadata = BuildOperationMetadata(mapping=operation.metadata)
        result.build_id = operation_metadata.build.id
        result.log_url = operation_metadata.build.log_url
        result.logs_bucket = operation_metadata.build.logs_bucket

        # Block and wait for the result
        operation_result = operation.result(timeout=86400)

        result.duration = datetime.datetime.now() - time_start
        result.is_pass = True
        print(f"{notebook} PASSED in {format_timedelta(result.duration)}.")
    except Exception as error:
        result.error_message = str(error)

        if operation and should_get_tail_logs:
            # Extract the logs
            logs_bucket = operation_metadata.build.logs_bucket

            # Download tail end of logs file
            log_file_uri = f"{logs_bucket}/log-{result.build_id}.txt"

            # Use gcloud to get tail
            try:
                result.error_message = subprocess.check_output(
                    ["gsutil", "cat", "-r", "-1000", log_file_uri], encoding="UTF-8"
                )
            except Exception as error:
                result.error_message = str(error)

        result.duration = datetime.datetime.now() - time_start
        result.is_pass = False

        print(
            f"{notebook} FAILED in {format_timedelta(result.duration)}: {result.error_message}"
        )

    return result


def get_changed_notebooks(
    test_paths_file: str,
    base_branch: Optional[str] = None,
) -> List[str]:
    """
    Get the notebooks that exist under the folders defined in the test_paths_file.
    It only returns notebooks that have differences from the Git base_branch.
    """

    test_paths = []
    with open(test_paths_file) as file:
        lines = [line.strip() for line in file.readlines()]
        lines = [line for line in lines if len(line) > 0]
        test_paths = [line for line in lines]

    if len(test_paths) == 0:
        raise RuntimeError("No test folders found.")

    print(f"Checking folders: {test_paths}")

    # Find notebooks
    notebooks = []

    # Instantiate GitPython objects
    repo = git.Repo(os.getcwd())
    index = repo.index

    if base_branch:
        # Get the point at which this branch branches off from main
        branching_commits = repo.merge_base("HEAD", f"origin/{base_branch}")

        if len(branching_commits) > 0:
            branching_commit = branching_commits[0]
            print(f"Looking for notebooks that changed from branch: {branching_commit}")

            notebooks = [
                diff.b_path
                for diff in index.diff(branching_commit, paths=test_paths)
                if diff.b_path is not None
            ]
        else:
            notebooks = []
    else:
        print(f"Looking for all notebooks.")
        notebooks_str = subprocess.check_output(["git", "ls-files"] + test_paths)
        notebooks = notebooks_str.decode("utf-8").split("\n")

    notebooks = [notebook for notebook in notebooks if notebook.endswith(".ipynb")]
    notebooks = [notebook for notebook in notebooks if len(notebook) > 0]
    notebooks = [notebook for notebook in notebooks if pathlib.Path(notebook).exists()]

    if len(notebooks) > 0:
        print(f"Found {len(notebooks)} notebooks:")
        for notebook in notebooks:
            print(f"\t{notebook}")

    return notebooks


def process_and_execute_notebooks(
    notebooks: List[str],
    container_uri: str,
    staging_bucket: str,
    artifacts_bucket: str,
    should_parallelize: bool,
    timeout: int,
    variable_project_id: str,
    variable_region: str,
    variable_service_account: str,
    variable_vpc_network: Optional[str] = None,
    private_pool_id: Optional[str] = None,
):
    """
    Run the notebooks that exist under the folders defined in the test_paths_file.
    It only runs notebooks that have differences from the Git base_branch.

    The executed notebooks are saved in the artifacts_bucket.

    Variables are also injected into the notebooks such as the variable_project_id and variable_region.

    Args:
        test_paths_file (str):
            Required. The new-line delimited file to folders and files that need checking.
            Folders are checked recursively.
        base_branch (str):
            Optional. If provided, only the files that have changed from the base_branch will be checked.
            If not provided, all files will be checked.
        staging_bucket (str):
            Required. The GCS staging bucket to write source code to.
        artifacts_bucket (str):
            Required. The GCS staging bucket to write executed notebooks to.
        variable_project_id (str):
            Required. The value for PROJECT_ID to inject into notebooks.
        variable_region (str):
            Required. The value for REGION to inject into notebooks.
        should_parallelize (bool):
            Required. Should run notebooks in parallel using a thread pool as opposed to in sequence.
        timeout (str):
            Required. Timeout string according to https://cloud.google.com/build/docs/build-config-file-schema#timeout.
    """

    # Calculate deadline
    deadline = datetime.datetime.now() + datetime.timedelta(
        seconds=max(timeout - WORKER_TIMEOUT_BUFFER_IN_SECONDS, 0)
    )

    if len(notebooks) >= 1:
        notebook_execution_results: List[NotebookExecutionResult] = []

        print(f"Found {len(notebooks)} modified notebooks: {notebooks}")

        if should_parallelize and len(notebooks) > 1:
            print(
                "Running notebooks in parallel, so no logs will be displayed. Please wait..."
            )
            with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
                print(f"Max workers: {executor._max_workers}")

                notebook_execution_results = list(
                    executor.map(
                        functools.partial(
                            process_and_execute_notebook,
                            container_uri,
                            staging_bucket,
                            artifacts_bucket,
                            variable_project_id,
                            variable_region,
                            variable_service_account,
                            variable_vpc_network,
                            private_pool_id,
                            deadline,
                        ),
                        notebooks,
                    )
                )
        else:
            notebook_execution_results = [
                process_and_execute_notebook(
                    container_uri=container_uri,
                    staging_bucket=staging_bucket,
                    artifacts_bucket=artifacts_bucket,
                    variable_project_id=variable_project_id,
                    variable_region=variable_region,
                    variable_service_account=variable_service_account,
                    variable_vpc_network=variable_vpc_network,
                    private_pool_id=private_pool_id,
                    deadline=deadline,
                    notebook=notebook,
                )
                for notebook in notebooks
            ]

        print("\n=== RESULTS ===\n")

        results_sorted = sorted(
            notebook_execution_results,
            key=lambda result: result.is_pass,
            reverse=True,
        )

        # Print results
        print(
            tabulate(
                [
                    [
                        result.name,
                        "PASSED" if result.is_pass else "FAILED",
                        format_timedelta(result.duration),
                        result.log_url,
                        result.output_uri,
                        result.output_uri_web,
                        result.logs_bucket,
                    ]
                    for result in results_sorted
                ],
                headers=[
                    "build_tag",
                    "status",
                    "duration",
                    "log_url",
                    "output_uri",
                    "output_uri_web",
                    "logs_bucket",
                ],
            )
        )

        if len(notebooks) == 1:
            print("=" * 100)
            print("The notebook execution build log:\n")
            print("=" * 100)

            build_id = results_sorted[0].build_id
            logs_bucket_name = (results_sorted[0].logs_bucket).removeprefix("gs://")
            log_file_name = f"log-{build_id}.txt"

            log_contents = util.download_blob_into_memory(
                bucket_name=logs_bucket_name,
                blob_name=log_file_name,
                download_as_text=True,
            )

            # Remove extra steps from the log
            match = re.search("starting Step #4", log_contents, flags=re.IGNORECASE)

            if match is not None:
                match_index = match.span()[0]
                print(log_contents[match_index:])
            else:
                print(log_contents)

        print("\n=== END RESULTS===\n")

        total_notebook_duration = functools.reduce(
            operator.add,
            [datetime.timedelta(seconds=0)]
            + [result.duration for result in results_sorted],
        )

        print(
            f"Cumulative notebook duration: {format_timedelta(total_notebook_duration)}"
        )

        # Raise error if any notebooks failed
        if not all([result.is_pass for result in results_sorted]):
            raise RuntimeError("Notebook failures detected. See logs for details")
    else:
        print("No notebooks modified in this pull request.")
