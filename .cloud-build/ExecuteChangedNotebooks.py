#!/bin/bash
# Copyright 2019 Google LLC
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
import argparse
import pathlib
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import ExecuteNotebook


def run_changed_notebooks(
    test_paths_file: str,
    output_folder: str,
    variable_project_id: str,
    variable_region: str,
    base_branch: Optional[str],
):
    """
    Run the notebooks that exist under the folders defined in the test_paths_file.
    It only runs notebooks that have differences from the Git base_branch.

    The executed notebooks are saved in the output_folder.

    Variables are also injected into the notebooks such as the variable_project_id and variable_region.

    Args:
        test_paths_file (str):
            Required. The new-line delimited file to folders and files that need checking.
            Folders are checked recursively.
        base_branch (str):
            Optional. If provided, only the files that have changed from the base_branch will be checked.
            If not provided, all files will be checked.
        output_folder (str):
            Required. The folder to write executed notebooks to.
        variable_project_id (str):
            Required. The value for PROJECT_ID to inject into notebooks.
        variable_region (str):
            Required. The value for REGION to inject into notebooks.
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
    if base_branch:
        print(f"Looking for notebooks that changed from branch: {base_branch}")
        notebooks = subprocess.check_output(
            ["git", "diff", "--name-only", f"origin/{base_branch}..."] + test_paths
        )
    else:
        print(f"Looking for all notebooks.")
        notebooks = subprocess.check_output(["git", "ls-files"] + test_paths)

    notebooks = notebooks.decode("utf-8").split("\n")
    notebooks = [notebook for notebook in notebooks if notebook.endswith(".ipynb")]
    notebooks = [notebook for notebook in notebooks if len(notebook) > 0]
    notebooks = [notebook for notebook in notebooks if Path(notebook).exists()]

    # Create paths
    artifacts_path = Path(output_folder)
    artifacts_path.mkdir(parents=True, exist_ok=True)
    artifacts_path.joinpath("success").mkdir(parents=True, exist_ok=True)
    artifacts_path.joinpath("failure").mkdir(parents=True, exist_ok=True)

    passed_notebooks: List[str] = []
    failed_notebooks: List[str] = []

    if len(notebooks) > 0:
        print(f"Found {len(notebooks)} modified notebooks: {notebooks}")

        for notebook in notebooks:
            print(f"Running notebook: {notebook}")

            # TODO: Handle cases where multiple notebooks have the same name
            try:
                ExecuteNotebook.execute_notebook(
                    notebook_file_path=notebook,
                    output_file_folder=artifacts_path,
                    replacement_map={
                        "PROJECT_ID": variable_project_id,
                        "REGION": variable_region,
                    },
                )
                print(f"Notebook finished successfully.")
                passed_notebooks.append(notebook)
            except Exception as error:
                print(f"Notebook finished with failure: {error}")
                failed_notebooks.append(notebook)
    else:
        print("No notebooks modified in this pull request.")

    if len(failed_notebooks) > 0:
        print(f"{len(failed_notebooks)} notebooks failed:")
        print(failed_notebooks)
        print(f"{len(passed_notebooks)} notebooks passed:")
        print(passed_notebooks)
    elif len(passed_notebooks) > 0:
        print("All notebooks executed successfully:")
        print(passed_notebooks)


parser = argparse.ArgumentParser(description="Run changed notebooks.")
parser.add_argument(
    "--test_paths_file",
    type=pathlib.Path,
    help="The path to the file that has newline-limited folders of notebooks that should be tested.",
    required=True,
)
parser.add_argument(
    "--base_branch",
    help="The base git branch to diff against to find changed files.",
    required=False,
)
parser.add_argument(
    "--output_folder",
    type=pathlib.Path,
    help="The path to the folder to store executed notebooks.",
    required=True,
)
parser.add_argument(
    "--variable_project_id",
    type=str,
    help="The GCP project id. This is used to inject a variable value into the notebook before running.",
    required=True,
)
parser.add_argument(
    "--variable_region",
    type=str,
    help="The GCP region. This is used to inject a variable value into the notebook before running.",
    required=True,
)

args = parser.parse_args()
run_changed_notebooks(
    test_paths_file=args.test_paths_file,
    base_branch=args.base_branch,
    output_folder=args.output_folder,
    variable_project_id=args.variable_project_id,
    variable_region=args.variable_region,
)
