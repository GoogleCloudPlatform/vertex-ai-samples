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

import json
import sys
import nbformat
import os
import errno
from NotebookProcessors import RemoveNoExecuteCells, UpdateVariablesPreprocessor
from typing import Dict, Tuple
import papermill as pm
import shutil
import virtualenv
import uuid
from jupyter_client.kernelspecapp import KernelSpecManager

# This script is used to execute a notebook and write out the output notebook.
# The replaces calling the nbconvert via command-line, which doesn't write the output notebook correctly when there are errors during execution.

STAGING_FOLDER = "staging"
ENVIRONMENTS_PATH = "environments"
KERNELS_SPECS_PATH = "kernel_specs"


def create_and_install_kernel() -> Tuple[str, str]:
    # Create environment
    kernel_name = str(uuid.uuid4())
    env_name = f"{ENVIRONMENTS_PATH}/{kernel_name}"
    # venv.create(env_name, system_site_packages=True, with_pip=True)
    virtualenv.cli_run([env_name, "--system-site-packages"])

    # Create kernel spec
    kernel_spec = {
        "argv": [
            f"{env_name}/bin/python",
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ],
        "display_name": "Python 3",
        "language": "python",
    }
    kernel_spec_folder = os.path.join(KERNELS_SPECS_PATH, kernel_name)
    kernel_spec_file = os.path.join(kernel_spec_folder, "kernel.json")

    # Create kernel spec folder
    if not os.path.exists(os.path.dirname(kernel_spec_file)):
        try:
            os.makedirs(os.path.dirname(kernel_spec_file))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(kernel_spec_file, mode="w", encoding="utf-8") as f:
        json.dump(kernel_spec, f)

    # Install kernel
    kernel_spec_manager = KernelSpecManager()
    kernel_spec_manager.install_kernel_spec(
        source_dir=kernel_spec_folder, kernel_name=kernel_name
    )

    return kernel_name, env_name


def execute_notebook(
    notebook_file_path: str,
    output_file_folder: str,
    replacement_map: Dict[str, str],
    should_log_output: bool,
    should_use_new_kernel: bool,
):
    # Create staging directory if it doesn't exist
    staging_file_path = f"{STAGING_FOLDER}/{notebook_file_path}"
    if not os.path.exists(os.path.dirname(staging_file_path)):
        try:
            os.makedirs(os.path.dirname(staging_file_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    file_name = os.path.basename(os.path.normpath(notebook_file_path))

    # Create environments folder
    if not os.path.exists(ENVIRONMENTS_PATH):
        try:
            os.makedirs(ENVIRONMENTS_PATH)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    # Create and install kernel
    kernel_name = next(
        iter(KernelSpecManager().find_kernel_specs().keys()), None
    )  # Find first existing kernel and use as default
    env_name = None
    if should_use_new_kernel:
        kernel_name, env_name = create_and_install_kernel()

    # Read notebook
    with open(notebook_file_path) as f:
        nb = nbformat.read(f, as_version=4)

    has_error = False

    # Execute notebook
    try:
        # Create preprocessors
        remove_no_execute_cells_preprocessor = RemoveNoExecuteCells()
        update_variables_preprocessor = UpdateVariablesPreprocessor(
            replacement_map=replacement_map
        )

        # Use no-execute preprocessor
        (
            nb,
            resources,
        ) = remove_no_execute_cells_preprocessor.preprocess(nb)

        (nb, resources) = update_variables_preprocessor.preprocess(nb, resources)

        # print(f"Staging modified notebook to: {staging_file_path}")
        with open(staging_file_path, mode="w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        # Execute notebook
        pm.execute_notebook(
            input_path=staging_file_path,
            output_path=staging_file_path,
            kernel_name=kernel_name,
            progress_bar=should_log_output,
            request_save_on_cell_execute=should_log_output,
            log_output=should_log_output,
            stdout_file=sys.stdout if should_log_output else None,
            stderr_file=sys.stderr if should_log_output else None,
        )
    except Exception:
        # print(f"Error executing the notebook: {notebook_file_path}.\n\n")
        has_error = True

        raise

    finally:
        # Clear env
        if env_name is not None:
            shutil.rmtree(path=env_name)

        # Copy execute notebook
        output_file_path = os.path.join(
            output_file_folder, "failure" if has_error else "success", file_name
        )

        # Create directories if they don't exist
        if not os.path.exists(os.path.dirname(output_file_path)):
            try:
                os.makedirs(os.path.dirname(output_file_path))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        # print(f"Writing output to: {output_file_path}")
        shutil.move(staging_file_path, output_file_path)
