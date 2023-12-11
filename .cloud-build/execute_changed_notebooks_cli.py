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

"""A CLI to process changed notebooks and execute them on Google Cloud Build"""

import argparse
import pathlib
import os
import csv

import execute_changed_notebooks_helper


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser(description="Run changed notebooks.")
parser.add_argument(
    "--test_paths_file",
    type=pathlib.Path,
    help="The path to the file that has newline-delimited folders of notebooks that should be tested.",
    required=True,
)
parser.add_argument(
    "--test_percent",
    type=int,
    help="The percent of notebooks to be tested (between 1 and 100).",
    required=False,
    default=100,
)
parser.add_argument(
    "--build_id",
    type=str,
    help="The build id (which may be a Cloud Build job specific or user explicit.",
    required=True
)
parser.add_argument(
    "--base_branch",
    help="The base git branch to diff against to find changed files.",
    required=False,
)
parser.add_argument(
    "--container_uri",
    type=str,
    help="The container uri to run each notebook in.",
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
parser.add_argument(
    "--variable_service_account",
    type=str,
    help="A service account. This is used to inject a variable value into the notebook before running. This is not the account that will run the notebook.",
    required=True,
)
parser.add_argument(
    "--variable_vpc_network",
    type=str,
    help="The full VPC network name. See https://cloud.google.com/compute/docs/networks-and-firewalls#networks. Format is projects/{project}/global/networks/{network}, where {project} is a project number, as in '12345', and {network} is network name. See <https://cloud.google.com/compute/docs/reference/rest/v1/networks/insert> for details. This is used to inject a variable value into the notebook before running.",
    required=False,
)
parser.add_argument(
    "--staging_bucket",
    type=str,
    help="The GCP directory for staging temporary files.",
    required=True,
)
parser.add_argument(
    "--artifacts_bucket",
    type=str,
    help="The GCP directory for storing executed notebooks.",
    required=True,
)
parser.add_argument(
    "--timeout",
    type=int,
    help="Timeout in seconds",
    default=86400,
    required=False,
)
parser.add_argument(
    "--private_pool_id",
    type=str,
    help="The private pool id.",
    required=False,
)
parser.add_argument(
    "--should_parallelize",
    type=str2bool,
    nargs="?",
    const=True,
    default=True,
    help="Should run notebooks in parallel.",
)
parser.add_argument(
    "--concurrent_notebooks",
    type=int,
    help="Maximum number of parallel notebook executions per minute",
    default=10,
    required=False,
)
parser.add_argument(
    "--run_first_file",
    type=pathlib.Path,
    help="The path to the file that has newline-delimited of notebooks to run in the first batch",
    default=None,
    required=False,
)
parser.add_argument(
    "--aiplatform_whl",
    type=str,
    help="The GCS path to a whl version google-cloud-aiplatform",
    default=None,
    required=False,
)
parser.add_argument(
    "--dry_run",
    type=str2bool,
    default=False,
    help="Dry run for testing - no execution",
)

args = parser.parse_args()

changed_notebooks = execute_changed_notebooks_helper.get_changed_notebooks(
    test_paths_file=args.test_paths_file,
    base_branch=args.base_branch,
)


results_bucket = f"{args.artifacts_bucket}"
# artifacts_bucket may get set by trigger to a full gs:// folder path
if results_bucket.startswith("gs://"):
    results_bucket = results_bucket[5:]
results_bucket = results_bucket.split('/')[0]
results_file = f"build_results/{args.build_id}.json"

if args.test_percent == 100:
    notebooks = changed_notebooks
    accumulative_results = {}
else:
    accumulative_results = execute_changed_notebooks_helper.load_results(results_bucket, results_file)

    notebooks = [changed_notebook for changed_notebook in changed_notebooks if execute_changed_notebooks_helper.select_notebook(changed_notebook, accumulative_results, args.test_percent)]
    # cap the number of notebooks to the specified percentage
    max_notebooks = int((len(changed_notebooks) * (args.test_percent/100)))
    if (len(notebooks) > max_notebooks):
        notebooks = notebooks[:max_notebooks]

run_first = []
if args.run_first_file:
    if not os.path.isfile(args.run_first_file):
        print("Error: file does not exist", args.run_first_file)
    else:
        with open(args.run_first_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                notebook = row[0]
                run_first.append(notebook)

    for notebook in run_first:
        if notebook in notebooks:
            # remove from existing list
            notebooks.remove(notebook)
            # add back to the front of the list
            notebooks.insert(0, notebook)
            print(f"Run first: {notebook}")

if args.dry_run:
    print("Dry run ...\n")
    for notebook in notebooks:
        print(f"Would execute: {notebook}")
else:
    execute_changed_notebooks_helper.process_and_execute_notebooks(
        notebooks=notebooks,
        container_uri=args.container_uri,
        staging_bucket=args.staging_bucket,
        artifacts_bucket=args.artifacts_bucket,
        results_file=results_file,
        should_parallelize=args.should_parallelize,
        timeout=args.timeout,
        variable_project_id=args.variable_project_id,
        variable_region=args.variable_region,
        variable_service_account=args.variable_service_account,
        variable_vpc_network=args.variable_vpc_network,
        private_pool_id=args.private_pool_id,
        concurrent_notebooks=args.concurrent_notebooks,
        aiplatform_whl=args.aiplatform_whl
)
