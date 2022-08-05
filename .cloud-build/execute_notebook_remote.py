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

"""Methods to run a notebook on Google Cloud Build"""

from re import sub
from typing import Optional

import google.auth
import yaml
from google.api_core import client_options, operation
from google.cloud.aiplatform import utils
from google.cloud.devtools import cloudbuild_v1
from google.cloud.devtools.cloudbuild_v1.types import Source, StorageSource
from google.protobuf import duration_pb2
from yaml.loader import FullLoader

CLOUD_BUILD_FILEPATH = ".cloud-build/notebook-execution-test-cloudbuild-single.yaml"
SERVICE_BASE_PATH = "cloudbuild.googleapis.com"


def execute_notebook_remote(
    code_archive_uri: str,
    notebook_uri: str,
    notebook_output_uri: str,
    container_uri: str,
    private_pool_id: Optional[str],
    private_pool_region: Optional[str],
    tag: Optional[str],
    timeout_in_seconds: Optional[int] = None,
    python_version: Optional[str] = None
) -> operation.Operation:
    """Create and execute a single notebook on Google Cloud Build"""
    # Load build steps from YAML

    cloudbuild_config = yaml.load(open(CLOUD_BUILD_FILEPATH), Loader=FullLoader)

    substitutions = {
        "_PYTHON_IMAGE": container_uri,
        "_NOTEBOOK_GCS_URI": notebook_uri,
        "_NOTEBOOK_OUTPUT_GCS_URI": notebook_output_uri,
    }

    if python_version is not None:
      substitutions["_PYTHON_VERION"] = python_version

    build = cloudbuild_v1.Build()

    options: Optional[client_options.ClientOptions] = None
    if private_pool_id and private_pool_region:
        # substitutions["_PRIVATE_POOL_NAME"] = private_pool_id
        build.options = cloudbuild_config.get("options")
        build.options.pool = {"name": private_pool_id}

        # Switch to the regional endpoint of the pool
        options = client_options.ClientOptions(
            api_endpoint=f"{private_pool_region}-{SERVICE_BASE_PATH}"
        )

    # Authorize the client with Google defaults
    credentials, project_id = google.auth.default()

    client = cloudbuild_v1.services.cloud_build.CloudBuildClient(client_options=options)

    (
        source_archived_file_gcs_bucket,
        source_archived_file_gcs_object,
    ) = utils.extract_bucket_and_prefix_from_gcs_path(code_archive_uri)

    build.source = Source(
        storage_source=StorageSource(
            bucket=source_archived_file_gcs_bucket,
            object_=source_archived_file_gcs_object,
        )
    )

    build.steps = cloudbuild_config["steps"]
    build.substitutions = substitutions
    build.timeout = duration_pb2.Duration(seconds=timeout_in_seconds)
    build.queue_ttl = duration_pb2.Duration(seconds=timeout_in_seconds)

    if tag:
        build.tags = [tag]

    operation = client.create_build(project_id=project_id, build=build)
    # Print the in-progress operation
    # print("IN PROGRESS:")
    # print(operation.metadata)

    # Print the completed status
    # print("RESULT:", result.status)
    return operation
