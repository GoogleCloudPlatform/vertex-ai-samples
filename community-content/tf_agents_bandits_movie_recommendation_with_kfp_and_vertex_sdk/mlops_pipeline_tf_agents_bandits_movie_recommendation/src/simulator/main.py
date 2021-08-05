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

"""The Simulator component for sending recurrent prediction requests."""
import logging
import os
from typing import Any, Dict

import dataclasses
from google import cloud  # For patch of google.cloud.aiplatform to work.
from google.cloud import aiplatform  # For using the module.  # pylint: disable=unused-import
import tensorflow as tf  # For tf_agents to work.  # pylint: disable=unused-import
from tf_agents.bandits.environments import movielens_py_environment


@dataclasses.dataclass
class EnvVars:
  """A class containing environment variables and their values.

  Attributes:
    project_id: A string of the GCP project ID.
    region: A string of the GCP service region.
    endpoint_id: A string of the Vertex AI endpoint ID.
    raw_data_path: A string of the path to MovieLens 100K's "u.data" file.
    rank_k: An integer of the rank for matrix factorization in the MovieLens
      environment; also the observation dimension.
    batch_size: A integer of the batch size of environment generated quantities.
    num_actions: A integer of the number of actions (movie items) to choose
      from.
  """
  project_id: str
  region: str
  endpoint_id: str
  raw_data_path: str
  rank_k: int
  batch_size: int
  num_actions: int


def get_env_vars() -> EnvVars:
  """Gets a set of environment variables necessary for `simulate`.

  Returns:
    A `EnvVars` of environment variables for configuring `simulate`.
  """
  return EnvVars(
      project_id=os.getenv("PROJECT_ID"),
      region=os.getenv("REGION"),
      endpoint_id=os.getenv("ENDPOINT_ID"),
      raw_data_path=os.getenv("RAW_DATA_PATH"),
      rank_k=int(os.getenv("RANK_K")),
      batch_size=int(os.getenv("BATCH_SIZE")),
      num_actions=int(os.getenv("NUM_ACTIONS")))


def simulate(event: Dict[str, Any], context) -> None:  # pylint: disable=unused-argument
  """Gets observations and sends prediction requests to endpoints.

  Queries the MovieLens simulation environment for observations and sends
  prediction requests with the observations to the Vertex endpoint.

  Serves as the Simulator and the entrypoint of Cloud Functions.

  Note: In production, this function can be modified to hold the logic of
  gathering real-world input features as observations, getting prediction
  results from the endpoint and communicating those results to real-world
  users.

  Args:
    event: Triggering event of this function.
    context: Trigerring context of this function.
      This is of type `functions_v1.context.Context` but not specified since
      it is not importable for a local environment that wants to run unit
      tests.
  """
  env_vars = get_env_vars()

  # Create MovieLens simulation environment.
  env = movielens_py_environment.MovieLensPyEnvironment(
      env_vars.raw_data_path, env_vars.rank_k, env_vars.batch_size,
      num_movies=env_vars.num_actions, csv_delimiter="\t")

  # Get environment observation.
  observation_array = env._observe()  # pylint: disable=protected-access
  # Convert to nested list to be sent to the endpoint for prediction.
  observation = [
      list(observation_batch) for observation_batch in observation_array
  ]

  cloud.aiplatform.init(
      project=env_vars.project_id, location=env_vars.region)
  endpoint = cloud.aiplatform.Endpoint(env_vars.endpoint_id)

  # Send prediction request to endpoint and get prediction result.
  predictions = endpoint.predict(
      instances=[
          {"observation": observation},
      ]
  )

  logging.info("prediction result: %s", predictions[0])
  logging.info("prediction model ID: %s", predictions[1])
