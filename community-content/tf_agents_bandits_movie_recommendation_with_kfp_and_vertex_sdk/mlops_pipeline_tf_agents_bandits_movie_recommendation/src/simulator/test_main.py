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

"""The unit testing module for the Simulator."""
import unittest
from unittest import mock

# In order to make mocking in setUp work.
from google.cloud import aiplatform  # pylint: disable=unused-import

import numpy as np
from src.simulator import main


# Paths and configurations
PROJECT_ID = "project-id"
REGION = "region"
ENDPOINT_ID = "endpoint-id"
RAW_DATA_PATH = "gs://bucket-name/dataset-dir/u.data"

# Hyperparameters
BATCH_SIZE = "8"

# MovieLens simulation environment parameters
RANK_K = "20"
NUM_ACTIONS = "20"

# Simulator variables
ENV_VARS = {
    "PROJECT_ID": PROJECT_ID,
    "REGION": REGION,
    "ENDPOINT_ID": ENDPOINT_ID,
    "RAW_DATA_PATH": RAW_DATA_PATH,
    "RANK_K": RANK_K,
    "BATCH_SIZE": BATCH_SIZE,
    "NUM_ACTIONS": NUM_ACTIONS,
}

OBSERVATION_ARRAY = np.zeros((int(BATCH_SIZE), int(RANK_K)))
OBSERVATION_LIST = [
    list(observation_batch) for observation_batch in OBSERVATION_ARRAY
]


def build_side_effect_function(env_vars):
  """Builds a side effect function for `os.getenv` with `env_vars` mapping.

  Args:
    env_vars: A dict mapping environment variable names to their values.

  Returns:
    A function that can serve as the side effect function for `os.getenv`.
  """
  def side_effect_os_getenv(name):
    return env_vars[name]
  return side_effect_os_getenv


class TestSimulator(unittest.TestCase):
  """Test class for the Simulator."""

  def setUp(self):
    super().setUp()

    self.mock_os_getenv = mock.patch("os.getenv").start()

    self.mock_env = mock.MagicMock()
    self.mock_env._observe.return_value = OBSERVATION_ARRAY

    self.mock_movielens = mock.patch(
        "tf_agents.bandits.environments.movielens_py_environment." +
        "MovieLensPyEnvironment").start()
    self.mock_movielens.return_value = self.mock_env

    self.mock_endpoint = mock.MagicMock()

    self.mock_aiplatform = mock.patch("google.cloud.aiplatform").start()
    self.mock_aiplatform.Endpoint.return_value = self.mock_endpoint

  def tearDown(self):
    super().tearDown()
    mock.patch.stopall()

  def test_given_valid_arguments_simulate_work(self):
    """Tests simulate() works given valid arguments."""
    self.mock_os_getenv.side_effect = build_side_effect_function(ENV_VARS)

    main.simulate(None, None)

    # Assert the MovieLens environment is built.
    self.mock_movielens.assert_called_once_with(
        RAW_DATA_PATH,
        int(RANK_K),
        int(BATCH_SIZE),
        num_movies=int(NUM_ACTIONS),
        csv_delimiter="\t")

    # Assert observations from the environment are generated.
    self.mock_env._observe.assert_called_once()

    # Assert prediction requests with observations are sent to the endpoint.
    self.mock_aiplatform.init.assert_called_once_with(
        project=PROJECT_ID, location=REGION)
    self.mock_aiplatform.Endpoint.assert_called_once_with(ENDPOINT_ID)
    self.mock_endpoint.predict.assert_called_once()

  def test_simulate_reformat_observation_array_to_a_list_with_same_data(self):
    """Tests simulate() reformats the observation array to list of the same data.
    """
    self.mock_os_getenv.side_effect = build_side_effect_function(ENV_VARS)

    main.simulate(None, None)

    self.mock_endpoint.predict.assert_called_once_with(
        instances=[
            {"observation": OBSERVATION_LIST},
        ],
    )

  def test_given_float_rank_k_simulate_raise_exception(self):
    """Tests given a float as `RANK_K` simulate() raises an exception."""
    env_vars = ENV_VARS.copy()
    env_vars["RANK_K"] = "20.0"
    self.mock_os_getenv.side_effect = build_side_effect_function(env_vars)

    with self.assertRaises(
        ValueError):  # Invalid literal for int() with base 10.
      main.simulate(None, None)

  def test_given_float_batch_size_simulate_raise_exception(self):
    """Tests given a float as `BATCH_SIZE` simulate() raises an exception."""
    env_vars = ENV_VARS.copy()
    env_vars["BATCH_SIZE"] = "8.0"
    self.mock_os_getenv.side_effect = build_side_effect_function(env_vars)

    with self.assertRaises(
        ValueError):  # Invalid literal for int() with base 10.
      main.simulate(None, None)

  def test_given_float_num_actions_simulate_raise_exception(self):
    """Tests given a float as `NUM_ACTIONS` simulate() raises an exception."""
    env_vars = ENV_VARS.copy()
    env_vars["NUM_ACTIONS"] = "20.0"
    self.mock_os_getenv.side_effect = build_side_effect_function(env_vars)

    with self.assertRaises(
        ValueError):  # Invalid literal for int() with base 10.
      main.simulate(None, None)


if __name__ == "__main__":
  unittest.main()
