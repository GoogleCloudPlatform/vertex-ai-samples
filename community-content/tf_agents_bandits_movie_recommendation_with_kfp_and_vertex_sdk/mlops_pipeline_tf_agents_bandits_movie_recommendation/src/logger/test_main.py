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

"""The unit testing module for the Logger."""
import base64
import json
import os
import tempfile
import unittest
from unittest import mock

from google.cloud import bigquery

import numpy as np
from src.logger import main


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

# BigQuery parameters
BIGQUERY_TMP_FILE = "tmp.json"
BIGQUERY_DATASET_ID = f"{PROJECT_ID}.movielens_dataset"
BIGQUERY_LOCATION = "us"
BIGQUERY_TABLE_ID = f"{BIGQUERY_DATASET_ID}.training_dataset"
DATASET_FILE = os.path.join(tempfile.gettempdir(), BIGQUERY_TMP_FILE)

# Logger environment variables
ENV_VARS = {
    "PROJECT_ID": PROJECT_ID,
    "RAW_DATA_PATH": RAW_DATA_PATH,
    "BATCH_SIZE": BATCH_SIZE,
    "RANK_K": RANK_K,
    "NUM_ACTIONS": NUM_ACTIONS,
    "BIGQUERY_TMP_FILE": BIGQUERY_TMP_FILE,
    "BIGQUERY_DATASET_ID": BIGQUERY_DATASET_ID,
    "BIGQUERY_LOCATION": BIGQUERY_LOCATION,
    "BIGQUERY_TABLE_ID": BIGQUERY_TABLE_ID,
}

OBSERVATION = np.zeros((int(BATCH_SIZE), int(RANK_K))).tolist()
PREDICTED_ACTION = np.zeros(int(BATCH_SIZE), dtype=np.int).tolist()
NUM_OBSERVATIONS = 5
OBSERVATIONS = [
    {"observation": OBSERVATION} for _ in range(NUM_OBSERVATIONS)
]
PREDICTED_ACTIONS = [
    {"predicted_action": PREDICTED_ACTION} for _ in range(NUM_OBSERVATIONS)
]

DATA_JSON = json.dumps({
    "observations": OBSERVATIONS,
    "predicted_actions": PREDICTED_ACTIONS,
})
DATA_BYTES = DATA_JSON.encode("utf-8")
EVENT = {"data": base64.b64encode(DATA_BYTES)}

NUM_TRAJECTORY_ELEMENTS = len(
    ("step_type", "observation", "action", "policy_info", "next_step_type",
     "reward", "discount"))


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


class TestLogger(unittest.TestCase):
  """Test class for the Logger."""

  def setUp(self):
    super().setUp()

    self.mock_os_getenv = mock.patch("os.getenv").start()

    self.mock_env = mock.MagicMock()

    self.mock_movielens_env = mock.patch(
        "tf_agents.bandits.environments.movielens_py_environment." +
        "MovieLensPyEnvironment").start()
    mock_tf_py_env = mock.patch(
        "tf_agents.environments.tf_py_environment.TFPyEnvironment").start()
    mock_tf_py_env.return_value = self.mock_env

    self.mock_transition = mock.patch(
        "tf_agents.trajectories.from_transition").start()

    tmp_fd, self.bigquery_tmp_file = tempfile.mkstemp()
    self.addCleanup(os.close, tmp_fd)
    self.addCleanup(os.remove, self.bigquery_tmp_file)

    self.mock_json_dumps = mock.patch("json.dumps").start()
    def create_placeholder_json(json_data: str) -> None:
      del json_data  # Unused.
      return "{'content': 'PLACEHOLDER JSON'}"
    self.mock_json_dumps.side_effect = create_placeholder_json

    self.mock_client = mock.patch("google.cloud.bigquery.Client").start()
    self.mock_client.return_value = mock.MagicMock()
    self.mock_dataset = mock.patch("google.cloud.bigquery.Dataset").start()
    self.mock_load_job_config = mock.patch(
        "google.cloud.bigquery.LoadJobConfig").start()

  def tearDown(self):
    super().tearDown()
    mock.patch.stopall()

  def test_given_valid_arguments_log_execute_all_steps(self):
    """Tests `log` executes all steps given valid arguments."""
    self.mock_os_getenv.side_effect = build_side_effect_function(ENV_VARS)
    patcher_write_trajectories = mock.patch(
        "src.logger.main.write_trajectories_to_file")
    mock_write_trajectories = patcher_write_trajectories.start()
    patcher_append_dataset = mock.patch(
        "src.logger.main.append_dataset_to_bigquery")
    mock_append_dataset = patcher_append_dataset.start()

    main.log_prediction_to_bigquery(EVENT, None)

    # Assert the MovieLens environment is built.
    self.mock_movielens_env.assert_called_once_with(
        RAW_DATA_PATH,
        int(RANK_K),
        int(BATCH_SIZE),
        num_movies=int(NUM_ACTIONS),
        csv_delimiter="\t")

    # Assert write_trajectories_to_file is called.
    mock_write_trajectories.assert_called_once()

    # Assert append_dataset_to_bigquery is called.
    mock_append_dataset.assert_called_once_with(
        project_id=PROJECT_ID,
        dataset_file=DATASET_FILE,
        bigquery_dataset_id=BIGQUERY_DATASET_ID,
        bigquery_location=BIGQUERY_LOCATION,
        bigquery_table_id=BIGQUERY_TABLE_ID)

    patcher_write_trajectories.stop()
    patcher_append_dataset.stop()

  def test_log_unpack_observations_and_predicted_actions_correctly(self):
    """Tests `log` correctly unpacks the Python quantities from `EVENT`."""
    self.mock_os_getenv.side_effect = build_side_effect_function(ENV_VARS)
    patcher_write_trajectories = mock.patch(
        "src.logger.main.write_trajectories_to_file")
    mock_write_trajectories = patcher_write_trajectories.start()
    patcher_append_dataset = mock.patch(
        "src.logger.main.append_dataset_to_bigquery")
    patcher_append_dataset.start()

    main.log_prediction_to_bigquery(EVENT, None)

    mock_write_trajectories.assert_called_once_with(
        dataset_file=DATASET_FILE,
        environment=self.mock_env,
        observations=OBSERVATIONS,
        predicted_actions=PREDICTED_ACTIONS)

    patcher_write_trajectories.stop()
    patcher_append_dataset.stop()

  def test_given_float_batch_size_log_raise_exception(self):
    """Tests given a float as `BATCH_SIZE` `log` raises an exception."""
    env_vars = ENV_VARS.copy()
    env_vars["BATCH_SIZE"] = "8.0"
    self.mock_os_getenv.side_effect = build_side_effect_function(env_vars)
    patcher_write_trajectories = mock.patch(
        "src.logger.main.write_trajectories_to_file")
    patcher_write_trajectories.start()
    patcher_append_dataset = mock.patch(
        "src.logger.main.append_dataset_to_bigquery")
    patcher_append_dataset.start()

    with self.assertRaises(
        ValueError):  # Invalid literal for int() with base 10.
      main.log_prediction_to_bigquery(EVENT, None)

    patcher_write_trajectories.stop()
    patcher_append_dataset.stop()

  def test_given_float_rank_k_log_raise_exception(self):
    """Tests given a float as `RANK_K` `log` raises an exception."""
    env_vars = ENV_VARS.copy()
    env_vars["RANK_K"] = "20.0"
    self.mock_os_getenv.side_effect = build_side_effect_function(env_vars)
    patcher_write_trajectories = mock.patch(
        "src.logger.main.write_trajectories_to_file")
    patcher_write_trajectories.start()
    patcher_append_dataset = mock.patch(
        "src.logger.main.append_dataset_to_bigquery")
    patcher_append_dataset.start()

    with self.assertRaises(
        ValueError):  # Invalid literal for int() with base 10.
      main.log_prediction_to_bigquery(EVENT, None)

    patcher_write_trajectories.stop()
    patcher_append_dataset.stop()

  def test_given_float_num_actions_log_raise_exception(self):
    """Tests given a float as `NUM_ACTIONS` `log` raises an exception."""
    env_vars = ENV_VARS.copy()
    env_vars["NUM_ACTIONS"] = "20.0"
    self.mock_os_getenv.side_effect = build_side_effect_function(env_vars)
    patcher_write_trajectories = mock.patch(
        "src.logger.main.write_trajectories_to_file")
    patcher_write_trajectories.start()
    patcher_append_dataset = mock.patch(
        "src.logger.main.append_dataset_to_bigquery")
    patcher_append_dataset.start()

    with self.assertRaises(
        ValueError):  # Invalid literal for int() with base 10.
      main.log_prediction_to_bigquery(EVENT, None)

    patcher_write_trajectories.stop()
    patcher_append_dataset.stop()

  def test_write_trajectories_to_file_correctly_unpacks_number_of_data(self):
    """Tests the Logger correctly unpacks observations and prediction actions."""
    patcher_get_trajectory = mock.patch(
        "src.logger.main.get_trajectory_from_environment")
    mock_get_trajectory = patcher_get_trajectory.start()
    patcher_build_dict = mock.patch(
        "src.logger.main.build_dict_from_trajectory")
    mock_build_dict = patcher_build_dict.start()

    main.write_trajectories_to_file(
        dataset_file=self.bigquery_tmp_file,
        environment=self.mock_env,
        observations=OBSERVATIONS,
        predicted_actions=PREDICTED_ACTIONS)

    self.assertEqual(mock_get_trajectory.call_count, NUM_OBSERVATIONS)
    self.assertEqual(mock_build_dict.call_count, NUM_OBSERVATIONS)

    patcher_get_trajectory.stop()
    patcher_build_dict.stop()

  def test_write_trajectories_to_file_write_newline_per_json_dump(self):
    """Tests the Logger writes a newline after each Trajectory JSON."""
    patcher_get_trajectory = mock.patch(
        "src.logger.main.get_trajectory_from_environment")
    patcher_get_trajectory.start()
    patcher_build_dict = mock.patch(
        "src.logger.main.build_dict_from_trajectory")
    patcher_build_dict.start()

    main.write_trajectories_to_file(
        dataset_file=self.bigquery_tmp_file,
        environment=self.mock_env,
        observations=OBSERVATIONS,
        predicted_actions=PREDICTED_ACTIONS)

    with open(self.bigquery_tmp_file, "r") as f:
      newline_count = sum(1 for trajectory_json in f)
    self.assertEqual(newline_count, self.mock_json_dumps.call_count)

    patcher_get_trajectory.stop()
    patcher_build_dict.stop()

  def test_append_to_dataset_to_bigquery_fetch_correct_data(self):
    """Tests `append_dataset_to_bigquery` fetches the correct data."""
    main.append_dataset_to_bigquery(
        project_id=PROJECT_ID,
        dataset_file=self.bigquery_tmp_file,
        bigquery_dataset_id=BIGQUERY_DATASET_ID,
        bigquery_location=BIGQUERY_LOCATION,
        bigquery_table_id=BIGQUERY_TABLE_ID)

    self.mock_client.assert_called_once_with(project=PROJECT_ID)
    self.mock_dataset.assert_called_once_with(BIGQUERY_DATASET_ID)
    args, _ = self.mock_client.return_value.load_table_from_file.call_args
    self.assertEqual(args[1], BIGQUERY_TABLE_ID)

  def test_append_to_dataset_to_bigquery_do_not_overwrite(self):
    """Tests `append_dataset_to_bigquery` indeed appends to the dataset."""
    main.append_dataset_to_bigquery(
        project_id=PROJECT_ID,
        dataset_file=self.bigquery_tmp_file,
        bigquery_dataset_id=BIGQUERY_DATASET_ID,
        bigquery_location=BIGQUERY_LOCATION,
        bigquery_table_id=BIGQUERY_TABLE_ID)

    _, kwargs = self.mock_load_job_config.call_args
    self.assertEqual(kwargs["write_disposition"],
                     bigquery.WriteDisposition.WRITE_APPEND)

  def test_append_to_dataset_to_bigquery_with_num_trajectory_elements(self):
    """Tests the function uses a schema with `NUM_TRAJECTORY_ELEMENTS`."""
    main.append_dataset_to_bigquery(
        project_id=PROJECT_ID,
        dataset_file=self.bigquery_tmp_file,
        bigquery_dataset_id=BIGQUERY_DATASET_ID,
        bigquery_location=BIGQUERY_LOCATION,
        bigquery_table_id=BIGQUERY_TABLE_ID)

    _, kwargs = self.mock_load_job_config.call_args
    self.assertEqual(len(kwargs["schema"]), NUM_TRAJECTORY_ELEMENTS)


if __name__ == "__main__":
  unittest.main()
