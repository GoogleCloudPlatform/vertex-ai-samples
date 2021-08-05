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

"""The unit testing module for the Generator component."""
import os
import tempfile
import unittest
from unittest import mock

from src.generator import generator_component
import tensorflow as tf
from tf_agents.bandits.environments import movielens_py_environment


# Paths and configurations
RAW_DATA_PATH = "gs://[your-bucket-name]/[your-dataset-dir]/u.data"  # FILL IN
PROJECT_ID = "project-id"
BIGQUERY_DATASET_ID = f"{PROJECT_ID}.movielens_dataset"
BIGQUERY_LOCATION = "us"
BIGQUERY_TABLE_ID = f"{BIGQUERY_DATASET_ID}.training_dataset"


# Set Generator parameters.
DRIVER_STEPS = 2

# Hyperparameters
BATCH_SIZE = 8

# MovieLens simulation environment parameters
RANK_K = 20
NUM_ACTIONS = 20

NUM_TRAJECTORY_ELEMENTS = len(
    ("step_type", "observation", "action", "policy_info", "next_step_type",
     "reward", "discount"))


class TestGeneratorComponent(unittest.TestCase):
  """Test class for the Generator component."""

  def setUp(self):
    super().setUp()
    self.mock_env = mock.patch(
        "tf_agents.bandits.environments.movielens_py_environment").start()

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

    self.env = movielens_py_environment.MovieLensPyEnvironment(
        RAW_DATA_PATH,
        RANK_K,
        BATCH_SIZE,
        num_movies=NUM_ACTIONS,
        csv_delimiter="\t")  # This can catch dataset errors as well.
    self.mock_env.MovieLensPyEnvironment.return_value = self.env

  def tearDown(self):
    super().tearDown()
    mock.patch.stopall()

  def test_given_valid_arguments_generate_work(self):
    """Tests given valid arguments the component works."""
    bigquery_dataset_id, bigquery_location, bigquery_table_id = generator_component.generate_movielens_dataset_for_bigquery(
        project_id=PROJECT_ID,
        raw_data_path=RAW_DATA_PATH,
        batch_size=BATCH_SIZE,
        rank_k=RANK_K,
        num_actions=NUM_ACTIONS,
        driver_steps=DRIVER_STEPS,
        bigquery_tmp_file=self.bigquery_tmp_file,
        bigquery_dataset_id=BIGQUERY_DATASET_ID,
        bigquery_location=BIGQUERY_LOCATION,
        bigquery_table_id=BIGQUERY_TABLE_ID)

    # Assert generate_simulation_data is called.
    self.mock_env.MovieLensPyEnvironment.assert_called_once_with(
        RAW_DATA_PATH,
        RANK_K,
        BATCH_SIZE,
        num_movies=NUM_ACTIONS,
        csv_delimiter="\t")

    # Assert write_replay_buffer_to_file is called.
    self.mock_json_dumps.assert_called()

    # Assert load_dataset_into_bigquery is called.
    self.mock_client.assert_called_once_with(project=PROJECT_ID)
    self.mock_dataset.assert_called_once_with(BIGQUERY_DATASET_ID)
    self.mock_client.return_value.load_table_from_file.assert_called_once()

    # Check component outputs.
    self.assertEqual(bigquery_dataset_id, BIGQUERY_DATASET_ID)
    self.assertEqual(bigquery_location, BIGQUERY_LOCATION)
    self.assertEqual(bigquery_table_id, BIGQUERY_TABLE_ID)

  def test_given_zero_driver_steps_generate_work(self):
    """Tests the component works when `driver_steps` is zero."""
    bigquery_dataset_id, bigquery_location, bigquery_table_id = generator_component.generate_movielens_dataset_for_bigquery(
        project_id=PROJECT_ID,
        raw_data_path=RAW_DATA_PATH,
        batch_size=BATCH_SIZE,
        rank_k=RANK_K,
        num_actions=NUM_ACTIONS,
        driver_steps=0,
        bigquery_tmp_file=self.bigquery_tmp_file,
        bigquery_dataset_id=BIGQUERY_DATASET_ID,
        bigquery_location=BIGQUERY_LOCATION,
        bigquery_table_id=BIGQUERY_TABLE_ID)

    # Assert generate_simulation_data is called.
    self.mock_env.MovieLensPyEnvironment.assert_called_once_with(
        RAW_DATA_PATH,
        RANK_K,
        BATCH_SIZE,
        num_movies=NUM_ACTIONS,
        csv_delimiter="\t")

    # Assert write_replay_buffer_to_file is called.
    self.mock_json_dumps.assert_not_called()

    # Assert load_dataset_into_bigquery is called.
    self.mock_client.assert_called_once_with(project=PROJECT_ID)
    self.mock_dataset.assert_called_once_with(BIGQUERY_DATASET_ID)
    self.mock_client.return_value.load_table_from_file.assert_called_once()

    # Check component outputs.
    self.assertEqual(bigquery_dataset_id, BIGQUERY_DATASET_ID)
    self.assertEqual(bigquery_location, BIGQUERY_LOCATION)
    self.assertEqual(bigquery_table_id, BIGQUERY_TABLE_ID)

  def test_given_negative_driver_steps_generate_raise_exception(
      self):
    """Tests the component raises an exception when `driver_steps` is negative.
    """
    with self.assertRaises(tf.errors.InvalidArgumentError):
      generator_component.generate_movielens_dataset_for_bigquery(
          project_id=PROJECT_ID,
          raw_data_path=RAW_DATA_PATH,
          batch_size=BATCH_SIZE,
          rank_k=RANK_K,
          num_actions=NUM_ACTIONS,
          driver_steps=-1,
          bigquery_tmp_file=self.bigquery_tmp_file,
          bigquery_dataset_id=BIGQUERY_DATASET_ID,
          bigquery_location=BIGQUERY_LOCATION,
          bigquery_table_id=BIGQUERY_TABLE_ID)

  def test_given_float_driver_steps_generate_raise_exception(
      self):
    """Tests the component raises an exception when `driver_steps` is a float.
    """
    with self.assertRaises(TypeError):
      generator_component.generate_movielens_dataset_for_bigquery(
          project_id=PROJECT_ID,
          raw_data_path=RAW_DATA_PATH,
          batch_size=BATCH_SIZE,
          rank_k=RANK_K,
          num_actions=NUM_ACTIONS,
          driver_steps=0.5,
          bigquery_tmp_file=self.bigquery_tmp_file,
          bigquery_dataset_id=BIGQUERY_DATASET_ID,
          bigquery_location=BIGQUERY_LOCATION,
          bigquery_table_id=BIGQUERY_TABLE_ID)

  def test_append_json_write_newline_per_json_dump(self):
    """Tests the component writes a newline after each Trajectory JSON."""
    generator_component.generate_movielens_dataset_for_bigquery(
        project_id=PROJECT_ID,
        raw_data_path=RAW_DATA_PATH,
        batch_size=BATCH_SIZE,
        rank_k=RANK_K,
        num_actions=NUM_ACTIONS,
        driver_steps=DRIVER_STEPS,
        bigquery_tmp_file=self.bigquery_tmp_file,
        bigquery_dataset_id=BIGQUERY_DATASET_ID,
        bigquery_location=BIGQUERY_LOCATION,
        bigquery_table_id=BIGQUERY_TABLE_ID)

    with open(self.bigquery_tmp_file, "r") as f:
      newline_count = sum(1 for trajectory_json in f)
    self.assertEqual(newline_count, self.mock_json_dumps.call_count)

  def test_query_dataset_with_num_trajectory_elements(self):
    """Tests the component queries with a schema with `NUM_TRAJECTORY_ELEMENTS`.
    """
    generator_component.generate_movielens_dataset_for_bigquery(
        project_id=PROJECT_ID,
        raw_data_path=RAW_DATA_PATH,
        batch_size=BATCH_SIZE,
        rank_k=RANK_K,
        num_actions=NUM_ACTIONS,
        driver_steps=DRIVER_STEPS,
        bigquery_tmp_file=self.bigquery_tmp_file,
        bigquery_dataset_id=BIGQUERY_DATASET_ID,
        bigquery_location=BIGQUERY_LOCATION,
        bigquery_table_id=BIGQUERY_TABLE_ID)

    _, kwargs = self.mock_load_job_config.call_args
    self.assertEqual(len(kwargs["schema"]), NUM_TRAJECTORY_ELEMENTS)


if __name__ == "__main__":
  unittest.main()
