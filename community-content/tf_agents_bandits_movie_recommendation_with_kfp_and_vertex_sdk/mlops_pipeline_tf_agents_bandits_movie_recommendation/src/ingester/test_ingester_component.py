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

"""The unit testing module for the Ingester component."""
import unittest
from unittest import mock

# In order to make mocking in setUp work.
from google.cloud import bigquery  # pylint: disable=unused-import
import numpy as np
from src.ingester import ingester_component
import tensorflow as tf


# Paths and configurations
PROJECT_ID = "project-id"
BIGQUERY_DATASET_ID = f"{PROJECT_ID}.movielens_dataset"
BIGQUERY_TABLE_ID = f"{BIGQUERY_DATASET_ID}.training_dataset"
BIGQUERY_MAX_ROWS = 10
TFRECORD_FILE = "gs://bucket-name/dataset.tfrecord"

# (Hyper)Parameters.
BATCH_SIZE = 8
RANK_K = 20

# BigQuery table settings.
MUTABLE_TABLE_ROW = {
    "step_type": np.zeros(BATCH_SIZE),
    "observation": [{
        "observation_batch": np.zeros(RANK_K)
    } for _ in range(BATCH_SIZE)],
    "action": np.zeros(BATCH_SIZE),
    "policy_info": 0,
    "next_step_type": np.zeros(BATCH_SIZE),
    "reward": np.zeros(BATCH_SIZE),
    "discount": np.zeros(BATCH_SIZE),
}
TABLE_ROW = dict(tuple(MUTABLE_TABLE_ROW.items()))
NUM_ROWS = 5


class TestIngesterComponent(unittest.TestCase):
  """Test class for the Ingester component."""

  def setUp(self):
    super().setUp()

    self.mock_table = mock.MagicMock()
    # Mock `bigquery.table.RowIterator`.
    self.mock_table.total_rows = NUM_ROWS
    self.mock_table.__iter__.return_value = [TABLE_ROW for _ in range(NUM_ROWS)]

    self.mock_query_job = mock.MagicMock()
    self.mock_query_job.result.return_value = self.mock_table

    self.mock_client = mock.MagicMock()
    self.mock_client.query.return_value = self.mock_query_job

    self.mock_bigquery = mock.patch("google.cloud.bigquery").start()
    self.mock_bigquery.Client.return_value = self.mock_client

    self.mock_writer = mock.MagicMock()
    self.patcher_tfrecord_writer = mock.patch("tensorflow.io.TFRecordWriter")
    self.mock_tfrecord_writer = self.patcher_tfrecord_writer.start()
    # Mock `tensorflow.io.TFRecordWriter` context manager.
    self.mock_tfrecord_writer.return_value.__enter__.return_value = self.mock_writer

  def tearDown(self):
    super().tearDown()
    mock.patch.stopall()

  def test_given_valid_arguments_ingest_work(self):
    """Tests given valid arguments the component works."""
    tfrecord_file, = ingester_component.ingest_bigquery_dataset_into_tfrecord(
        project_id=PROJECT_ID,
        bigquery_table_id=BIGQUERY_TABLE_ID,
        bigquery_max_rows=BIGQUERY_MAX_ROWS,
        tfrecord_file=TFRECORD_FILE)

    # Assert read_data_from_bigquery is called.
    self.mock_bigquery.Client.assert_called_once_with(project=PROJECT_ID)
    self.mock_client.query.assert_called_once()

    # Assert write_tfrecords is called.
    self.mock_tfrecord_writer.assert_called_once()

    # Check component output.
    self.assertEqual(tfrecord_file, TFRECORD_FILE)

  def test_query_for_specified_num_results(self):
    """Tests the component queries for a specified num_results."""
    ingester_component.ingest_bigquery_dataset_into_tfrecord(
        project_id=PROJECT_ID,
        bigquery_table_id=BIGQUERY_TABLE_ID,
        bigquery_max_rows=BIGQUERY_MAX_ROWS,
        tfrecord_file=TFRECORD_FILE)

    self.mock_query_job.result.assert_called_once_with(
        max_results=BIGQUERY_MAX_ROWS)

  def test_write_once_in_tfrecord_per_data_row(self):
    """Tests the component writes once in TFRecord file per data row."""
    ingester_component.ingest_bigquery_dataset_into_tfrecord(
        project_id=PROJECT_ID,
        bigquery_table_id=BIGQUERY_TABLE_ID,
        bigquery_max_rows=BIGQUERY_MAX_ROWS,
        tfrecord_file=TFRECORD_FILE)

    self.mock_table.__iter__.assert_called_once()
    self.assertEqual(self.mock_writer.write.call_count, NUM_ROWS)

  def test_given_dir_as_tfrecord_file_ingest_raise_exception(
      self):
    """Tests the component raises an exception when `tfrecord_file` is a dir."""
    self.patcher_tfrecord_writer.stop()

    with self.assertRaises(tf.errors.FailedPreconditionError):
      ingester_component.ingest_bigquery_dataset_into_tfrecord(
          project_id=PROJECT_ID,
          bigquery_table_id=BIGQUERY_TABLE_ID,
          bigquery_max_rows=BIGQUERY_MAX_ROWS,
          tfrecord_file="./")

    self.mock_tfrecord_writer = self.patcher_tfrecord_writer.start()


if __name__ == "__main__":
  unittest.main()
