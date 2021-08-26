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

"""The Ingester component for ingesting BigQuery data into TFRecords."""
from typing import NamedTuple


def ingest_bigquery_dataset_into_tfrecord(
    project_id: str,
    bigquery_table_id: str,
    tfrecord_file: str,
    bigquery_max_rows: int = None
) -> NamedTuple("Outputs", [
    ("tfrecord_file", str),
]):
  """Ingests data from BigQuery, formats them and outputs TFRecord files.

  Serves as the Ingester pipeline component:
  1. Reads data in BigQuery that contains 7 pieces of data: `step_type`,
    `observation`, `action`, `policy_info`, `next_step_type`, `reward`,
    `discount`.
  2. Packages the data as `tf.train.Example` objects and outputs them as
    TFRecord files.

  This function is to be built into a Kubeflow Pipelines (KFP) component. As a
  result, this function must be entirely self-contained. This means that the
  import statements and helper functions must reside within itself.

  Args:
    project_id: GCP project ID. This is required because otherwise the BigQuery
      client will use the ID of the tenant GCP project created as a result of
      KFP, which doesn't have proper access to BigQuery.
    bigquery_table_id: A string of the BigQuery table ID in the format of
      "project.dataset.table".
    tfrecord_file: Path to file to write the ingestion result TFRecords.
    bigquery_max_rows: Optional; maximum number of rows to ingest.

  Returns:
    A NamedTuple of the path to the output TFRecord file.
  """
  # pylint: disable=g-import-not-at-top
  import collections
  from typing import Optional

  from google.cloud import bigquery

  import tensorflow as tf

  def read_data_from_bigquery(
      project_id: str,
      bigquery_table_id: str,
      bigquery_max_rows: Optional[int]) -> bigquery.table.RowIterator:
    """Reads data from BigQuery at `bigquery_table_id` and creates an iterator.

    The table contains 7 columns that form `trajectories.Trajectory` objects:
    `step_type`, `observation`, `action`, `policy_info`, `next_step_type`,
    `reward`, `discount`.

    Args:
      project_id: GCP project ID. This is required because otherwise the
        BigQuery client will use the ID of the tenant GCP project created as a
        result of KFP, which doesn't have proper access to BigQuery.
      bigquery_table_id: A string of the BigQuery table ID in the format of
        "project.dataset.table".
      bigquery_max_rows: Optional; maximum number of rows to fetch.

    Returns:
      A row iterator over all data at `bigquery_table_id`.
    """
    # Construct a BigQuery client object.
    client = bigquery.Client(project=project_id)

    # Get dataset.
    query_job = client.query(
        f"""
        SELECT * FROM {bigquery_table_id}
        """
    )
    table = query_job.result(max_results=bigquery_max_rows)

    return table

  def _bytes_feature(tensor: tf.Tensor) -> tf.train.Feature:
    """Returns a `tf.train.Feature` with bytes from `tensor`.

    Args:
      tensor: A `tf.Tensor` object.

    Returns:
      A `tf.train.Feature` object containing bytes that represent the content of
      `tensor`.
    """
    value = tf.io.serialize_tensor(tensor)
    if isinstance(value, type(tf.constant(0))):
      value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  def build_example(data_row: bigquery.table.Row) -> tf.train.Example:
    """Builds a `tf.train.Example` from `data_row` content.

    Args:
      data_row: A `bigquery.table.Row` object that contains 7 pieces of data:
        `step_type`, `observation`, `action`, `policy_info`, `next_step_type`,
        `reward`, `discount`. Each piece of data except `observation` is a 1D
        array; `observation` is a 1D array of `{"observation_batch": 1D array}.`

    Returns:
      A `tf.train.Example` object holding the same data as `data_row`.
    """
    feature = {
        "step_type":
            _bytes_feature(data_row.get("step_type")),
        "observation":
            _bytes_feature([
                observation["observation_batch"]
                for observation in data_row.get("observation")
            ]),
        "action":
            _bytes_feature(data_row.get("action")),
        "policy_info":
            _bytes_feature(data_row.get("policy_info")),
        "next_step_type":
            _bytes_feature(data_row.get("next_step_type")),
        "reward":
            _bytes_feature(data_row.get("reward")),
        "discount":
            _bytes_feature(data_row.get("discount")),
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto

  def write_tfrecords(
      tfrecord_file: str,
      table: bigquery.table.RowIterator) -> None:
    """Writes the row data in `table` into TFRecords in `tfrecord_file`.

    Args:
      tfrecord_file: Path to file to write the TFRecords.
      table: A row iterator over all data to be written.
    """
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
      for data_row in table:
        example = build_example(data_row)
        writer.write(example.SerializeToString())

  table = read_data_from_bigquery(
      project_id=project_id,
      bigquery_table_id=bigquery_table_id,
      bigquery_max_rows=bigquery_max_rows)

  write_tfrecords(tfrecord_file, table)

  outputs = collections.namedtuple(
      "Outputs",
      ["tfrecord_file"])

  return outputs(tfrecord_file)


if __name__ == "__main__":
  from kfp.components import create_component_from_func

  ingest_bigquery_dataset_into_tfrecord_op = create_component_from_func(
    func=ingest_bigquery_dataset_into_tfrecord,
    base_image="tensorflow/tensorflow:2.5.0",
    output_component_file="component.yaml",
    packages_to_install=[
      "google-cloud-bigquery==2.20.0",
      "tensorflow==2.5.0",
    ],
  )
