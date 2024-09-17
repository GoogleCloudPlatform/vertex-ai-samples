"""Converts ICN CSV/JSONL files to TFRecord with apache beam."""

import json
from os import path
from typing import Any, Dict, Sequence, Union, cast

from absl import logging
import apache_beam as beam
import pandas as pd
import tensorflow as tf

from data_converter import common_lib


_COLUMN_NAMES = [
    common_lib.COLUMN_NAME_ML_USE,
    common_lib.COLUMN_NAME_GCS_FILE_PATH,
    common_lib.COLUMN_NAME_LABEL,
]
_JSON_GCS_URI_KEY = 'imageGcsUri'
_JSON_CLASS_ANNOTATION_KEY = 'classificationAnnotation'
_JSON_RESOURCE_LABEL_KEY = 'dataItemResourceLabels'
_JSON_CLASS_NAME_KEY = 'displayName'
_JSON_ML_USE_KEY = 'aiplatform.googleapis.com/ml_use'


def build_tf_example(element: Dict[str, Union[str, int]]) -> tf.train.Example:
  """Builds a TF Example from an image uri and label.

  Args:
    element: A dict with the keys gcs_file_path and label.

  Returns:
    The created TF Example.
  """
  image_uri = cast(str, element[common_lib.COLUMN_NAME_GCS_FILE_PATH])
  label = cast(int, element[common_lib.COLUMN_NAME_LABEL])
  image_bytes, shape = common_lib.encode_image(image_uri, image_format='jpeg')
  features = tf.train.Features(
      feature={
          'image/encoded': common_lib.convert_to_feature(image_bytes),
          'image/format': common_lib.convert_to_string_feature('jpeg'),
          'image/height': common_lib.convert_to_feature(shape[0]),
          'image/width': common_lib.convert_to_feature(shape[1]),
          'image/class/label': common_lib.convert_to_feature(label),
      },
  )
  return tf.train.Example(features=features)


def _run_convert_pipeline(
    output_dir: str, df: pd.DataFrame, num_shards: Sequence[int]
) -> None:
  """Starts a Beam pipeline to write DataFrame as TF Records.

  Args:
    output_dir: TF Records output directory.
    df: DataFrame to convert from.
    num_shards: Number of shards for train/validation/test TFRecord files.
  """
  images_list = df.to_dict('records')

  def pipeline(root: beam.Pipeline):
    common_lib.beam_convert_tfexamples(
        root,
        images_list,
        build_tf_example,
        output_dir,
        num_shards,
    )

  common_lib.run_beam_pipeline(pipeline)


def _convert_df_to_tfrecord(
    df: pd.DataFrame,
    output_dir: str,
    split_ratio: Sequence[float],
    num_shard: Sequence[int],
) -> None:
  """Converts a DataFrame into three separate tfrecords for training, validation, and testing into output_dir.

  Args:
    df: DataFrame to convert.
    output_dir: The directory to save TFRecords and label_map.yaml.
    split_ratio: List specifying the training, validation, and testing splits
      for unassigned TFRecords.
    num_shard: Number of shards for train/validation/test TFRecord files.
  """
  # Replaces ml_use with common_lib string constants for consistency.
  common_lib.format_ml_use_column(df)
  common_lib.insert_missing_ml_use(df)

  # Ignores invalid rows.
  dropped_row_num = common_lib.drop_invalid_rows(df)
  if dropped_row_num > 0:
    logging.warning('Ignored %d invalid rows.', dropped_row_num)

  common_lib.replace_unassigned_ml_use(
      df[common_lib.COLUMN_NAME_ML_USE], split_ratio
  )

  # Converts labels to integers as required by training.
  new_labels, label_map = common_lib.create_label_map(
      df[common_lib.COLUMN_NAME_LABEL]
  )
  df[common_lib.COLUMN_NAME_LABEL] = new_labels
  label_map_path = path.join(output_dir, common_lib.LABEL_MAP_NAME)
  logging.info('Writing label map to %s.', label_map_path)
  common_lib.write_label_map(label_map_path, label_map)

  _run_convert_pipeline(output_dir, df, num_shard)


def convert_csv_to_tfrecord(
    input_csv: str,
    output_dir: str,
    split_ratio: Sequence[float] = (0.8, 0.1, 0.1),
    num_shard: Sequence[int] = (10, 10, 10),
) -> None:
  """Parses input_csv file into three separate tfrecords for training, validation, and testing into output_dir.

  The csv format is shown in
    https://cloud.google.com/vertex-ai/docs/image-data/classification/prepare-data#csv.

  If an ml_use column is not provided, one will be created.

  label_map.yaml containing the label map will be placed in output_dir.

  Args:
    input_csv: Name of the csv file.
    output_dir: The directory to save TFRecords and label_map.yaml.
    split_ratio: List specifying the training, validation, and testing splits
      for unassigned TFRecords.
    num_shard: Number of shards for train/validation/test TFRecord files.
  """
  with tf.io.gfile.GFile(input_csv, 'r') as f:
    df: pd.DataFrame = pd.read_csv(
        f, header=None, names=_COLUMN_NAMES, on_bad_lines='warn'
    )

  _convert_df_to_tfrecord(df, output_dir, split_ratio, num_shard)


def convert_jsonl_to_tfrecord(
    input_jsonl: str,
    output_dir: str,
    split_ratio: Sequence[float] = (0.8, 0.1, 0.1),
    num_shard: Sequence[int] = (10, 10, 10),
) -> None:
  """Parses input_jsonl file into three separate tfrecords for training, validation, and testing into output_dir.

  The JSONL format is shown in
    https://cloud.google.com/vertex-ai/docs/image-data/classification/prepare-data#json-lines.

  If an ml_use column is not provided, one will be created.

  label_map.yaml containing the label map will be placed in output_dir.

  Args:
    input_jsonl: Name of the JSONL file.
    output_dir: The directory to save TFRecords and label_map.yaml.
    split_ratio: List specifying the training, validation, and testing splits
      for unassigned TFRecords.
    num_shard: Number of shards for train/validation/test TFRecord files.
  """
  df_rows = []
  with tf.io.gfile.GFile(input_jsonl, 'r') as f:
    lines = f.read().rstrip().splitlines()

  for i, line in enumerate(lines, 1):
    try:
      item: Dict[str, Any] = json.loads(line)

      gcs_uri = item.get(_JSON_GCS_URI_KEY)
      label = item.get(_JSON_CLASS_ANNOTATION_KEY, {}).get(_JSON_CLASS_NAME_KEY)
      if not gcs_uri or not label:
        logging.warning('Invalid JSON at line %d, skipped.', i)
        continue

      ml_use = item.get(_JSON_RESOURCE_LABEL_KEY, {}).get(
          _JSON_ML_USE_KEY, common_lib.ML_USE_UNASSIGNED
      )
    except (json.JSONDecodeError, AttributeError):
      logging.warning('Invalid JSON at line %d, skipped.', i)
      continue

    df_rows.append([ml_use, gcs_uri, label])

  df = pd.DataFrame(
      data=df_rows,
      columns=[
          common_lib.COLUMN_NAME_ML_USE,
          common_lib.COLUMN_NAME_GCS_FILE_PATH,
          common_lib.COLUMN_NAME_LABEL,
      ],
  )

  _convert_df_to_tfrecord(df, output_dir, split_ratio, num_shard)
