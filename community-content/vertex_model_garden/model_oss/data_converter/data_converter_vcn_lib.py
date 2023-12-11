"""Converts VCN CSV/JSONL files to TFRecord with apache beam."""

import json
from os import path
from typing import Any, Dict, Iterator, Sequence, Union, cast

from absl import logging
import apache_beam as beam
from apache_beam.io import tfrecordio
import numpy as np
import pandas as pd
import tensorflow as tf

from data_converter import common_lib
from util import constants


_COLUMN_NAMES = [
    common_lib.COLUMN_NAME_ML_USE,
    common_lib.COLUMN_NAME_GCS_FILE_PATH,
    common_lib.COLUMN_NAME_LABEL,
    common_lib.COLUMN_NAME_START_SEC,
    common_lib.COLUMN_NAME_END_SEC,
]
_JSON_GCS_URI_KEY = 'videoGcsUri'
_JSON_CLASS_ANNOTATION_KEY = 'timeSegmentAnnotations'
_JSON_CLASS_NAME_KEY = 'displayName'
_JSON_START_TIME_KEY = 'startTime'
_JSON_END_TIME_KEY = 'endTime'
_JSON_RESOURCE_LABEL_KEY = 'dataItemResourceLabels'
_JSON_ML_USE_KEY = 'aiplatform.googleapis.com/ml_use'


def build_tf_example(
    video_uri: str,
    label: int,
    start_sec: float,
    end_sec: float,
    output_fps: int,
) -> tf.train.SequenceExample:
  """Builds a TF Example from a video clip.

  Args:
    video_uri: GCS URI to the video file.
    label: Class label as an integer.
    start_sec: Start timestamp of the video clip in seconds.
    end_sec: End timestamp of the video clip in seconds.
    output_fps: The output frame rate per second.

  Returns:
    The created TF Example.
  """
  frame_bytes = common_lib.encode_video(
      video_uri, start_sec, end_sec, output_fps, image_format='jpg'
  )
  seq_example = tf.train.SequenceExample()
  seq_example.context.feature['clip/label/index'].int64_list.value[:] = [label]
  for frame in frame_bytes:
    seq_example.feature_lists.feature_list.get_or_create(
        'image/encoded'
    ).feature.add().bytes_list.value[:] = [frame]

  return seq_example


class AcquireTFExampleDoFn(beam.DoFn):
  """Beam DoFn to build TF Examples from a DataFrame row dict for VCN."""

  def __init__(self, output_fps: int):
    self._success_counter = beam.metrics.Metrics.counter(
        self.__class__.__name__, 'Success'
    )
    self._failure_counter = beam.metrics.Metrics.counter(
        self.__class__.__name__, 'Failure'
    )
    self._output_fps = output_fps

  def process(
      self, element: Dict[str, Union[float, int, str]]
  ) -> Iterator[tf.train.SequenceExample]:
    ml_use: str = cast(str, element[common_lib.COLUMN_NAME_ML_USE])
    video_uri: str = cast(str, element[common_lib.COLUMN_NAME_GCS_FILE_PATH])

    try:
      label: int = int(element[common_lib.COLUMN_NAME_LABEL])
      start_sec: float = float(element[common_lib.COLUMN_NAME_START_SEC])
      end_sec: float = float(element[common_lib.COLUMN_NAME_END_SEC])

      tf_example = build_tf_example(
          video_uri,
          label,
          start_sec,
          end_sec,
          self._output_fps,
      )
      self._success_counter.inc()
      yield beam.pvalue.TaggedOutput(ml_use, tf_example)
    except (ValueError, IOError) as err:
      logging.error('Failed to process %s', video_uri)
      logging.exception(err)
      self._failure_counter.inc()


def _run_convert_pipeline(
    output_dir: str,
    df: pd.DataFrame,
    num_shards: Sequence[int],
    output_fps: int,
) -> None:
  """Starts a Beam pipeline to write DataFrame as TF Records.

  Args:
    output_dir: TF Records output directory.
    df: DataFrame to convert from.
    num_shards: Number of shards for train/validation/test TFRecord files.
    output_fps: The output frame rate per second.
  """
  clip_list = df.to_dict('records')

  def pipeline(root):
    train, val, test = (
        root
        | 'Create PCollection' >> beam.Create(clip_list)
        | 'Convert to TF Example'
        >> beam.ParDo(AcquireTFExampleDoFn(output_fps)).with_outputs(
            constants.ML_USE_TRAINING,
            constants.ML_USE_VALIDATION,
            constants.ML_USE_TEST,
        )
    )
    _ = train | 'Save train TF Record' >> tfrecordio.WriteToTFRecord(
        path.join(output_dir, common_lib.TRAIN_TFRECORD_NAME),
        coder=beam.coders.ProtoCoder(tf.train.Example),
        num_shards=num_shards[0],
    )
    _ = val | 'Save val TF Record' >> tfrecordio.WriteToTFRecord(
        path.join(output_dir, common_lib.VALIDATION_TFRECORD_NAME),
        coder=beam.coders.ProtoCoder(tf.train.Example),
        num_shards=num_shards[1],
    )
    _ = test | 'Save test TF Record' >> tfrecordio.WriteToTFRecord(
        path.join(output_dir, common_lib.TEST_TFRECORD_NAME),
        coder=beam.coders.ProtoCoder(tf.train.Example),
        num_shards=num_shards[2],
    )

  common_lib.run_beam_pipeline(pipeline)


def _convert_df_to_tfrecord(
    df: pd.DataFrame,
    output_dir: str,
    split_ratio: Sequence[float],
    num_shard: Sequence[int],
    output_fps: int,
) -> None:
  """Converts a DataFrame into three separate tfrecords for training, validation, and testing into output_dir.

  Args:
    df: DataFrame to convert.
    output_dir: The directory to save TFRecords and label_map.yaml.
    split_ratio: List specifying the training, validation, and testing splits
      for unassigned TFRecords.
    num_shard: Number of shards for train/validation/test TFRecord files.
    output_fps: The output frame rate per second.
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

  # Missing start / end times are treated as 0, inf, respectively.
  df[common_lib.COLUMN_NAME_START_SEC].fillna(0, inplace=True)
  df[common_lib.COLUMN_NAME_END_SEC].fillna(np.inf, inplace=True)

  _run_convert_pipeline(output_dir, df, num_shard, output_fps)


def convert_csv_to_tfrecord(
    input_csv: str,
    output_dir: str,
    output_fps: int,
    split_ratio: Sequence[float] = (0.8, 0.1, 0.1),
    num_shard: Sequence[int] = (10, 10, 10),
) -> None:
  """Parses input_csv file into three separate tfrecords for training, validation, and testing into output_dir.

  The csv format is shown in
    https://cloud.google.com/vertex-ai/docs/video-data/classification/prepare-data#csv

  If an ml_use column is not provided, one will be created.

  label_map.yaml containing the label map will be placed in output_dir.

  Args:
    input_csv: Name of the csv file.
    output_dir: The directory to save TFRecords and label_map.yaml.
    output_fps: The output frame rate per second.
    split_ratio: List specifying the training, validation, and testing splits
      for unassigned TFRecords.
    num_shard: Number of shards for train/validation/test TFRecord files.
  """
  with tf.io.gfile.GFile(input_csv, 'r') as f:
    df: pd.DataFrame = pd.read_csv(
        f, header=None, names=_COLUMN_NAMES, on_bad_lines='warn'
    )

  _convert_df_to_tfrecord(df, output_dir, split_ratio, num_shard, output_fps)


def convert_jsonl_to_tfrecord(
    input_jsonl: str,
    output_dir: str,
    output_fps: int,
    split_ratio: Sequence[float] = (0.8, 0.1, 0.1),
    num_shard: Sequence[int] = (10, 10, 10),
) -> None:
  """Parses input_jsonl file into three separate tfrecords for training, validation, and testing into output_dir.

  The JSONL format is shown in
    https://cloud.google.com/vertex-ai/docs/video-data/classification/prepare-data#jsonl.

  If an ml_use column is not provided, one will be created.

  label_map.yaml containing the label map will be placed in output_dir.

  Args:
    input_jsonl: Name of the JSONL file.
    output_dir: The directory to save TFRecords and label_map.yaml.
    output_fps: The output frame rate per second.
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
      if not gcs_uri:
        logging.warning('Invalid JSON at line %d, skipped.', i)
        continue

      annotations = item.get(_JSON_CLASS_ANNOTATION_KEY, [])
      ml_use = item.get(_JSON_RESOURCE_LABEL_KEY, {}).get(
          _JSON_ML_USE_KEY, common_lib.ML_USE_UNASSIGNED
      )

      for j, annotation in enumerate(annotations):
        label = annotation.get(_JSON_CLASS_NAME_KEY)
        if not label:
          logging.warning('Invalid annotation #%d at line %d, skipped.', j, i)
          continue
        # The example in external documentation uses strings like "1.0s", so we
        # need to remove the "s" suffix.
        start_time = annotation.get(_JSON_START_TIME_KEY, '0').removesuffix('s')
        end_time = annotation.get(_JSON_END_TIME_KEY, 'inf').removesuffix('s')
        df_rows.append([ml_use, gcs_uri, label, start_time, end_time])
    except (json.JSONDecodeError, AttributeError):
      logging.warning('Invalid JSON at line %d, skipped.', i)
      continue

  df = pd.DataFrame(
      data=df_rows,
      columns=_COLUMN_NAMES,
  )

  _convert_df_to_tfrecord(df, output_dir, split_ratio, num_shard, output_fps)
