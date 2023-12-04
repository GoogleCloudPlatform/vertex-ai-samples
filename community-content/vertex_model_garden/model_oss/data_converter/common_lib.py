"""Library with functions to use for data conversion."""

import json
import os
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import uuid

from absl import logging
import apache_beam as beam
import cv2
import numpy as np
import pandas as pd
import PIL
from PIL import Image
import tensorflow as tf
import yaml

from util import constants
from util import fileutils
from apache_beam.options import pipeline_options

REFORMATTED_CSV_SUFFIX = '-reformatted.csv'

LABEL_MAP_NAME = 'label_map.yaml'

_SPLIT_RATIO_ERROR_THRESHOLD = 1e-5
# Internal constant. Only for distinguishing rows without ML use.
ML_USE_UNASSIGNED = 'unassigned'
ALL_ML_USES = (
    constants.ML_USE_TRAINING,
    constants.ML_USE_VALIDATION,
    constants.ML_USE_TEST,
    ML_USE_UNASSIGNED,
)
COLUMN_NAME_ML_USE = 'ml_use'
COLUMN_NAME_GCS_FILE_PATH = 'gcs_file_path'
COLUMN_NAME_LABEL = 'label'
COLUMN_NAME_START_SEC = 'start_sec'
COLUMN_NAME_END_SEC = 'end_sec'
# Output filenames
TRAIN_TFRECORD_NAME = 'train.tfrecord'
VALIDATION_TFRECORD_NAME = 'val.tfrecord'
TEST_TFRECORD_NAME = 'test.tfrecord'
# Jsonl keys
JSON_GCS_URI_KEY = 'imageGcsUri'
JSON_RESOURCE_LABEL_KEY = 'dataItemResourceLabels'
JSON_ML_USE_KEY = 'aiplatform.googleapis.com/ml_use'
# I/O parameters
READ_CHUNK_SIZE = 1024 * 1024 * 1024  # 1GB


class WriteToTFRecord(beam.DoFn):
  """DoFn to write TF examples to sharded TF record files."""

  def __init__(
      self,
      output_prefix: str,
      num_shards: int,
      convert_fn: Callable[[Dict[str, Any]], tf.train.Example],
  ):
    self.output_prefix = output_prefix
    self.num_shards = num_shards
    self.writer: list[tf.io.TFRecordWriter] = []
    self.sharded_files: list[str] = []
    self.convert_fn = convert_fn
    self.success_counter = beam.metrics.Metrics.counter(
        self.__class__.__name__, 'Success'
    )
    self.failure_counter = beam.metrics.Metrics.counter(
        self.__class__.__name__, 'Failure'
    )

  def start_bundle(self):
    logging.info('Start writing TF Record to %s.', self.output_prefix)
    unique_str = uuid.uuid4().hex
    for i in range(self.num_shards):
      uri = f'{self.output_prefix}-{i}-{unique_str}'
      self.sharded_files.append(uri)
      self.writer.append(tf.io.TFRecordWriter(uri))

  def process(self, data: Dict[str, Any]) -> Iterable[Tuple[int, str]]:
    try:
      example = self.convert_fn(data)
      data = example.SerializeToString()
      idx = hash(data) % self.num_shards
      self.writer[idx].write(data)
      self.success_counter.inc()
      yield (idx, self.sharded_files[idx])
    # pylint: disable-next=broad-exception-caught
    except Exception as err:
      logging.error('Failed to process %s', data)
      logging.exception(err)
      self.failure_counter.inc()

  def finish_bundle(self):
    logging.info('Finish writing TF Record to %s.', self.output_prefix)
    for writer in self.writer:
      writer.close()
    self.writer = []


def convert_to_feature(
    value: Union[List[Union[int, float, bytes]], int, float, bytes],
    value_type: Optional[str] = None,
) -> tf.train.Feature:
  """Converts the given python object to a tf.train.Feature.

  This is copied from tensorflow_models/official/vision/data/tfrecord_lib.py.

  Args:
    value: int, float, bytes or a list of them.
    value_type: optional, if specified, forces the feature to be of the given
      type. Otherwise, type is inferred automatically. Can be one of ['bytes',
      'int64', 'float', 'bytes_list', 'int64_list', 'float_list']

  Returns:
    feature: A tf.train.Feature object.
  """

  if value_type is None:
    element = value[0] if isinstance(value, list) else value

    if isinstance(element, bytes):
      value_type = 'bytes'

    elif isinstance(element, (int, np.integer)):
      value_type = 'int64'

    elif isinstance(element, (float, np.floating)):
      value_type = 'float'

    else:
      raise ValueError(
          'Cannot convert type {} to feature'.format(type(element))
      )

    if isinstance(value, list):
      value_type = value_type + '_list'

  if value_type == 'int64':
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

  elif value_type == 'int64_list':
    value = np.asarray(value).astype(np.int64).reshape(-1)
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

  elif value_type == 'float':
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

  elif value_type == 'float_list':
    value = np.asarray(value).astype(np.float32).reshape(-1)
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

  elif value_type == 'bytes':
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  elif value_type == 'bytes_list':
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

  else:
    raise ValueError('Unknown value_type parameter - {}'.format(value_type))


def convert_to_string_feature(
    value: str, encoding: str = 'utf-8'
) -> tf.train.Feature:
  """Returns a bytes_list from an encoded string."""
  return convert_to_feature(value.encode(encoding))


def convert_to_list_string_feature(
    lst: list[str], encoding: str = 'utf-8'
) -> tf.train.Feature:
  """Returns a bytes_list from a list of encoded strings."""
  return convert_to_feature([value.encode(encoding) for value in lst])


def create_ml_use_array_with_split(
    total_size: int,
    split_ratio: Sequence[float],
) -> list[str]:
  """Create randomized list of 'training', 'validation', 'test'.

  The list of will be of length total_size with ratios according to train_size,
  validation_size, and test_size.

  Args:
    total_size: Length of sequence to return
    split_ratio: Proportions to split into 'training', 'validation', and 'test'

  Returns:
    List containing 'training', 'validation', and 'test'
  """
  train_size, validation_size, _ = split_ratio
  num_train = round(train_size * total_size)
  num_validation = round(validation_size * total_size)
  num_test = total_size - num_train - num_validation
  ml_use_row = (
      [constants.ML_USE_TRAINING] * num_train
      + [constants.ML_USE_VALIDATION] * num_validation
      + [constants.ML_USE_TEST] * num_test
  )
  random.shuffle(ml_use_row)
  return ml_use_row


def format_ml_use_column(df: pd.DataFrame):
  df[COLUMN_NAME_ML_USE].replace(
      # We need to support non-standard ML uses other than documented ones,
      # since they are used by some existing datasets.
      [r'(?i)^train(ing)?$', r'(?i)^test$', r'(?i)^validat(ion|e)$'],
      [
          constants.ML_USE_TRAINING,
          constants.ML_USE_TEST,
          constants.ML_USE_VALIDATION,
      ],
      inplace=True,
      regex=True,
  )


def insert_missing_ml_use(df: pd.DataFrame) -> None:
  """For every row that does not have ml_use as the first column, insert a column containing 'unassigned' to the front.

  Args:
    df: The DataFrame to process. The first column should be 'ml_use'.
  """
  df[COLUMN_NAME_ML_USE].fillna(ML_USE_UNASSIGNED, inplace=True)
  rows_to_fill = ~df[COLUMN_NAME_ML_USE].isin(ALL_ML_USES)
  df.loc[rows_to_fill] = df[rows_to_fill].shift(
      axis=1, fill_value=ML_USE_UNASSIGNED
  )


def replace_unassigned_ml_use(
    ml_uses: List[str],
    split_ratio: Sequence[float],
):
  """Replace `unassigned` in ml_uses with `training`, `validation`, and `test` with ratios according to split_ratio.

  Args:
    ml_uses: List of ml_use string values.
    split_ratio: Proportions to split into `training`, `validation`, and `test`.
  """
  unassigned_indices = [
      i for i, ml_use in enumerate(ml_uses) if ml_use == ML_USE_UNASSIGNED
  ]
  ml_use_arr = create_ml_use_array_with_split(
      len(unassigned_indices), split_ratio
  )
  for unassigned_index, ml_use in zip(unassigned_indices, ml_use_arr):
    ml_uses[unassigned_index] = ml_use


def merge_seq_into_dicts(
    key: str, values: Sequence[Any], dicts: Sequence[Dict[Any, Any]]
):
  """Merges a list of values into a list of dicts, inserted with the given key.

  Args:
    key: Key to insert or overwrite in the dictionary.
    values: A list of values to insert.
    dicts: A list of dictionaries. Each value will be inserted into the
      corresponding dictionary. The original value will be overwritten if the
      key already existed.

  Raises:
    ValueError: The values and dicts have different lengths.
  """
  if len(values) != len(dicts):
    raise ValueError(
        f'Length of values and dicts must match, got {len(values)} and'
        f' {len(dicts)}'
    )
  for val, d in zip(values, dicts):
    d[key] = val


def drop_invalid_rows(df: pd.DataFrame) -> int:
  """Drops DataFrame rows missing the gcs_file_path column or the label column.

  Args:
    df: The DataFrame to process in place.

  Returns:
    The number of rows dropped.
  """
  original_rows = df.shape[0]
  df.dropna(subset=[COLUMN_NAME_GCS_FILE_PATH, COLUMN_NAME_LABEL], inplace=True)
  dropped_num = original_rows - df.shape[0]
  if dropped_num > 0:
    df.reset_index(drop=True, inplace=True)
  return dropped_num


def check_split_ratio(split_ratio: Sequence[float]):
  """Checks if the give split ratio is valid.

  Args:
    split_ratio: Proportions to split into 'training', 'validation', and 'test'

  Raises:
    ValueError: Must have valid entries, correct length, and sum to 1.
  """
  if len(split_ratio) != 3:
    raise ValueError('split_ratio must contain exactly 3 values.')
  if abs(sum(split_ratio) - 1) > _SPLIT_RATIO_ERROR_THRESHOLD:
    raise ValueError('split_ratio must sum to 1.')
  if not all([0 <= val <= 1 for val in split_ratio]):
    raise ValueError('Entries of split_ratio must be in the range [0, 1].')


def check_num_shard(num_shard: Sequence[int]):
  """Checks if the number of shards is valid.

  Args:
    num_shard: The number of shards for each tfrecord.

  Raises:
    ValueError: Must have valid entries and correct length.
  """
  if len(num_shard) != 3:
    raise ValueError('num_shard must contain exactly 3 values.')
  if not all([val >= 1 for val in num_shard]):
    raise ValueError('Shards must be at least 1.')


def create_label_map_yaml(meta_data_path: str, output_dir: str) -> None:
  """Generate label_map.yaml from meta_data.yaml.

  Args:
    meta_data_path: Path to a meta_data.yaml file.
    output_dir: Directory to output label_map.yaml.
  """
  tf.io.gfile.copy(
      meta_data_path, os.path.join(output_dir, LABEL_MAP_NAME), overwrite=True
  )


def reformat_bbox(
    bbox: Sequence[int], img_width: int, img_height: int
) -> Tuple[float, float, float, float]:
  """Converts XYWH unnormalized bounding box with to a normalized XYXY bounding box.

  Args:
    bbox: Relative bounding box with unnormalized coordinates as [x, y, width,
      height].
    img_width: Image's pixel width.
    img_height: Image's pixel height.

  Returns:
    Absolute bounding box with normalized coordinates as
      [xmin, ymin, xmax, ymax].
  """
  x, y, width, height = bbox
  xmin = x / img_width
  ymin = y / img_height
  xmax = (x + width) / img_width
  ymax = (y + height) / img_height
  return xmin, ymin, xmax, ymax


def encode_image(
    filepath: str,
    output_shape: Optional[Sequence[int]] = None,
    image_format: str = 'png',
) -> Tuple[bytes, Sequence[int]]:
  """Encodes an image at the given path.

  Args:
    filepath: Path to the image.
    output_shape: The output shape of the image, (height, width).
    image_format: The format of the output image.

  Returns:
    The encoded image data in bytes and the shape of the image, (height, width).

  Raises:
    IOError: The image file is corrupt.
  """
  filepath = fileutils.force_gcs_fuse_path(filepath)
  with open(filepath, 'rb') as f:
    # If an output_shape is specified, resize the image and set data to the new
    # bytes.
    try:
      img = Image.open(f)
    except PIL.UnidentifiedImageError as e:
      raise IOError(f'Failed to open {filepath}') from e

    try:
      if output_shape is not None:
        rgb_img = img.resize((output_shape[1], output_shape[0])).convert('RGB')
      else:
        rgb_img = img.convert('RGB')
      rgb_img = np.array(rgb_img)

      _, data = cv2.imencode(f'.{image_format}', rgb_img)
      data = data.tobytes()
      return data, rgb_img.shape
    except cv2.error as e:
      raise IOError(f'Failed to encode {filepath}') from e
    finally:
      img.close()


def encode_video(
    filepath: str,
    start_sec: float,
    end_sec: float,
    output_fps: int = 5,
    output_shape: Optional[Sequence[int]] = None,
    image_format: str = 'jpg',
) -> Sequence[bytes]:
  """Encodes a video clip at the given path with start and end timestamps.

  Args:
    filepath: Path to the video.
    start_sec: Start timestamp of the video clip in seconds.
    end_sec: End timestamp of the video clip in seconds.
    output_fps: The output frame rate per second.
    output_shape: The output shape of each frame, (height, width).
    image_format: The format of the encoded frames.

  Returns:
    A list of the encoded frames data in bytes.

  Raises:
    IOError if the video file is corrupt.
  """
  filepath = fileutils.force_gcs_fuse_path(filepath)
  video = None

  try:
    video = cv2.VideoCapture(filepath)
    frames = []
    frame_interval = 1 / output_fps
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    original_fps = video.get(cv2.CAP_PROP_FPS)
    if not original_fps:
      # 0 or None indicates the video is invalid
      raise IOError(f'Failed to load {filepath}')
    video_length = total_frames / original_fps
    start_sec = max(start_sec, 0)
    end_sec = min(end_sec, video_length)
    for t in np.arange(start_sec, end_sec, frame_interval):
      frame_idx = min(total_frames - 1, round(t * original_fps))
      video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
      ret, frame = video.read()
      if not ret:
        raise IOError(f'Failed to load {filepath} at frame {frame_idx}')
      if output_shape is not None:
        frame = cv2.resize(frame, (output_shape[1], output_shape[0]))
      _, data = cv2.imencode(f'.{image_format}', frame)
      frames.append(data.tobytes())
  except cv2.error as e:
    raise IOError(f'Failed to load {filepath}') from e
  finally:
    if video:
      video.release()
  return frames


def create_label_map(
    labels: Sequence[str],
) -> Tuple[Sequence[int], Dict[int, str]]:
  """Creates a label map from a sequence of label strings.

  Args:
    labels: The sequence of labels to create label map from. Must not contain
      invalid values, which means data without labels should be filtered first.

  Returns:
    The integer labels and the mapping from integers to the original strings.
  """
  inverse_label_map: Dict[str, int] = dict()
  num_labels = 0
  for label in labels:
    if label not in inverse_label_map:
      num_labels += 1
      inverse_label_map[label] = num_labels
  int_labels = [inverse_label_map[label] for label in labels]
  label_map = {value: key for key, value in inverse_label_map.items()}
  return int_labels, label_map


def write_label_map(output_file: str, label_map: Dict[int, str]) -> None:
  """Writes a label map to the output file, which can be a GCS uri."""
  with tf.io.gfile.GFile(output_file, 'w') as f:
    yaml.dump({'label_map': label_map}, f)


def detectron_json_to_image_rows(input_json: str) -> list[Dict[str, Any]]:
  """Converts a Detectron JSON file to a list of image rows.

  Args:
    input_json: A path to a Detectron JSON or JSONL file.

  Returns:
    A list of dictionaries, where each dictionary contains Detectron format
    entry.

  Raises:
    ValueError: If the input JSON is invalid.
  """

  image_rows = []
  with tf.io.gfile.GFile(input_json, 'r') as f:
    for line in f:
      json_data = json.loads(line)
      if isinstance(json_data, dict):
        image_rows.append(json_data)
      elif isinstance(json_data, list):
        image_rows.extend(json_data)
      else:
        raise ValueError(
            'The input JSON is invalid. Dict or list is expected, but got '
            f'{type(json_data)}.'
        )
  return image_rows


def coco_json_to_image_rows(
    input_json: str,
) -> List[Dict[str, Any]]:
  """Converts a COCO JSON file to a list of image rows.

  Args:
    input_json: A path to a COCO JSON or JSONL file.

  Returns:
    A list of dictionaries, where each dictionary contains COCO format entry.

  Raises:
    ValueError: If the input JSON is invalid.
  """

  with tf.io.gfile.GFile(input_json, 'r') as f:
    coco_json = json.load(f)
  if 'annotations' not in coco_json:
    raise ValueError('"annotations" is not in the dataset.')
  if 'images' not in coco_json:
    raise ValueError('"images" is not in the dataset.')

  images = coco_json['images']
  return images


def partition_by_ml_use(element: Dict[str, Any], num_partitions: int) -> int:
  """Beam partition function to split data by ml_use."""
  del num_partitions
  try:
    partition = ALL_ML_USES.index(element[COLUMN_NAME_ML_USE])
  except Exception as e:
    raise ValueError(f'Invalid ML use: {element[COLUMN_NAME_ML_USE]}') from e
  return partition


def run_beam_pipeline(pipeline: Any) -> None:
  """Runs a beam pipeline. Works in both internal and docker environment."""
  options = pipeline_options.PipelineOptions([
      '--runner=FlinkRunner',
      '--faster_copy',
      '--max_parallelism', '8',
  ])
  p = beam.Pipeline(options=options)
  pipeline(p)
  result = p.run()
  result.wait_until_finish()
  for counter in result.metrics().query()['counters']:
    logging.info('%s counter: %s.', counter.key.metric.name, counter)
  logging.info('Completing beam pipeline.')


def beam_convert_tfexamples(
    root: beam.Pipeline,
    data_list: Sequence[Dict[str, Any]],
    convert_fn: Callable[[Dict[str, Any]], tf.train.Example],
    output_dir: str,
    num_shards: Sequence[int],
) -> None:
  """Constructs beam pipelines to convert train, val, test TF Examples."""
  names = [TRAIN_TFRECORD_NAME, VALIDATION_TFRECORD_NAME, TEST_TFRECORD_NAME]
  split_data = (
      root
      | 'Create PCollection' >> beam.Create(data_list)
      | 'Data split' >> beam.Partition(partition_by_ml_use, 3)
  )
  for i in range(3):
    ml_use: str = ALL_ML_USES[i]
    num_shard = num_shards[i]
    output_prefix = os.path.join(output_dir, names[i])
    _ = (
        split_data[i]
        | f'Convert {ml_use} TF Examples'
        >> beam.ParDo(WriteToTFRecord(output_prefix, num_shard, convert_fn))
        | f'Group {ml_use} TF Record files' >> beam.GroupBy(lambda x: x[0])
        | f'Merge {ml_use} TF Record files'
        >> beam.Map(merge_tfrecords_func(output_prefix, num_shard))
    )


def merge_tfrecords_func(output_prefix: str, num_shard: int) -> ...:
  """Returns a function to merge sharded worker output into expected shards."""
  output_prefix = fileutils.force_gcs_fuse_path(output_prefix)

  def merge_tfrecords(worker_output: Tuple[int, Sequence[Tuple[int, str]]]):
    idx = worker_output[0]
    files: Sequence[str] = np.unique([x[1] for x in worker_output[1]])
    output_file = f'{output_prefix}-{idx:05d}-of-{num_shard:05d}'
    with open(output_file, 'wb') as f:
      for file in files:
        logging.info('Merging %s.', file)
        file = fileutils.force_gcs_fuse_path(file)
        with open(file, 'rb') as fin:
          while True:
            data = fin.read(READ_CHUNK_SIZE)
            if not data:
              break
            f.write(data)
        os.remove(file)

  return merge_tfrecords
