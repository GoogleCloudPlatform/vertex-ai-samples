"""Converts IOD dataset files to TFRecord with apache beam."""

import collections
import json
from os import path
from typing import Any, Dict, Sequence

from absl import logging
import apache_beam as beam
import pandas as pd
import tensorflow as tf

from data_converter import common_lib
from util import constants

COLUMN_NAME_LABEL_INT = 'label_int'
_COLUMN_NAME_XMIN = 'X_MIN'
_COLUMN_NAME_YMIN = 'Y_MIN'
_COLUMN_NAME_XMAX = 'X_MAX'
_COLUMN_NAME_YMAX = 'Y_MAX'
COLUMN_NAMES = [
    common_lib.COLUMN_NAME_ML_USE,
    common_lib.COLUMN_NAME_GCS_FILE_PATH,
    common_lib.COLUMN_NAME_LABEL,
    _COLUMN_NAME_XMIN,
    _COLUMN_NAME_YMIN,
    'XMAX_NOT_USED',
    'YMIN_NOT_USED',
    _COLUMN_NAME_XMAX,
    _COLUMN_NAME_YMAX,
    'XMIN_NOT_USED',
    'YMAX_NOT_USED',
]
_BOUNDING_BOX_COLUMNS = [
    _COLUMN_NAME_XMIN,
    _COLUMN_NAME_YMIN,
    _COLUMN_NAME_XMAX,
    _COLUMN_NAME_YMAX,
]
_JSON_BBOX_ANNOTATIONS_KEY = 'boundingBoxAnnotations'
_JSON_DISPLAY_NAME_KEY = 'displayName'
_JSON_X_MIN_KEY = 'xMin'
_JSON_X_MAX_KEY = 'xMax'
_JSON_Y_MIN_KEY = 'yMin'
_JSON_Y_MAX_KEY = 'yMax'


def build_tf_example(image_row: Dict[str, Any]) -> tf.train.Example:
  """Builds a TF Example from an image row.

  Args:
    image_row: A dictionary containing information about the image, such as its
      GCS uri, labels, and bounding box coordinates.

  Returns:
    A tf.train.Example containing the encoded image and optionally a
      bounding box and label.
  """
  image_uri = image_row[common_lib.COLUMN_NAME_GCS_FILE_PATH]
  image_bytes, shape = common_lib.encode_image(image_uri, image_format='jpeg')
  feature = {
      'image/encoded': common_lib.convert_to_feature(image_bytes),
      'image/format': common_lib.convert_to_string_feature('jpeg'),
      'image/height': common_lib.convert_to_feature(shape[0]),
      'image/width': common_lib.convert_to_feature(shape[1]),
      'image/source_id': common_lib.convert_to_string_feature(image_uri),
      'image/object/bbox/xmin': common_lib.convert_to_feature(
          image_row[_COLUMN_NAME_XMIN]
      ),
      'image/object/bbox/ymin': common_lib.convert_to_feature(
          image_row[_COLUMN_NAME_YMIN]
      ),
      'image/object/bbox/xmax': common_lib.convert_to_feature(
          image_row[_COLUMN_NAME_XMAX]
      ),
      'image/object/bbox/ymax': common_lib.convert_to_feature(
          image_row[_COLUMN_NAME_YMAX]
      ),
      'image/object/class/text': common_lib.convert_to_list_string_feature(
          image_row[common_lib.COLUMN_NAME_LABEL]
      ),
      'image/object/class/label': common_lib.convert_to_feature(
          image_row[COLUMN_NAME_LABEL_INT]
      ),
  }
  return tf.train.Example(features=tf.train.Features(feature=feature))


def _run_convert_pipeline(
    output_dir: str,
    image_rows: Sequence[Dict[str, Any]],
    num_shards: Sequence[int],
) -> None:
  """Starts a Beam pipeline to write DataFrame as TF Records.

  Args:
    output_dir: TF Records output directory.
    image_rows: Contains all necessary information to create a TF Example.
    num_shards: Number of shards for train/validation/test TFRecord files.
  """

  def pipeline(root: beam.Pipeline):
    common_lib.beam_convert_tfexamples(
        root,
        image_rows,
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

  # Specify bounding box columns to be numeric.
  df[_BOUNDING_BOX_COLUMNS] = df[_BOUNDING_BOX_COLUMNS].apply(pd.to_numeric)

  # Ignores invalid rows.
  dropped_row_num = common_lib.drop_invalid_rows(df)
  dropped_row_num += drop_rows_without_bbox(df)
  if dropped_row_num > 0:
    logging.warning('Ignored %d invalid rows.', dropped_row_num)

  # Converts labels to integers as required by training.
  int_labels, label_map = common_lib.create_label_map(
      df[common_lib.COLUMN_NAME_LABEL]
  )
  df[COLUMN_NAME_LABEL_INT] = int_labels
  label_map_path = path.join(output_dir, common_lib.LABEL_MAP_NAME)
  logging.info('Writing label map to %s.', label_map_path)
  common_lib.write_label_map(label_map_path, label_map)

  image_rows = _condense_bounding_boxes(df.to_dict(orient='records'))
  ml_uses = [row[common_lib.COLUMN_NAME_ML_USE] for row in image_rows]
  common_lib.replace_unassigned_ml_use(ml_uses, split_ratio)
  common_lib.merge_seq_into_dicts(
      common_lib.COLUMN_NAME_ML_USE, ml_uses, image_rows
  )

  _run_convert_pipeline(output_dir, image_rows, num_shard)


def _condense_bounding_boxes(
    image_rows: Sequence[Dict[str, Any]]
) -> Sequence[Dict[str, Any]]:
  """Gather all the bounding boxes in an image and put them in the same dictionary.

  Args:
    image_rows: List of dictionaries, each containing information about the
      image, such as its GCS uri, labels, and bounding box coordinates.

  Returns:
    List of dictionaries such that each contains all the bounding boxes for a
    given gcs_file_path.

  Raises:
    RuntimeError: This is raised when the input data contains images that have
      annotations in different ml_use classes.
  """
  output = {}
  for image_row in image_rows:
    ml_use = image_row[common_lib.COLUMN_NAME_ML_USE]
    gcs_file_path = image_row[common_lib.COLUMN_NAME_GCS_FILE_PATH]
    label = image_row[common_lib.COLUMN_NAME_LABEL]
    xmin = image_row[_COLUMN_NAME_XMIN]
    ymin = image_row[_COLUMN_NAME_YMIN]
    xmax = image_row[_COLUMN_NAME_XMAX]
    ymax = image_row[_COLUMN_NAME_YMAX]
    label_int = image_row[COLUMN_NAME_LABEL_INT]
    if gcs_file_path in output:
      d = output[gcs_file_path]
      if ml_use != common_lib.ML_USE_UNASSIGNED:
        if d[common_lib.COLUMN_NAME_ML_USE] == common_lib.ML_USE_UNASSIGNED:
          d[common_lib.COLUMN_NAME_ML_USE] = ml_use
        elif ml_use != d[common_lib.COLUMN_NAME_ML_USE]:
          raise RuntimeError(
              f'Image {gcs_file_path} can only be placed in one of'
              f' training/validation/test. It is currently in {ml_use} and'
              f' {d[common_lib.COLUMN_NAME_ML_USE]}.'
          )
      d[common_lib.COLUMN_NAME_LABEL].append(label)
      d[_COLUMN_NAME_XMIN].append(xmin)
      d[_COLUMN_NAME_YMIN].append(ymin)
      d[_COLUMN_NAME_XMAX].append(xmax)
      d[_COLUMN_NAME_YMAX].append(ymax)
      d[COLUMN_NAME_LABEL_INT].append(label_int)
    else:
      output[gcs_file_path] = {
          common_lib.COLUMN_NAME_ML_USE: ml_use,
          common_lib.COLUMN_NAME_GCS_FILE_PATH: gcs_file_path,
          common_lib.COLUMN_NAME_LABEL: [label],
          _COLUMN_NAME_XMIN: [xmin],
          _COLUMN_NAME_YMIN: [ymin],
          _COLUMN_NAME_XMAX: [xmax],
          _COLUMN_NAME_YMAX: [ymax],
          COLUMN_NAME_LABEL_INT: [label_int],
      }
  return list(output.values())


def convert_csv_to_tfrecord(
    input_csv: str,
    output_dir: str,
    split_ratio: Sequence[float] = (0.8, 0.1, 0.1),
    num_shard: Sequence[int] = (10, 10, 10),
) -> None:
  """Parses input_csv file into three separate tfrecords for training, validation, and testing into output_dir.

  The csv format is shown in
    https://cloud.google.com/vertex-ai/docs/image-data/object-detection/prepare-data#csv.

  If an ml_use column is not provided, one will be created.

  label_map.yaml containing the label map will be placed in output_dir.

  Args:
    input_csv: Name of the csv file.
    output_dir: The directory to save TFRecords and label_map.yaml.
    split_ratio: List specifying the train, validation, and test splits for
      unassigned TFRecords.
    num_shard: Number of shards for train/validation/test TFRecord files.
  """
  with tf.io.gfile.GFile(input_csv, 'r') as f:
    df: pd.DataFrame = pd.read_csv(
        f, header=None, names=COLUMN_NAMES, on_bad_lines='warn'
    )

  _convert_df_to_tfrecord(df, output_dir, split_ratio, num_shard)


def drop_rows_without_bbox(df: pd.DataFrame) -> int:
  """Drops DataFrame rows without bounding_boxes.

  Args:
    df: The DataFrame to process in place.

  Returns:
    The number of rows dropped.
  """
  invalid_rows = df.index[~(df[_BOUNDING_BOX_COLUMNS].notnull().all(axis=1))]
  dropped_num = len(invalid_rows)
  if dropped_num > 0:
    invalid_df = df.loc[invalid_rows].to_dict(orient='records')
    for entry in invalid_df:
      logging.warning('Skipping entry due to missing bounding box: %s.', entry)
    df.drop(invalid_rows, inplace=True)
    df.reset_index(drop=True, inplace=True)
  return dropped_num


def convert_coco_json_categories_to_label_map(
    categories: Sequence[Dict[str, Any]]
) -> Dict[int, str]:
  return {category['id']: category['name'] for category in categories}


def convert_coco_json_to_tfrecord(
    input_coco_json: str,
    output_dir: str,
    split_ratio: Sequence[float] = (0.8, 0.1, 0.1),
    num_shard: Sequence[int] = (10, 10, 10),
) -> None:
  """Parses input_csv file into three separate tfrecords for training, validation, and testing into output_dir.

  The COCO json format is shown here: https://cocodataset.org/#format-data.

  label_map.yaml containing the label map will be placed in output_dir.

  Args:
    input_coco_json: Name of coco json file.
    output_dir: The directory to save TFRecords and label_map.yaml.
    split_ratio: List specifying the train, validation, and test splits for
      dataset.
    num_shard: Number of shards for train/validation/test TFRecord files.
  """
  with tf.io.gfile.GFile(input_coco_json, 'r') as f:
    coco_json = json.load(f)
  # Writes label map from coco json categories.
  label_map = convert_coco_json_categories_to_label_map(
      coco_json[constants.COCO_JSON_CATEGORIES]
  )
  label_map_path = path.join(output_dir, common_lib.LABEL_MAP_NAME)
  logging.info('Writes label map to %s.', label_map_path)
  common_lib.write_label_map(label_map_path, label_map)

  img_to_anns = collections.defaultdict(list)
  imgs = {}
  if constants.COCO_JSON_ANNOTATIONS in coco_json:
    for ann in coco_json[constants.COCO_JSON_ANNOTATIONS]:
      img_to_anns[ann[constants.COCO_JSON_ANNOTATION_IMAGE_ID]].append(ann)

  if constants.COCO_JSON_IMAGES in coco_json:
    for img in coco_json[constants.COCO_JSON_IMAGES]:
      imgs[img[constants.COCO_JSON_IMAGE_ID]] = img

  df_rows = []

  for image_id, annotations in img_to_anns.items():
    img = imgs[image_id]
    for ann in annotations:
      xmin, ymin, xmax, ymax = common_lib.reformat_bbox(
          ann[constants.COCO_ANNOTATION_BBOX],
          img[constants.COCO_JSON_IMAGE_WIDTH],
          img[constants.COCO_JSON_IMAGE_HEIGHT],
      )
      df_rows.append([
          common_lib.ML_USE_UNASSIGNED,
          img[constants.COCO_JSON_IMAGE_COCO_URL],
          label_map[ann[constants.COCO_JSON_ANNOTATION_CATEGORY_ID]],
          xmin,
          ymin,
          xmax,
          ymin,
          xmax,
          ymax,
          xmin,
          ymax,
          ann[constants.COCO_JSON_ANNOTATION_CATEGORY_ID],
      ])
  df = pd.DataFrame(
      data=df_rows,
      columns=COLUMN_NAMES + [COLUMN_NAME_LABEL_INT],
  )

  # Replaces ml_use with common_lib string constants for consistency.
  common_lib.format_ml_use_column(df)
  common_lib.insert_missing_ml_use(df)

  # Species bounding box columns to be numeric.
  df[_BOUNDING_BOX_COLUMNS] = df[_BOUNDING_BOX_COLUMNS].apply(pd.to_numeric)

  # Ignores invalid rows.
  dropped_row_num = common_lib.drop_invalid_rows(df)
  dropped_row_num += drop_rows_without_bbox(df)
  if dropped_row_num > 0:
    logging.warning('Ignored %d invalid rows.', dropped_row_num)

  image_rows = _condense_bounding_boxes(df.to_dict(orient='records'))
  ml_uses = [row[common_lib.COLUMN_NAME_ML_USE] for row in image_rows]
  common_lib.replace_unassigned_ml_use(ml_uses, split_ratio)
  common_lib.merge_seq_into_dicts(
      common_lib.COLUMN_NAME_ML_USE, ml_uses, image_rows
  )

  _run_convert_pipeline(output_dir, image_rows, num_shard)


def convert_jsonl_to_tfrecord(
    input_jsonl: str,
    output_dir: str,
    split_ratio: Sequence[float] = (0.8, 0.1, 0.1),
    num_shard: Sequence[int] = (10, 10, 10),
) -> None:
  """Parses input_jsonl file into three separate tfrecords for training, validation, and testing into output_dir.

  The JSONL format is shown in
    https://cloud.google.com/vertex-ai/docs/image-data/object-detection/prepare-data#json-lines.

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

  for i, line in enumerate(lines, start=1):
    try:
      item: Dict[str, Any] = json.loads(line)
    except (json.JSONDecodeError, AttributeError):
      logging.warning('Invalid JSON at line %d skipped.', i)
      continue

    gcs_uri = item.get(common_lib.JSON_GCS_URI_KEY)
    if not gcs_uri:
      logging.warning(
          'Invalid JSON at line %d skipped. Missing gcs_uri_key.', i
      )
      continue
    ml_use = item.get(common_lib.JSON_RESOURCE_LABEL_KEY, {}).get(
        common_lib.JSON_ML_USE_KEY, common_lib.ML_USE_UNASSIGNED
    )

    for bbox in item.get(_JSON_BBOX_ANNOTATIONS_KEY, []):
      label = bbox.get(_JSON_DISPLAY_NAME_KEY)
      xmin = bbox.get(_JSON_X_MIN_KEY)
      ymin = bbox.get(_JSON_Y_MIN_KEY)
      xmax = bbox.get(_JSON_X_MAX_KEY)
      ymax = bbox.get(_JSON_Y_MAX_KEY)

      df_rows.append([ml_use, gcs_uri, label, xmin, ymin, xmax, ymax])

  df = pd.DataFrame(
      data=df_rows,
      columns=[
          common_lib.COLUMN_NAME_ML_USE,
          common_lib.COLUMN_NAME_GCS_FILE_PATH,
          common_lib.COLUMN_NAME_LABEL,
          _COLUMN_NAME_XMIN,
          _COLUMN_NAME_YMIN,
          _COLUMN_NAME_XMAX,
          _COLUMN_NAME_YMAX,
      ],
  )
  _convert_df_to_tfrecord(df, output_dir, split_ratio, num_shard)
