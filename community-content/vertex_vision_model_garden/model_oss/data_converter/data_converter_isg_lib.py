"""Python script to convert different file formats for ISG to tfrecords."""

import hashlib
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from absl import logging
import apache_beam as beam
from apache_beam.io import tfrecordio
import cv2
import numpy as np
from pycocotools import coco
import tensorflow as tf
import yaml

from data_converter import common_lib
from util import constants
from util import fileutils

_IMAGE_FORMAT = 'PNG'


def build_tf_example(
    image_info: dict[str, Union[str, int]],
    segmentation_image: List[List[int]],
    output_shape: Optional[Tuple[int, int]] = None,
) -> tf.train.Example:
  """Encodes an image and its segmentation mask into a tf.train.Example.

  Args:
    image_info: A dictionary containing information about the image, such as its
      file name, height, and width.
    segmentation_image: 2D image in list of lists having category ids.
    output_shape: The desired output shape of the image. If None, the original
      image shape will be used.

  Returns:
    A tf.train.Example containing the encoded image and segmentation mask.

  Raises:
    IOError: If image cannot be found in the path.
  """
  file_name = image_info[constants.COCO_JSON_FILE_NAME]
  height = int(image_info[constants.COCO_JSON_IMAGE_HEIGHT])
  width = int(image_info[constants.COCO_JSON_IMAGE_WIDTH])

  segmentation_image = np.expand_dims(
      np.asarray(segmentation_image, dtype=np.int32), axis=-1
  )
  _, encoded_seg = cv2.imencode(f'.{_IMAGE_FORMAT.lower()}', segmentation_image)
  encoded_seg = encoded_seg.tobytes()

  encoded_img, _ = common_lib.encode_image(
      image_info[constants.COCO_JSON_IMAGE_COCO_URL],
      output_shape=output_shape,
      image_format=_IMAGE_FORMAT.lower(),
  )

  key = hashlib.sha256(encoded_img).hexdigest()

  return tf.train.Example(
      features=tf.train.Features(
          feature={
              'image/height': common_lib.convert_to_feature(height),
              'image/width': common_lib.convert_to_feature(width),
              'image/filename': common_lib.convert_to_string_feature(file_name),
              'image/sha256': common_lib.convert_to_string_feature(key),
              'image/encoded': common_lib.convert_to_feature(encoded_img),
              'image/format': common_lib.convert_to_string_feature(
                  _IMAGE_FORMAT
              ),
              'image/segmentation/class/encoded': common_lib.convert_to_feature(
                  encoded_seg
              ),
              'image/segmentation/class/format': (
                  common_lib.convert_to_string_feature(_IMAGE_FORMAT)
              ),
              'image/segmentation/class/height': common_lib.convert_to_feature(
                  height
              ),
              'image/segmentation/class/width': common_lib.convert_to_feature(
                  width
              ),
          }
      )
  )


class AcquireTFExampleDoFn(beam.DoFn):
  """Beam DoFn to build TF Examples from a single row of image_info data."""

  # These tags will be used to tag the outputs of this DoFn.
  output_tag_train = constants.ML_USE_TRAINING
  output_tag_validation = constants.ML_USE_VALIDATION
  output_tag_test = constants.ML_USE_TEST

  valid_ml_use_set = set(
      [output_tag_train, output_tag_validation, output_tag_test]
  )

  def __init__(self, output_shape: Optional[Tuple[int, int]] = None):
    self.acquired_examples_counter = beam.metrics.Metrics.counter(
        self.__class__.__name__, 'Success'
    )
    self.failure_counter = beam.metrics.Metrics.counter(
        self.__class__.__name__, 'Failure'
    )
    self.output_shape = output_shape

  def process(
      self,
      row: Tuple[str, Dict[str, Union[str, int]], List[List[int]]],
  ) -> Iterator[tf.train.Example]:
    ml_use, image_info, annotation_info = row
    if ml_use not in self.valid_ml_use_set:
      logging.warning('ml_use invalid: %s', ml_use)
      self.failure_counter.inc()
      return

    try:
      tf_example = build_tf_example(
          image_info, annotation_info, self.output_shape
      )
    except IOError as e:
      logging.warning('Failed to build TF Example: %s', e)
      self.failure_counter.inc()
    else:
      self.acquired_examples_counter.inc()
      yield beam.pvalue.TaggedOutput(ml_use, tf_example)


def _define_data_conversion_pipeline(
    root: beam.Pipeline,
    ml_use_rows: List[str],
    image_rows: List[Dict[str, Union[str, int]]],
    segmentation_rows: List[List[List[int]]],
    output_dir: str,
    output_shape: Optional[Tuple[int, int]],
    num_shard_list: List[int],
):
  """Define a data conversion pipeline.

  Args:
    root: A Beam pipeline.
    ml_use_rows: List containing the ml_use.
    image_rows: List of dictionaries containing information about the image,
      such as its file name, height, and width.
    segmentation_rows: List of 2D images of integers representing segmentation
      masks.
    output_dir: Directory where the output TFRecords will be written.
    output_shape: Desired output shape of the image. If None, the original image
      shape will be used.
    num_shard_list: Number of shards to write to each output TFRecord.

  Returns:
    A Beam pipeline.
  """
  train, validation, test = (
      root
      | 'Load ml use and image rows to beam'
      >> beam.Create(zip(ml_use_rows, image_rows, segmentation_rows))
      | 'Build TF Examples'
      >> beam.ParDo(AcquireTFExampleDoFn(output_shape)).with_outputs(
          AcquireTFExampleDoFn.output_tag_train,
          AcquireTFExampleDoFn.output_tag_validation,
          AcquireTFExampleDoFn.output_tag_test,
      )
  )

  # Save each split to TFRecord.
  _ = train | 'Save train split to TFRecord' >> tfrecordio.WriteToTFRecord(
      os.path.join(output_dir, common_lib.TRAIN_TFRECORD_NAME),
      coder=beam.coders.ProtoCoder(tf.train.Example),
      num_shards=num_shard_list[0],
  )
  _ = (
      validation
      | 'Save validation split to TFRecord'
      >> tfrecordio.WriteToTFRecord(
          os.path.join(output_dir, common_lib.VALIDATION_TFRECORD_NAME),
          coder=beam.coders.ProtoCoder(tf.train.Example),
          num_shards=num_shard_list[1],
      )
  )
  _ = test | 'Save test split to TFRecord' >> tfrecordio.WriteToTFRecord(
      os.path.join(output_dir, common_lib.TEST_TFRECORD_NAME),
      coder=beam.coders.ProtoCoder(tf.train.Example),
      num_shards=num_shard_list[2],
  )


def _image_info_to_segmentation_image(
    img: Dict[str, Any],
    coco_dataset: coco.COCO,
    label_id_by_category_id: Dict[int, int],
) -> List[List[int]]:
  """Convert image information to a segmentation image.

  Args:
      img: The image information.
      coco_dataset: The COCO dataset.
      label_id_by_category_id: The mapping from label id used for training to
        category_id defined in dataset.

  Returns:
      The segmentation image.

  Raises:
    ValueError: If the mask size does not match the image or if a pixel has
      multiple labels.
  """
  seg_img = np.zeros(
      shape=(
          img[constants.COCO_JSON_IMAGE_HEIGHT],
          img[constants.COCO_JSON_IMAGE_WIDTH],
      ),
      dtype=np.int32,
  )
  for ann in coco_dataset.imgToAnns[img[constants.COCO_JSON_IMAGE_ID]]:
    new_category_id = ann[constants.COCO_JSON_ANNOTATION_CATEGORY_ID]
    binary_mask = coco_dataset.annToMask(ann)
    if seg_img.shape != binary_mask.shape:
      raise ValueError(
          'Binary mask does not have the same shape as image. image_id:'
          f' {img["id"]}'
      )
    boolean_mask = binary_mask == 1
    if (seg_img[boolean_mask] != 0).any():
      raise ValueError(
          'Error: Some pixels have more than one label in image_id:'
          f' {img["id"]}.'
      )
    seg_img[boolean_mask] = label_id_by_category_id[new_category_id]

  return seg_img.tolist()


def get_input_rows(
    coco_dataset: coco.COCO,
    split_ratio: List[float],
    label_id_by_category_id: Dict[int, int],
) -> Tuple[List[str], List[Dict[str, Union[str, int]]], List[List[List[int]]]]:
  """Get input rows for training and validation.

  Args:
      coco_dataset: The COCO dataset.
      split_ratio: The split ratio for training and validation.
      label_id_by_category_id: The mapping from label id used for training to
        category_id defined in dataset.

  Returns:
      - A list of ml_use strings.
      - A list of image informations.
      - A list of segmentation images for the corresponding images.
  """
  image_rows = coco_dataset.dataset[constants.COCO_JSON_IMAGES]

  segmentation_rows = [
      _image_info_to_segmentation_image(
          img, coco_dataset, label_id_by_category_id
      )
      for img in image_rows
  ]

  ml_use_rows = common_lib.create_ml_use_array_with_split(
      len(image_rows), split_ratio
  )
  return ml_use_rows, image_rows, segmentation_rows


def beam_build_tfrecord_from_coco_json(
    input_json: str,
    output_dir: str,
    split_ratio: List[float],
    num_shard_list: List[int],
    output_shape: Optional[Tuple[int, int]] = None,
) -> None:
  """Builds TFRecord files from COCO dataset.

  The output file names are `_TRAIN_TFRECORD_NAME`, `_VALIDATION_TFRECORD_NAME`,
  and `_TEST_TFRECORD_NAME`.

  Args:
    input_json: Path to a COCO JSON or JSONL file.
    output_dir: Directory to output the TFRecord files.
    split_ratio: List of how to split entries to train, validation, and test
      TFRecords.
    num_shard_list: List of the number of shards for each TFRecord file.
    output_shape: The desired output shape of the image. If None, the original
      image shape will be used.
  """
  # `coco` cannot access gcs uri. Use gcsfuse, it is faster.
  input_json = fileutils.force_gcs_fuse_path(input_json)
  coco_dataset = coco.COCO(input_json)

  label_map = {}
  label_id_by_category_id = {}
  for idx, category in enumerate(
      coco_dataset.dataset[constants.COCO_JSON_CATEGORIES], start=1
  ):
    label_map[idx] = category[constants.COCO_JSON_CATEGORY_NAME]
    label_id_by_category_id[category[constants.COCO_JSON_CATEGORY_ID]] = idx
  label_map_path = os.path.join(output_dir, common_lib.LABEL_MAP_NAME)
  logging.info('Writing label map to %s.', label_map_path)
  common_lib.write_label_map(label_map_path, label_map)

  with tf.io.gfile.GFile(
      os.path.join(output_dir, 'label_id_by_category_id.yaml'), 'w'
  ) as f:
    yaml.dump(label_id_by_category_id, f)

  ml_use_rows, image_rows, segmentation_rows = get_input_rows(
      coco_dataset, split_ratio, label_id_by_category_id
  )

  def pipeline(root):
    _define_data_conversion_pipeline(
        root,
        ml_use_rows,
        image_rows,
        segmentation_rows,
        output_dir,
        output_shape,
        num_shard_list,
    )

  logging.info('Beginning beam pipeline to acquire tfrecords.')
  common_lib.run_beam_pipeline(pipeline)
