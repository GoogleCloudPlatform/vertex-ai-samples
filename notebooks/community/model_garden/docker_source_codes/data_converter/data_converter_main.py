r"""Python script to convert user input data to training docker format.


Note: the training format is designed to be tfrecord as in the design doc.
If there are training efficiency issues for pytorch algorithms, we will also
support pytorch formats as well.
"""

from absl import app
from absl import flags
from absl import logging
from data_converter import common_lib
from data_converter import data_converter_icn_lib
from data_converter import data_converter_iod_lib
from data_converter import data_converter_isg_lib
from data_converter import data_converter_vcn_lib
from util import constants


_INPUT_FILE_PATH = flags.DEFINE_string(
    'input_file_path',
    None,
    'Input file path.',
    required=True,
)
_INPUT_FILE_TYPE = flags.DEFINE_enum(
    'input_file_type',
    None,
    [
        constants.INPUT_FILE_TYPE_CSV,
        constants.INPUT_FILE_TYPE_JSONL,
        constants.INPUT_FILE_TYPE_COCO_JSON,
    ],
    'Input file type.',
    required=True,
)
_OBJECTIVE = flags.DEFINE_enum(
    'objective',
    None,
    [
        constants.OBJECTIVE_IMAGE_CLASSIFICATION,
        constants.OBJECTIVE_IMAGE_OBJECT_DETECTION,
        constants.OBJECTIVE_IMAGE_SEGMENTATION,
        constants.OBJECTIVE_VIDEO_CLASSIFICATION,
    ],
    'The objective of this training job.',
    required=True,
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    None,
    'The output directory for converted data and label map files.',
    required=True,
)
_SPLIT_RATIO = flags.DEFINE_list(
    'split_ratio',
    '0.8,0.1,0.1',
    'Proportion of data to split into train/validation/test.',
)
_NUM_SHARD = flags.DEFINE_list(
    'num_shard', '10,10,10', 'The number of shards for train/validation/test.'
)
_OUTPUT_FPS = flags.DEFINE_integer(
    'output_fps', 5, 'For videos only. The output frames rate per second.'
)


def main(_) -> None:
  logging.info(
      (
          'Start data converter on: %s (type: %s) with split: %s for %s'
          ' (shard=%s), and output to %s.'
      ),
      _INPUT_FILE_PATH.value,
      _INPUT_FILE_TYPE.value,
      _SPLIT_RATIO.value,
      _OBJECTIVE.value,
      _NUM_SHARD.value,
      _OUTPUT_DIR.value,
  )
  split_ratio = list(map(float, _SPLIT_RATIO.value))
  num_shard = list(map(int, _NUM_SHARD.value))
  common_lib.check_split_ratio(split_ratio)
  common_lib.check_num_shard(num_shard)
  if (
      _OBJECTIVE.value == constants.OBJECTIVE_IMAGE_OBJECT_DETECTION
      and _INPUT_FILE_TYPE.value == constants.INPUT_FILE_TYPE_CSV
  ):
    data_converter_iod_lib.convert_csv_to_tfrecord(
        _INPUT_FILE_PATH.value,
        _OUTPUT_DIR.value,
        split_ratio,
        num_shard,
    )
  elif (
      _OBJECTIVE.value == constants.OBJECTIVE_IMAGE_OBJECT_DETECTION
      and _INPUT_FILE_TYPE.value == constants.INPUT_FILE_TYPE_JSONL
  ):
    data_converter_iod_lib.convert_jsonl_to_tfrecord(
        _INPUT_FILE_PATH.value, _OUTPUT_DIR.value, split_ratio, num_shard
    )
  elif (
      _OBJECTIVE.value == constants.OBJECTIVE_IMAGE_OBJECT_DETECTION
      and _INPUT_FILE_TYPE.value == constants.INPUT_FILE_TYPE_COCO_JSON
  ):
    data_converter_iod_lib.convert_coco_json_to_tfrecord(
        _INPUT_FILE_PATH.value,
        _OUTPUT_DIR.value,
        split_ratio,
        num_shard,
    )
  elif _OBJECTIVE.value == constants.OBJECTIVE_IMAGE_SEGMENTATION:
    data_converter_isg_lib.beam_build_tfrecord_from_coco_json(
        _INPUT_FILE_PATH.value,
        _OUTPUT_DIR.value,
        split_ratio,
        num_shard,
    )
  elif (
      _OBJECTIVE.value == constants.OBJECTIVE_IMAGE_CLASSIFICATION
      and _INPUT_FILE_TYPE.value == constants.INPUT_FILE_TYPE_CSV
  ):
    data_converter_icn_lib.convert_csv_to_tfrecord(
        _INPUT_FILE_PATH.value,
        _OUTPUT_DIR.value,
        split_ratio,
        num_shard,
    )
  elif (
      _OBJECTIVE.value == constants.OBJECTIVE_IMAGE_CLASSIFICATION
      and _INPUT_FILE_TYPE.value == constants.INPUT_FILE_TYPE_JSONL
  ):
    data_converter_icn_lib.convert_jsonl_to_tfrecord(
        _INPUT_FILE_PATH.value, _OUTPUT_DIR.value, split_ratio, num_shard
    )
  elif (
      _OBJECTIVE.value == constants.OBJECTIVE_VIDEO_CLASSIFICATION
      and _INPUT_FILE_TYPE.value == constants.INPUT_FILE_TYPE_CSV
  ):
    data_converter_vcn_lib.convert_csv_to_tfrecord(
        _INPUT_FILE_PATH.value,
        _OUTPUT_DIR.value,
        _OUTPUT_FPS.value,
        split_ratio,
        num_shard,
    )
  elif (
      _OBJECTIVE.value == constants.OBJECTIVE_VIDEO_CLASSIFICATION
      and _INPUT_FILE_TYPE.value == constants.INPUT_FILE_TYPE_JSONL
  ):
    data_converter_vcn_lib.convert_jsonl_to_tfrecord(
        _INPUT_FILE_PATH.value,
        _OUTPUT_DIR.value,
        _OUTPUT_FPS.value,
        split_ratio,
        num_shard,
    )
  else:
    raise NotImplementedError(
        f'File format {_INPUT_FILE_TYPE.value} is not supported for'
        f' {_OBJECTIVE.value}.'
    )


if __name__ == '__main__':
  app.run(main)
