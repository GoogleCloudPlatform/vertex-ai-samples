r"""Converts COCO labels as yamls for model garden playground (IOD).
"""

import os
import urllib.request

from absl import app
from absl import flags
import tensorflow as tf
import yaml

from object_detection.utils import label_map_util

_CONVERT_LABEL_TYPE_COCO_80 = 'coco_80'
_CONVERT_LABEL_TYPE_COCO_91 = 'coco_91'

_CONVERT_LABEL_TYPE = flags.DEFINE_enum(
    'convert_label_type',
    None,
    [
        _CONVERT_LABEL_TYPE_COCO_80,
        _CONVERT_LABEL_TYPE_COCO_91,
    ],
    'Different types of label type conversion.',
    required=True,
)

_TEMPORARY_PATH = flags.DEFINE_string(
    'temporary_path',
    None,
    'The tempory path.',
    required=True,
)

_OUTPUT_YAML_FILEPATH = flags.DEFINE_string(
    'output_yaml_filepath',
    None,
    'The output yaml filepath.',
    required=True,
)


def convert_coco_label_map_91(
    output_yaml_filepath: str,
) -> None:
  """Converts coco label map 91."""
  input_proto_filepath = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt'
  local_input_proto_filepath = os.path.join(
      _TEMPORARY_PATH.value, 'mscoco_label_map.pbtxt'
  )
  with open(local_input_proto_filepath, 'w') as writer:
    contents = (
        urllib.request.urlopen(input_proto_filepath).read().decode('utf-8')
    )
    writer.write(contents)

  label_map = label_map_util.load_labelmap(local_input_proto_filepath)
  label_map_dict = label_map_util.get_label_map_dict(
      label_map, use_display_name=True
  )
  swapped_label_map_dict = {v: k for k, v in label_map_dict.items()}
  print(swapped_label_map_dict)

  # Saves new label maps as yamls.
  with tf.io.gfile.GFile(output_yaml_filepath, 'w') as writer:
    writer.write(yaml.dump(swapped_label_map_dict))


def convert_coco_label_map_80(
    output_yaml_filepath: str,
) -> None:
  """Converts coco label map 80."""
  # Loads label maps from texts.
  input_text_filepath = 'https://gist.githubusercontent.com/AruniRC/7b3dadd004da04c80198557db5da4bda/raw/2f10965ace1e36c4a9dca76ead19b744f5eb7e88/ms_coco_classnames.txt'
  local_input_text_filepath = os.path.join(
      _TEMPORARY_PATH.value, 'ms_coco_classnames.txt'
  )
  with open(local_input_text_filepath, 'w') as writer:
    contents = (
        urllib.request.urlopen(input_text_filepath).read().decode('utf-8')
    )
    writer.write(contents)
  with open(local_input_text_filepath, 'r') as file:
    content = file.read()
    label_map = yaml.safe_load(content)

  # Removes background in label maps.
  new_label_map = {}
  for k, v in label_map.items():
    if k == 0:
      continue
    new_label_map[k - 1] = v
  print(new_label_map)
  # Saves new label maps as yamls.
  with tf.io.gfile.GFile(output_yaml_filepath, 'w') as writer:
    writer.write(yaml.dump(new_label_map))


def main(_) -> None:
  if _CONVERT_LABEL_TYPE.value == _CONVERT_LABEL_TYPE_COCO_80:
    convert_coco_label_map_80(_OUTPUT_YAML_FILEPATH.value)
  elif _CONVERT_LABEL_TYPE.value == _CONVERT_LABEL_TYPE_COCO_91:
    convert_coco_label_map_91(
        _OUTPUT_YAML_FILEPATH.value,
    )
  else:
    print('Not supported convert label type: ', _CONVERT_LABEL_TYPE.value)


if __name__ == '__main__':
  app.run(main)
