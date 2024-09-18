r"""Converts ImageNet label texts as yamls for model garden playground.

# ImageNet1K will have label maps with background.
"""

import urllib.request
from absl import app
from absl import flags
import tensorflow as tf
import yaml


_INPUT_TEXT_FILEPATH = flags.DEFINE_string(
    'input_text_filepath',
    None,
    'The input text filepath.',
    required=True,
)

_ADD_BACKGROUND_LABEL = flags.DEFINE_boolean(
    'add_background_label',
    None,
    'Whether or not add background labels.',
    required=True,
)

_ADD_IDS = flags.DEFINE_boolean(
    'add_ids',
    None,
    'Whether or not add ids.',
    required=True,
)

_OUTPUT_YAML_FILEPATH = flags.DEFINE_string(
    'output_yaml_filepath',
    None,
    'The output yaml filepath.',
    required=True,
)


def convert_imagenet_label_map_from_text_to_yaml(
    input_text_filepath: str,
    add_background_label: bool,
    add_ids: bool,
    output_yaml_filepath: str,
) -> None:
  """Converts imagenet label map from text to yamls."""
  label_map = {}

  # Shifts all keys by 1, and add 0 as 'background'.
  if add_background_label:
    label_map = yaml.safe_load(
        urllib.request.urlopen(input_text_filepath).read()
    )
    new_label_map = {}
    for key, value in label_map.items():
      new_label_map[key + 1] = value
    new_label_map[0] = 'background'
    label_map = new_label_map

  # Adds maps from id to each line.
  if add_ids:
    lines = urllib.request.urlopen(input_text_filepath).readlines()
    current_id = 0
    for line in lines:
      label_map[current_id] = line.decode('ascii').strip()
      print(label_map[current_id])
      current_id += 1

  # Saves new label maps as yamls.
  with tf.io.gfile.GFile(output_yaml_filepath, 'w') as writer:
    writer.write(yaml.dump(label_map))


def main(_) -> None:
  convert_imagenet_label_map_from_text_to_yaml(
      _INPUT_TEXT_FILEPATH.value,
      _ADD_BACKGROUND_LABEL.value,
      _ADD_IDS.value,
      _OUTPUT_YAML_FILEPATH.value,
  )


if __name__ == '__main__':
  app.run(main)
