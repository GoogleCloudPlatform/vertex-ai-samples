"""Script to merge PEFT adapter with base model."""

from collections.abc import Mapping, Sequence
from typing import Any

from absl import app
from absl import flags

from util import dataset_validation_util
from vertex_vision_model_garden_peft.train.vmg import utils
from util import constants
from util import fileutils


_PRETRAINED_MODEL_NAME_OR_PATH = flags.DEFINE_string(
    'pretrained_model_name_or_path',
    None,
    'The pretrained model id. Supported models can be causal language modeling'
    ' models from https://github.com/huggingface/peft/tree/main. Note, there'
    ' might be different paddings for different models. This tool assumes the'
    ' pretrained_model_name_or_path contains model name, and then choose proper'
    ' padding methods. e.g. it must contain `llama` for `Llama2 models`.',
    required=True,
)

_MERGE_BASE_AND_LORA_OUTPUT_DIR = flags.DEFINE_string(
    'merge_base_and_lora_output_dir',
    None,
    'The directory to store the merged model with the base and lora adapter.',
)

_MERGE_MODEL_PRECISION_MODE = flags.DEFINE_enum(
    'merge_model_precision_mode',
    constants.PRECISION_MODE_16B,
    [
        constants.PRECISION_MODE_4,
        constants.PRECISION_MODE_8,
        constants.PRECISION_MODE_FP8,
        constants.PRECISION_MODE_16,
        constants.PRECISION_MODE_16B,
        constants.PRECISION_MODE_32,
    ],
    'Merging model precision mode.',
)

_FINETUNED_LORA_MODEL_DIR = flags.DEFINE_string(
    'finetuned_lora_model_dir',
    None,
    'The directory storing finetuned LoRA model weights.',
)

_HUGGINGFACE_ACCESS_TOKEN = flags.DEFINE_string(
    'huggingface_access_token',
    None,
    'The access token for loading huggingface gated models.',
)


@flags.multi_flags_validator(
    [
        _PRETRAINED_MODEL_NAME_OR_PATH.name,
        _FINETUNED_LORA_MODEL_DIR.name,
        _MERGE_BASE_AND_LORA_OUTPUT_DIR.name,
    ],
)
def check_merge_lora_model_flags(flags_dict: Mapping[str, Any]) -> bool:
  """Check if required flags are set on merge model LoRA task.

  Args:
    flags_dict: Dictionary containing task and flags to check.

  Returns:
    If required flags are not None.
  """
  return all(map(lambda x: x is not None, flags_dict.values()))


def main(unused_argv: Sequence[str]) -> None:
  pretrained_model_name_or_path = fileutils.force_gcs_path(
      _PRETRAINED_MODEL_NAME_OR_PATH.value
  )
  if dataset_validation_util.is_gcs_path(pretrained_model_name_or_path):
    pretrained_model_name_or_path = (
        dataset_validation_util.download_gcs_uri_to_local(
            pretrained_model_name_or_path
        )
    )

  finetuned_lora_model_dir = fileutils.force_gcs_path(
      _FINETUNED_LORA_MODEL_DIR.value
  )
  if dataset_validation_util.is_gcs_path(finetuned_lora_model_dir):
    finetuned_lora_model_dir = (
        dataset_validation_util.download_gcs_uri_to_local(
            finetuned_lora_model_dir
        )
    )
  utils.merge_causal_language_model_with_lora(
      pretrained_model_name_or_path=pretrained_model_name_or_path,
      precision_mode=_MERGE_MODEL_PRECISION_MODE.value,
      finetuned_lora_model_dir=finetuned_lora_model_dir,
      merged_model_output_dir=_MERGE_BASE_AND_LORA_OUTPUT_DIR.value,
      access_token=_HUGGINGFACE_ACCESS_TOKEN.value,
  )


if __name__ == '__main__':
  app.run(main)
