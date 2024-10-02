"""Script to merge PEFT adapter with base model."""

from typing import Any, Dict, Sequence

from absl import app
from absl import flags

from util import dataset_validation_util
from vertex_vision_model_garden_peft.train.vmg import utils
from util import constants
from util import fileutils


_PRETRAINED_MODEL_ID = flags.DEFINE_string(
    'pretrained_model_id',
    None,
    'The pretrained model id. Supported models can be causal language modeling'
    ' models from https://github.com/huggingface/peft/tree/main. Note, there'
    ' might be different paddings for different models. This tool assumes the'
    ' pretrained_model_id contains model name, and then choose proper padding'
    ' methods. e.g. it must contain `llama` for `Llama2 models`.',
    required=True,
)

_MERGE_BASE_AND_LORA_OUTPUT_DIR = flags.DEFINE_string(
    'merge_base_and_lora_output_dir',
    None,
    'The directory to store the merged model with the base and lora adapter.',
)

_MERGE_MODEL_PRECISION_MODE = flags.DEFINE_enum(
    'merge_model_precision_mode',
    constants.PRECISION_MODE_16,
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

_RESTRICT_MODEL_UPLOAD_DOCKER_URI = flags.DEFINE_string(
    'restrict_model_upload_docker_uri',
    '',
    'If set, mark output model as only uploadable to Model Registry with the'
    ' specified Docker URI.',
)

_EXECUTOR_INPUT = flags.DEFINE_string(
    'executor_input',
    '',
    'For internal use. Kubeflow pipeline context when running trainer as part'
    ' of an internal pipeline.',
)

_HUGGINGFACE_ACCESS_TOKEN = flags.DEFINE_string(
    'huggingface_access_token',
    None,
    'The access token for loading huggingface gated models.',
)


@flags.multi_flags_validator(
    [
        _PRETRAINED_MODEL_ID.name,
        _FINETUNED_LORA_MODEL_DIR.name,
        _MERGE_BASE_AND_LORA_OUTPUT_DIR.name,
    ],
)
def check_merge_lora_model_flags(flags_dict: Dict[str, Any]) -> bool:
  """Check if required flags are set on merge model LoRA task.

  Args:
    flags_dict: Dictionary containing task and flags to check.

  Returns:
    If required flags are not None.
  """
  return all(map(lambda x: x is not None, flags_dict.values()))


def main(unused_argv: Sequence[str]) -> None:
  pretrained_model_id = fileutils.force_gcs_path(_PRETRAINED_MODEL_ID.value)
  if dataset_validation_util.is_gcs_path(pretrained_model_id):
    pretrained_model_id = dataset_validation_util.download_gcs_uri_to_local(
        pretrained_model_id
    )

  finetuned_lora_model_dir = utils.GcsOrLocalDirectory(
      _FINETUNED_LORA_MODEL_DIR.value
  )

  merge_base_and_lora_output_dir = utils.GcsOrLocalDirectory(
      _MERGE_BASE_AND_LORA_OUTPUT_DIR.value
  )

  utils.merge_causal_language_model_with_lora(
      pretrained_model_id=pretrained_model_id,
      precision_mode=_MERGE_MODEL_PRECISION_MODE.value,
      finetuned_lora_model_dir=finetuned_lora_model_dir.local_dir,
      merged_model_output_dir=merge_base_and_lora_output_dir.local_dir,
      access_token=_HUGGINGFACE_ACCESS_TOKEN.value,
  )

  if _RESTRICT_MODEL_UPLOAD_DOCKER_URI.value:
    utils.write_first_party_model_metadata(
        merge_base_and_lora_output_dir.local_dir,
        _RESTRICT_MODEL_UPLOAD_DOCKER_URI.value,
    )

  if _EXECUTOR_INPUT.value:
    utils.write_kfp_outputs(
        _EXECUTOR_INPUT.value,
        {
            'saved_model': _MERGE_BASE_AND_LORA_OUTPUT_DIR.value,
        },
    )

  merge_base_and_lora_output_dir.upload_to_gcs(skip_if_exists=True)


if __name__ == '__main__':
  app.run(main)
