"""Main function to start PEFT finetuning."""
import subprocess

from absl import app
from absl import flags
from absl import logging

from peft import causal_language_modeling_lora
from peft import instruct_lora
from peft import sequence_classification_lora
from util import constants
from util import fileutils

_TASK = flags.DEFINE_string(
    'task',
    constants.CAUSAL_LANGUAGE_MODELING_LORA,
    'The supported PEFT tasks.',
)

_PRETRAINED_MODEL_ID = flags.DEFINE_string(
    'pretrained_model_id',
    None,
    'The pretrained model id. Supported models can be causal language modeling'
    ' models from https://github.com/huggingface/peft/tree/main.',
    required=True,
)

_DATASET_NAME = flags.DEFINE_string(
    'dataset_name',
    None,
    'The dataset name in huggingface.',
    required=True,
)

_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    None,
    'The output directory.',
    required=True,
)

_PRECISION_MODE = flags.DEFINE_string(
    'precision_mode',
    constants.PRECISION_MODE_16,
    'Supported finetuning precision_modes are `{}` and `{}`.'.format(
        constants.PRECISION_MODE_8, constants.PRECISION_MODE_16
    ),
)

_LORA_RANK = flags.DEFINE_integer(
    'lora_rank',
    16,
    'The rank of the update matrices, expressed in int. Lower rank results in'
    ' smaller update matrices with fewer trainable parameters, referring to'
    ' https://huggingface.co/docs/peft/conceptual_guides/lora.',
)

_LORA_ALPHA = flags.DEFINE_integer(
    'lora_alpha',
    32,
    'LoRA scaling factor, referring to'
    ' https://huggingface.co/docs/peft/conceptual_guides/lora.',
)

_LORA_DROPOUT = flags.DEFINE_float(
    'lora_dropout',
    0.05,
    'dropout probability of the LoRA layers, referring to'
    ' https://huggingface.co/docs/peft/task_guides/token-classification-lora.',
)

_WARMUP_STEPS = flags.DEFINE_integer(
    'warmup_steps',
    10,
    'Number of steps for the warmup in the learning rate scheduler.',
)

_WARMUP_RATIO = flags.DEFINE_float(
    'warmup_ratio',
    0.03,
    'The warmup ratio in the learning rate scheduler.',
)

_MAX_STEPS = flags.DEFINE_integer(
    'max_steps',
    10,
    'Total number of training steps.',
)

_MAX_SEQ_LENGTH = flags.DEFINE_integer(
    'max_seq_length',
    512,
    'The maximum sequence length.',
)

_NUM_EPOCHS = flags.DEFINE_integer(
    'num_epochs',
    20,
    'The number of training epochs.',
)

_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size',
    32,
    'The batch size.',
)

_LEARNING_RATE = flags.DEFINE_float(
    'learning_rate',
    2e-4,
    'The learning rate after the potential warmup period.',
)


def main(_) -> None:
  task = _TASK.value
  pretrained_model_id = _PRETRAINED_MODEL_ID.value
  local_pretrained_model_id = None
  if pretrained_model_id.startswith(constants.GCS_URI_PREFIX):
    logging.info(
        'Start to copy pretrained models locally: %s.', pretrained_model_id
    )
    fileutils.download_gcs_dir_to_local(
        pretrained_model_id, constants.LOCAL_BASE_MODEL_DIR
    )
    local_pretrained_model_id = constants.LOCAL_BASE_MODEL_DIR
    logging.info(
        'Finished copying pretrained models locally to: %s.',
        local_pretrained_model_id,
    )
  if task == constants.TEXT_TO_IMAGE_LORA:
    subprocess.run(['/bin/bash', 'train.sh'], check=True)
  elif task == constants.SEQUENCE_CLASSIFICATION_LORA:
    sequence_classification_lora.finetune_sequence_classification(
        pretrained_model_id=pretrained_model_id,
        dataset_name=_DATASET_NAME.value,
        output_dir=_OUTPUT_DIR.value,
        lora_rank=_LORA_RANK.value,
        lora_alpha=_LORA_ALPHA.value,
        lora_dropout=_LORA_DROPOUT.value,
        num_epochs=_NUM_EPOCHS.value,
        batch_size=_BATCH_SIZE.value,
        learning_rate=_LEARNING_RATE.value,
    )
  elif task == constants.CAUSAL_LANGUAGE_MODELING_LORA:
    causal_language_modeling_lora.finetune_causal_language_modeling(
        pretrained_model_id=pretrained_model_id,
        dataset_name=_DATASET_NAME.value,
        output_dir=_OUTPUT_DIR.value,
        precision_mode=_PRECISION_MODE.value,
        lora_rank=_LORA_RANK.value,
        lora_alpha=_LORA_ALPHA.value,
        lora_dropout=_LORA_DROPOUT.value,
        warmup_steps=_WARMUP_STEPS.value,
        max_steps=_MAX_STEPS.value,
        learning_rate=_LEARNING_RATE.value,
        local_pretrained_model_id=local_pretrained_model_id,
    )
  elif task == constants.INSTRUCT_LORA:
    instruct_lora.finetune_instruct(
        pretrained_model_id=pretrained_model_id,
        dataset_name=_DATASET_NAME.value,
        output_dir=_OUTPUT_DIR.value,
        lora_rank=_LORA_RANK.value,
        lora_alpha=_LORA_ALPHA.value,
        lora_dropout=_LORA_DROPOUT.value,
        warmup_ratio=_WARMUP_RATIO.value,
        max_steps=_MAX_STEPS.value,
        max_seq_length=_MAX_SEQ_LENGTH.value,
        learning_rate=_LEARNING_RATE.value,
    )
  else:
    raise ValueError('The task {} is not supported.'.format(task))


if __name__ == '__main__':
  app.run(main)
