"""Quantizes the model."""

import json
import os
from typing import Any, Dict, List, Sequence, Union

from absl import app
from absl import flags
from absl import logging
from auto_gptq import AutoGPTQForCausalLM
from auto_gptq import BaseQuantizeConfig
from awq import AutoAWQForCausalLM
from optimum.gptq.data import get_dataset
from transformers import AutoTokenizer

from util import dataset_validation_util
from vertex_vision_model_garden_peft.train.vmg import utils
from util import constants

_PRETRAINED_MODEL_ID = flags.DEFINE_string(
    'pretrained_model_id',
    None,
    'The pretrained model id. Supported models can be causal language modeling'
    ' models from https://github.com/huggingface/peft/tree/main. Note, there'
    ' might be different paddings for different models. This tool assumes the'
    ' pretrained_model_id contains model name, and then choose proper padding'
    ' methods. e.g. it must contain `llama` for `Llama2 models`.',
)

_QUANTIZATION_METHOD = flags.DEFINE_enum(
    'quantization_method',
    None,
    [constants.GPTQ, constants.AWQ],
    'The quantization method. Choose from ["gtpq", "awq"].',
)

_QUANTIZATION_PRECISION_MODE = flags.DEFINE_enum(
    'quantization_precision_mode',
    constants.PRECISION_MODE_4,
    [
        constants.PRECISION_MODE_8,
        constants.PRECISION_MODE_4,
        constants.PRECISION_MODE_3,
        constants.PRECISION_MODE_2,
    ],
    'Quantization precision mode.',
)

_QUANTIZATION_DATASET_NAME = flags.DEFINE_string(
    'quantization_dataset_name',
    None,
    'The dataset used for quantization. You can provide your own dataset in a'
    ' list of string or just use the original datasets used in GPTQ paper'
    ' ["wikitext2","c4","c4-new","ptb","ptb-new"] for GPTQ quantization. Using'
    " a dataset more appropriate to the model's training can improve"
    ' quantisation accuracy. Note that the GPTQ dataset is not the same as the'
    ' dataset used to train the model.',
)

_TEXT_COLUMN_IN_QUANTIZATION_DATASET = flags.DEFINE_string(
    'text_column_in_quantization_dataset',
    constants.DEFAULT_TEXT_COLUMN_IN_QUANTIZATION_DATASET,
    'The text column in quantization dataset.',
)

_QUANTIZATION_OUTPUT_DIR = flags.DEFINE_string(
    'quantization_output_dir',
    None,
    'The directory to store the quantized model.',
)

_QUANTIZATION_DEVICE_MAP = flags.DEFINE_string(
    'device_map', None, 'The device map.'
)

_QUANTIZATION_MAX_MEMORY = flags.DEFINE_string(
    'max_memory', None, 'The maximum memory.'
)

_GROUP_SIZE = flags.DEFINE_integer(
    'group_size',
    None,
    'The group size to use for quantization. Recommended value is 128 and -1'
    ' uses per-column quantization. Higher numbers use less VRAM, but have'
    ' lower quantisation accuracy. "None" is the lowest possible value.',
)

_DESC_ACT = flags.DEFINE_boolean(
    'desc_act',
    False,
    'Whether to quantize columns in order of decreasing activation size.'
    ' Setting it to False can significantly speed up inference but the'
    ' perplexity may become slightly worse. Also known as act-order.',
)

_DAMP_PERCENT = flags.DEFINE_float(
    'damp_percent',
    0.1,
    'The percent of the average Hessian diagonal to use for dampening.',
)

_CACHE_EXAMPLES_ON_GPU = flags.DEFINE_boolean(
    'cache_examples_on_gpu',
    True,
    'Whether to cache the examples on GPU. Disabling will reduce VRAM usage,'
    ' but increase quantization time.',
)

_AWQ_VERSION = flags.DEFINE_enum(
    'awq_version',
    constants.GEMM,
    [constants.GEMM, constants.GEMV],
    'The version of the AWQ to use. It determines how matrix multiplication'
    ' runs under the hood. GEMV is 20% faster than GEMM, only at batch size 1'
    ' (not good for large contexts). GEMM is much faster than FP16 at batch'
    ' sizes below 8 (good with large contexts).',
)


@flags.multi_flags_validator(
    [
        _PRETRAINED_MODEL_ID.name,
        _QUANTIZATION_METHOD.name,
        _QUANTIZATION_PRECISION_MODE.name,
        _QUANTIZATION_DATASET_NAME.name,
        _QUANTIZATION_OUTPUT_DIR.name,
    ],
)
def check_quantization_flags(flags_dict: Dict[str, Any]) -> bool:
  """Check if required flags are set on quantization task.

  Args:
    flags_dict: Dictionary containing task and flags to check.

  Returns:
    If required flags are not None.
  """
  required_flags = [
      _QUANTIZATION_METHOD.name,
      _PRETRAINED_MODEL_ID.name,
      _QUANTIZATION_PRECISION_MODE.name,
      _QUANTIZATION_DATASET_NAME.name,
      _QUANTIZATION_OUTPUT_DIR.name,
  ]

  return all(map(lambda x: flags_dict[x] is not None, required_flags))


def quantize_model(
    quantization_method: str,
    pretrained_model_id: str,
    quantization_output_dir: str,
    quantization_precision_mode: str = None,
    quantization_dataset_name: Union[List[str]] = None,
    text_column_in_quantization_dataset: str = constants.DEFAULT_TEXT_COLUMN_IN_QUANTIZATION_DATASET,
    group_size: int = None,
    desc_act: bool = True,
    damp_percent: float = 0.1,
    awq_version: str = 'GEMM',
    device_map: str = None,
    max_memory: Dict[Any, str] = None,
    cache_examples_on_gpu: bool = True,
) -> None:
  """Quantizes the model using `quantization_method`."""
  if quantization_method == constants.GPTQ:
    gptq_quantize_model(
        pretrained_model_id=pretrained_model_id,
        gptq_output_dir=quantization_output_dir,
        gptq_precision_mode=quantization_precision_mode,
        gptq_dataset_name=quantization_dataset_name,
        group_size=group_size,
        desc_act=desc_act,
        damp_percent=damp_percent,
        cache_examples_on_gpu=cache_examples_on_gpu,
    )
  elif quantization_method == constants.AWQ:
    awq_quantize_model(
        pretrained_model_id=pretrained_model_id,
        quantization_output_dir=quantization_output_dir,
        quantization_precision_mode=quantization_precision_mode,
        quantization_dataset_name=quantization_dataset_name,
        text_column_in_quantization_dataset=text_column_in_quantization_dataset,
        group_size=group_size,
        awq_version=awq_version,
        device_map=device_map,
        max_memory=max_memory,
    )


def awq_quantize_model(
    pretrained_model_id: str,
    quantization_output_dir: str,
    quantization_precision_mode: str = None,
    quantization_dataset_name: Union[List[str]] = None,
    text_column_in_quantization_dataset: str = constants.DEFAULT_TEXT_COLUMN_IN_QUANTIZATION_DATASET,
    group_size: int = None,
    awq_version: str = 'GEMM',
    device_map: str = None,
    max_memory: Dict[Any, str] = None,
) -> None:
  """Quantizes the model using AWQ."""
  if quantization_precision_mode != constants.PRECISION_MODE_4:
    raise ValueError(
        f'Invalid precision mode: {quantization_precision_mode} for AWQ. 4bit'
        ' quantization must be used.'
    )
  else:
    bits = 4
  if not group_size:
    group_size = 128
  if not device_map:
    device_map = 'cpu'
  if dataset_validation_util.is_gcs_path(quantization_dataset_name):
    logging.info('Using custom dataset: %s', quantization_dataset_name)
    with open(
        dataset_validation_util.force_gcs_fuse_path(quantization_dataset_name),
        'r',
    ) as f:
      quantization_dataset = [line.rstrip('\n') for line in f]
  else:
    quantization_dataset = quantization_dataset_name
  quant_config = {
      'zero_point': True,
      'q_group_size': group_size,
      'w_bit': bits,
      'version': awq_version,
  }
  logging.info('Quantization config: %s', quant_config)
  model = AutoAWQForCausalLM.from_pretrained(
      pretrained_model_id,
      trust_remote_code=True,
      device_map=device_map,
      max_memory=max_memory,
      low_cpu_mem_usage=True,
  )
  tokenizer = AutoTokenizer.from_pretrained(
      pretrained_model_id, trust_remote_code=True
  )
  model.quantize(
      tokenizer,
      quant_config=quant_config,
      calib_data=quantization_dataset,
      text_column=text_column_in_quantization_dataset,
  )
  model.save_quantized(quantization_output_dir)
  tokenizer.save_pretrained(quantization_output_dir)


def gptq_quantize_model(
    pretrained_model_id: str,
    gptq_output_dir: str,
    gptq_precision_mode: str = None,
    gptq_dataset_name: Union[List[str]] = None,
    group_size: int = -1,
    desc_act: bool = False,
    damp_percent: float = 0.1,
    cache_examples_on_gpu: bool = True,
) -> None:
  """Quantizes the model using GPTQ."""
  logging.info(
      'PYTORCH_CUDA_ALLOC_CONF: %s',
      os.environ.get('PYTORCH_CUDA_ALLOC_CONF', ''),
  )
  if dataset_validation_util.is_gcs_path(gptq_dataset_name):
    logging.info('Using custom dataset: %s', gptq_dataset_name)
    with open(
        dataset_validation_util.force_gcs_fuse_path(gptq_dataset_name), 'r'
    ) as f:
      gptq_dataset = [line.rstrip('\n') for line in f]
  else:
    gptq_dataset = gptq_dataset_name
  if gptq_precision_mode == constants.PRECISION_MODE_8:
    bits = 8
  elif gptq_precision_mode == constants.PRECISION_MODE_4:
    bits = 4
  elif gptq_precision_mode == constants.PRECISION_MODE_3:
    bits = 3
  elif gptq_precision_mode == constants.PRECISION_MODE_2:
    bits = 2
  else:
    raise ValueError(f'Invalid precision mode: {gptq_precision_mode} for GPTQ.')
  if not group_size:
    group_size = -1

  tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id)
  gptq_dataset = get_dataset(gptq_dataset, tokenizer)

  quantization_config = BaseQuantizeConfig(
      bits=bits,
      group_size=group_size,
      damp_percent=damp_percent,
      desc_act=desc_act,
  )

  logging.info('Quantization config: %s', quantization_config.to_dict())

  model = AutoGPTQForCausalLM.from_pretrained(
      pretrained_model_id,
      quantization_config,
      low_cpu_mem_usage=True,
      torch_dtype='auto',
      trust_remote_code=True,
  )
  model.quantize(
      examples=gptq_dataset,
      cache_examples_on_gpu=cache_examples_on_gpu,
  )

  if utils.should_add_pad_token(pretrained_model_id):
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
  model.save_pretrained(gptq_output_dir)
  tokenizer.save_pretrained(gptq_output_dir)


def main(unused_argv: Sequence[str]) -> None:
  pretrained_model_id = _PRETRAINED_MODEL_ID.value
  if dataset_validation_util.is_gcs_path(pretrained_model_id):
    pretrained_model_id = dataset_validation_util.download_gcs_uri_to_local(
        pretrained_model_id
    )
  pretrained_model_id = dataset_validation_util.force_gcs_fuse_path(
      pretrained_model_id
  )

  if _QUANTIZATION_MAX_MEMORY.value:
    max_memory = json.loads(_QUANTIZATION_MAX_MEMORY.value)
  else:
    max_memory = None

  quantize_model(
      quantization_method=_QUANTIZATION_METHOD.value,
      pretrained_model_id=pretrained_model_id,
      quantization_output_dir=_QUANTIZATION_OUTPUT_DIR.value,
      quantization_precision_mode=_QUANTIZATION_PRECISION_MODE.value,
      quantization_dataset_name=_QUANTIZATION_DATASET_NAME.value,
      text_column_in_quantization_dataset=_TEXT_COLUMN_IN_QUANTIZATION_DATASET.value,
      group_size=_GROUP_SIZE.value,
      desc_act=_DESC_ACT.value,
      damp_percent=_DAMP_PERCENT.value,
      awq_version=_AWQ_VERSION.value,
      device_map=_QUANTIZATION_DEVICE_MAP.value,
      max_memory=max_memory,
      cache_examples_on_gpu=_CACHE_EXAMPLES_ON_GPU.value,
  )


if __name__ == '__main__':
  app.run(main)
