"""Instruct/Chat with LoRA models."""

import dataclasses
import datetime
import json
import os
from typing import Any, Dict, Optional, Sequence
import warnings

from absl import app
from absl import flags
from absl import logging
from accelerate import DistributedType
from accelerate import PartialState
import bitsandbytes as bnb
import hypertune
from peft import get_peft_model
from peft import LoraConfig
import torch
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from trl import DataCollatorForCompletionOnlyLM
from trl import SFTTrainer
import wandb

from util import dataset_validation_util
from vertex_vision_model_garden_peft.train.vmg import callbacks
from vertex_vision_model_garden_peft.train.vmg import eval_lib
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

_HUGGINGFACE_ACCESS_TOKEN = flags.DEFINE_string(
    'huggingface_access_token',
    None,
    'The access token for loading huggingface gated models.',
)

_DATASET_NAME = flags.DEFINE_string(
    'dataset_name',
    None,
    'The dataset name in huggingface.',
)

_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    None,
    'The output directory.',
)

_LOGGING_OUTPUT_DIR = flags.DEFINE_string(
    'logging_output_dir',
    '',
    'The logging output directory, which defaults to same as output_dir.',
)

_PRECISION_MODE = flags.DEFINE_enum(
    'precision_mode',
    constants.PRECISION_MODE_16,
    [
        constants.PRECISION_MODE_4,
        constants.PRECISION_MODE_8,
        constants.PRECISION_MODE_16,
        constants.PRECISION_MODE_16B,
        constants.PRECISION_MODE_32,
    ],
    'Precision to load model weights for finetuning.',
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

_WEIGHT_DECAY = flags.DEFINE_float(
    'weight_decay',
    0.001,
    'The weight decay in the learning rate scheduler.',
)

_NUM_EPOCHS = flags.DEFINE_float(
    'num_epochs',
    None,
    'The number of training epochs. Only used for'
    ' "sequence-classification-lora" with an integer value and for'
    ' "instruct-lora" with a float value allowed.',
)

_MAX_STEPS = flags.DEFINE_integer(
    'max_steps',
    None,
    'Total number of training steps. Overrides num_epochs if set. Only used for'
    ' "instruct-lora."',
)

_MAX_SEQ_LENGTH = flags.DEFINE_integer(
    'max_seq_length',
    512,
    'The maximum sequence length.',
)

_LEARNING_RATE = flags.DEFINE_float(
    'learning_rate',
    2e-4,
    'The learning rate after the potential warmup period.',
)

_INSTRUCT_COLUMN_IN_DATASET = flags.DEFINE_string(
    'instruct_column_in_dataset',
    constants.DEFAULT_INSTRUCT_COLUMN_IN_DATASET,
    'The instruct column in dataset.',
)

_REPORT_TO = flags.DEFINE_string(
    'report_to',
    constants.REPORT_TO_NONE,
    'Where logging is reported to, which can be tensorboard or none.',
)

_PER_DEVICE_TRAIN_BATCH_SIZE = flags.DEFINE_integer(
    'per_device_train_batch_size',
    4,
    'The per device train batch size.',
)

_GRADIENT_ACCUMULATION_STEPS = flags.DEFINE_integer(
    'gradient_accumulation_steps',
    4,
    'The gradient accumulation steps.',
)

_ENABLE_GRADIENT_CHECKPOINTING = flags.DEFINE_boolean(
    'enable_gradient_checkpointing',
    False,
    'Whether to enable gradient checkpointing.',
)

_ENABLE_PEFT = flags.DEFINE_boolean(
    'enable_peft',
    True,
    'Whether to enable peft.',
)
_TEMPLATE = flags.DEFINE_string(
    'template',
    None,
    'Template for formatting language model training data. Must be a filename'
    ' under `templates` folder, without `.json` extension, e.g. `alpaca`, or a'
    ' Cloud Storage URI to a JSON file.',
)

_OPTIMIZER = flags.DEFINE_string(
    'optimizer',
    'adamw_torch',
    'The optimizer.',
)

_LR_SCHEDULER_TYPE = flags.DEFINE_string(
    'lr_scheduler_type',
    'cosine',
    'The learning rate scheduler type.',
)

_SAVE_STEPS = flags.DEFINE_integer(
    'save_steps',
    10,
    'The save steps.',
)

_LOGGING_STEPS = flags.DEFINE_integer(
    'logging_steps',
    10,
    'The logging steps.',
)

_EVAL_STEPS = flags.DEFINE_integer(
    'eval_steps',
    10,
    'The number of training steps between evaluations.',
)

_TRAIN_SPLIT_NAME = flags.DEFINE_string(
    'train_split_name',
    'train',
    'The train split name.',
)

_EVAL_TASKS = flags.DEFINE_list(
    'eval_tasks',
    None,
    'List of eval task names (can have wildcards) as in'
    ' https://github.com/EleutherAI/lm-evaluation-harness. Will not run'
    ' evaluation if not set. Runs the built-in trainer evaluation loop if set'
    ' to `builtin_eval`.',
)

_EVAL_PER_DEVICE_BATCH_SIZE = flags.DEFINE_integer(
    'eval_per_device_batch_size',
    1,
    'The per device batch size for model evaluation.',
)

_EVAL_NUM_FEWSHOT = flags.DEFINE_integer(
    'eval_num_fewshot',
    None,
    'Run N-shot language model evaluation. Not implemented in `builtin_eval`.',
)

_EVAL_LIMIT = flags.DEFINE_float(
    'eval_limit',
    None,
    'Limit the number of examples per task. If <1, limit is a percentage of the'
    ' total number of examples.',
)

_EVAL_METRIC_NAME = flags.DEFINE_string(
    'eval_metric_name',
    'acc',
    'The metric name to aggregate during model evaluation.',
)

_EVAL_DATASET_PATH = flags.DEFINE_string(
    'eval_dataset_path',
    None,
    'Overrides the default evaluation dataset path. In `builtin_eval` mode,'
    ' this can be any Hugging Face dataset name.',
)

# We set the default eval split as `test`, based on observation from
# https://huggingface.co/datasets/timdettmers/openassistant-guanaco/viewer/default/test.
_EVAL_SPLIT = flags.DEFINE_string(
    'eval_split',
    'test',
    'Eval split name in the eval dataset for `builtin_eval`.',
)

_EVAL_TEMPLATE = flags.DEFINE_string(
    'eval_template',
    None,
    'Template for formatting language model evaluation data for `builtin_eval`.'
    ' Must be a filename under `templates` folder, without `.json` extension,'
    ' e.g. `alpaca`, or a Cloud Storage URI to a JSON file.',
)

_EVAL_COLUMN = flags.DEFINE_string(
    'eval_column',
    None,
    'Eval column name in the eval dataset for `builtin_eval`.',
)

_TRAIN_PRECISION = flags.DEFINE_enum(
    'train_precision',
    constants.PRECISION_MODE_16B,
    [
        constants.PRECISION_MODE_16,
        constants.PRECISION_MODE_16B,
        constants.PRECISION_MODE_32,
    ],
    'Precision to train the model.',
)

_USE_EXAMPLE_PACKING = flags.DEFINE_boolean(
    'use_example_packing',
    False,
    'Enables example packing during training, which uses '
    '`ConstantLengthDataset` under the hood.',
)

_COMPLETION_ONLY = flags.DEFINE_boolean(
    'completion_only',
    False,
    'If set, it uses DataCollatorForCompletionOnlyLM to train the model on the'
    ' generated prompts only, i.e., masking out the input',
)

_ATTN_IMPLEMENTATION = flags.DEFINE_string(
    'attn_implementation',
    None,
    'Attention implementation, can be `eager`, `sdpa` or `flash_attention_2`',
)

_MAX_GRAD_NORM = flags.DEFINE_float(
    'max_grad_norm',
    0.3,
    'Maximum gradient norm used for gradient clipping',
)

_WARNINGS_FILTER = flags.DEFINE_string(
    'warnings_filter',
    'ignore',
    'Warning filter as defined in '
    'https://docs.python.org/3/library/warnings.html#the-warnings-filter',
)

_LOGGER_LEVEL = flags.DEFINE_string(
    'logger_level',
    'passive',
    'logging level passed to TrainingArguments. Note that this is for python'
    ' logging module, NOT the one from absl',
)

_BENCHMARK_OUT_FILE = flags.DEFINE_string(
    'benchmark_out_file', None, 'file path for writing benchmark result'
)


_NCCL_TIMEOUT = flags.DEFINE_integer(
    'nccl_timeout', 6000, 'nccl timeout in seconds'
)

_TUNING_DATA_STATS_FILE = flags.DEFINE_string(
    'tuning_data_stats_file', None, 'file path for writing tuning data stats.'
)

_TARGET_MODULES = flags.DEFINE_list(
    'target_modules', None, 'The names of the modules to apply LoRA adapter to.'
)


@flags.multi_flags_validator(
    [
        _COMPLETION_ONLY.name,
        _USE_EXAMPLE_PACKING.name,
    ],
    message=(
        '`use_example_packing=True` does not work with `completion_only=True`'
    ),
)
def check_example_packing(flags_dict: Dict[str, Any]) -> bool:
  """Check to make sure example packing is enabled properly.

  Args:
    flags_dict: Dictionary containing flags to check.

  Returns:
    If `use_example_packing` is set properly.
  """
  if (
      flags_dict[_COMPLETION_ONLY.name]
      and flags_dict[_USE_EXAMPLE_PACKING.name]
  ):
    return False
  return True


@flags.multi_flags_validator(
    [
        _COMPLETION_ONLY.name,
        _TEMPLATE.name,
    ],
    message='`template` should be provided if using `completion_only=True`',
)
def check_completion_only(flags_dict: Dict[str, Any]) -> bool:
  """Check to make sure completion_only is enabled properly.

  Args:
    flags_dict: Dictionary containing flags to check.

  Returns:
    If `completion_only` is set properly
  """
  if flags_dict[_COMPLETION_ONLY.name] and flags_dict[_TEMPLATE.name] is None:
    return False
  return True


# References:
# Huggingface SFT trainer example:
# https://github.com/huggingface/trl/blob/main/examples/scripts/sft_trainer.py.
# Huggingface sagemaker example:
# https://github.com/huggingface/notebooks/blob/main/sagemaker/28_train_llms_with_qlora/scripts/run_clm.py.


# Copied from https://github.com/artidoro/qlora/blob/main/qlora.py.
def find_all_linear_names(
    model: AutoModelForCausalLM, precision_mode: str
) -> list[str]:
  """Finds all linear module names."""
  if precision_mode == constants.PRECISION_MODE_4:
    cls = bnb.nn.Linear4bit
  elif precision_mode == constants.PRECISION_MODE_8:
    cls = bnb.nn.Linear8bitLt
  else:
    cls = torch.nn.Linear
  lora_module_names = set()
  for name, module in model.named_modules():
    if isinstance(module, cls):
      names = name.split('.')
      lora_module_names.add(names[0] if len(names) == 1 else names[-1])
  if 'lm_head' in lora_module_names:  # needed for 16-bit
    lora_module_names.remove('lm_head')
  return list(lora_module_names)


def finetune_instruct(
    pretrained_model_id: str,
    dataset_name: str,
    output_dir: str,
    logging_output_dir: str,
    lora_rank: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    warmup_ratio: int = 0.03,
    num_epochs: Optional[float] = None,
    max_steps: Optional[int] = None,
    warmup_steps: int = 10,
    max_seq_length: int = 512,
    learning_rate: float = 2e-4,
    precision_mode: str = None,
    instruct_column_in_dataset: str = constants.DEFAULT_INSTRUCT_COLUMN_IN_DATASET,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    optim: str = 'paged_adamw_32bit',
    weight_decay: float = 0.001,
    enable_gradient_checkpointing: bool = False,
    enable_peft: bool = True,
    template: str = None,
    lr_scheduler_type: str = 'constant',
    save_steps: int = 10,
    logging_steps: int = 10,
    train_split_name: str = 'train',
    eval_config: Optional[eval_lib.EvalConfig] = None,
    report_to: str = constants.REPORT_TO_NONE,
    access_token: Optional[str] = None,
    train_precision: str = constants.PRECISION_MODE_16B,
    use_example_packing: bool = False,
    attn_implementation: Optional[str] = None,
    max_grad_norm: float = 0.3,
    completion_only: bool = False,
    logger_level: str = 'passive',
    benchmark_out_file: Optional[str] = None,
    tuning_data_stats_file: Optional[str] = None,
    target_modules: Optional[str] = None,
) -> None:
  """Finetunes instruct."""
  logging.info('on entering instruct_lora, %s', utils.gpu_stats_str())
  gradient_checkpointing_kwargs = {}
  # DDP provides limited support with the reentrant variant of gradient
  # checkpoint [1]. Below is an indirect way of checking whether DDP will be
  # used. It is "indirect" because there are complex logic under the hood of
  # `SFTTrainer` and since those are not public API, they might change as we
  # update the library.
  if PartialState().distributed_type == DistributedType.MULTI_GPU:
    gradient_checkpointing_kwargs['use_reentrant'] = False

  tokenizer = utils.load_tokenizer(
      pretrained_model_id,
      'right',
      access_token=access_token,
  )

  train_dataset = dataset_validation_util.load_dataset_with_template(
      dataset_name,
      split=train_split_name,
      input_column=instruct_column_in_dataset,
      template=template,
      tokenizer=tokenizer,
  )

  if tuning_data_stats_file:
    with PartialState().main_process_first():
      effective_batch_size = (
          per_device_train_batch_size
          * gradient_accumulation_steps
          * PartialState().num_processes
      )
      logging.info(
          'getting tuning data stats with effective batch size %s',
          effective_batch_size,
      )
      train_dataset_stats = utils.get_dataset_stats(
          train_dataset,
          tokenizer,
          instruct_column_in_dataset,
          effective_batch_size,
      )
      logging.info('stats: %s', train_dataset_stats)
      tuning_data_stats_file = dataset_validation_util.force_gcs_fuse_path(
          tuning_data_stats_file
      )
      with open(tuning_data_stats_file, 'w') as out_f:
        json.dump(dataclasses.asdict(train_dataset_stats), out_f)

  model = utils.load_model(
      pretrained_model_id=pretrained_model_id,
      tokenizer=tokenizer,
      precision_mode=precision_mode,
      enable_gradient_checkpointing=enable_gradient_checkpointing,
      access_token=access_token,
      attn_implementation=attn_implementation,
      train_precision=train_precision,
  )

  if enable_peft:
    if target_modules is None:
      target_modules = find_all_linear_names(
          model, precision_mode=precision_mode
      )
    logging.info('applying lora adapters to modules: %s', target_modules)
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_rank,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=target_modules,
    )
    # If we pass in `peft_config` to SFTTrainer, it does a lot of magic under
    # the hood, e.g., calling `prepare_model_for_kbit_training` before calling
    # `get_peft_model`, which may revert other changes we did before. That's why
    # we are calling `get_peft_model` explicitly here.
    model = get_peft_model(model, peft_config)

    # This is to work-around mix-precision training. This issue is not fixed as
    # of transformers==4.41.2.
    # See b/332760883#comment30 for more details.
    if precision_mode in (
        constants.PRECISION_MODE_16,
        constants.PRECISION_MODE_16B,
    ):
      for param in filter(lambda p: p.requires_grad, model.parameters()):
        param.data = param.data.to(torch.float32)

  if not logging_output_dir:
    logging_output_dir = output_dir

  # To use singleton PartialState() without re-initializing it. See
  # b/357970482#comment3
  accelerator_config = {'use_configured_state': True}

  training_arguments = TrainingArguments(
      report_to=report_to,
      output_dir=output_dir,
      per_device_train_batch_size=per_device_train_batch_size,
      gradient_accumulation_steps=gradient_accumulation_steps,
      optim=optim,
      save_steps=save_steps,
      save_strategy='steps',
      save_total_limit=3,
      logging_dir=os.path.join(logging_output_dir, 'logs'),
      logging_steps=logging_steps,
      learning_rate=learning_rate,
      fp16=(train_precision == constants.PRECISION_MODE_16),
      bf16=(train_precision == constants.PRECISION_MODE_16B),
      max_grad_norm=max_grad_norm,
      num_train_epochs=num_epochs if num_epochs else -1,
      max_steps=max_steps if max_steps else -1,
      warmup_ratio=warmup_ratio,
      warmup_steps=warmup_steps,
      group_by_length=False,
      lr_scheduler_type=lr_scheduler_type,
      gradient_checkpointing=enable_gradient_checkpointing,
      gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
      weight_decay=weight_decay,
      log_level=logger_level,
      accelerator_config=accelerator_config,
  )
  trainer_kwargs = {}
  if completion_only and template:
    template_json = dataset_validation_util.get_template(template_path=template)
    instruction_sep = dataset_validation_util.get_instruction_separator(
        template_json
    )
    response_sep = dataset_validation_util.get_response_separator(template_json)
    if not response_sep:
      raise ValueError(
          '`response_separator` must be provided to use'
          ' `DataCollatorForCompletionOnlyLM`'
      )

    trainer_kwargs['data_collator'] = DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_sep,
        response_template=response_sep,
        tokenizer=tokenizer,
    )
    logging.info('using DataCollatorForCompletionOnlyLM')

  trainer_stats_callback = callbacks.TrainerStatsCallback(
      max_seq_length, benchmark_out_file
  )
  trainer = eval_lib.create_trainer(
      cls=SFTTrainer,
      eval_config=eval_config,
      model=model,
      train_dataset=train_dataset,
      dataset_text_field=instruct_column_in_dataset,
      max_seq_length=max_seq_length,
      tokenizer=tokenizer,
      args=training_arguments,
      packing=use_example_packing,
      callbacks=[trainer_stats_callback],
      **trainer_kwargs,
  )

  # `eval_lib.create_trainer` might modify the training args. Printing here
  # should capture what will be used by the trainer.
  if PartialState().is_main_process:
    logging.info('training args: %s', trainer.args)

  if enable_peft:
    trainer.model.print_trainable_parameters()

  if trainer.is_fsdp_enabled:
    logging.info('Trainer running with FSDP.')
  elif trainer.is_deepspeed_enabled:
    logging.info('Trainer running with DeepSpeed.')
  else:
    logging.info('Trainer running without parallelism.')

  trainer.train()

  # Always save the final checkpoint.
  final_checkpoint = utils.get_final_checkpoint_path(output_dir)
  logging.info('The final checkpoint is: %s.', final_checkpoint)

  if trainer.is_fsdp_enabled:
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type('FULL_STATE_DICT')
    # This method saves the sharded weights like `accelerator.save_state`, see
    # https://huggingface.co/docs/accelerate/en/usage_guides/fsdp#saving-and-loading
    trainer.save_model(output_dir)
    model = trainer.model.cpu()  # Avoids GPU OOM
    state_dict = trainer.accelerator.get_state_dict(model)
    # To aggregate the weights from all the devices, we need to use
    # `state_dict=state_dict`.
    model.save_pretrained(
        final_checkpoint,
        state_dict=state_dict,
        is_main_process=PartialState().is_main_process,
        save_embedding_layers=False,  # Only pad token is added. See go/lora-adapter-pad-token #pylint: disable=line-too-long
    )
    model.cuda()  # Move back to GPU to do eval.
  else:
    trainer.model.save_pretrained(
        final_checkpoint,
        is_main_process=PartialState().is_main_process,
        save_embedding_layers=False,  # Only pad token is added. See go/lora-adapter-pad-token #pylint: disable=line-too-long
    )

  if eval_config is not None and trainer.eval_dataset is not None:
    metrics = trainer.evaluate(metric_key_prefix='eval')
    # Both `log_metrics` and `save_metrics` are multiple process safe.
    # https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/trainer_pt_utils.py#L911 #pylint: disable=line-too-long
    # https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/trainer_pt_utils.py#L1001 #pylint: disable=line-too-long
    trainer.log_metrics('eval', metrics)
    trainer.save_metrics('eval', metrics)
    if PartialState().is_main_process:
      hp_metric = metrics[f'eval_{eval_config.metric_name}']
      hpt = hypertune.HyperTune()
      hpt.report_hyperparameter_tuning_metric(
          hyperparameter_metric_tag=constants.HP_METRIC_TAG,
          metric_value=hp_metric,
      )
      logging.info('Send HP metric: %f to hyperparameter tuning.', hp_metric)
    PartialState().wait_for_everyone()

  if not enable_peft:
    tokenizer.save_pretrained(
        final_checkpoint, is_main_process=PartialState().is_main_process
    )


def main(unused_argv: Sequence[str]) -> None:
  # This needs to be called before any other PartialState() calls.
  utils.init_partial_state(
      timeout=datetime.timedelta(seconds=_NCCL_TIMEOUT.value)
  )

  utils.print_library_versions()
  warnings.simplefilter(_WARNINGS_FILTER.value)

  pretrained_model_id = fileutils.force_gcs_path(_PRETRAINED_MODEL_ID.value)
  if dataset_validation_util.is_gcs_path(pretrained_model_id):
    pretrained_model_id = dataset_validation_util.download_gcs_uri_to_local(
        pretrained_model_id
    )

  output_dir = utils.GcsOrLocalDirectory(
      _OUTPUT_DIR.value, check_empty=True, upload_from_all_nodes=True
  )

  # GCS Fuse does not sync flushed files if not closed. See b/361771727.
  logging_output_dir = fileutils.force_gcs_path(_LOGGING_OUTPUT_DIR.value)

  # Creates evaluation config.
  if _EVAL_TASKS.value:
    eval_config = eval_lib.EvalConfig(
        tasks=_EVAL_TASKS.value,
        per_device_batch_size=_EVAL_PER_DEVICE_BATCH_SIZE.value,
        num_fewshot=_EVAL_NUM_FEWSHOT.value,
        limit=_EVAL_LIMIT.value,
        metric_name=_EVAL_METRIC_NAME.value,
        steps=_EVAL_STEPS.value,
        dataset_path=dataset_validation_util.force_gcs_fuse_path(
            _EVAL_DATASET_PATH.value
        ),
        split=_EVAL_SPLIT.value,
        template=_EVAL_TEMPLATE.value,
        column=_EVAL_COLUMN.value,
        tokenize_dataset=False,
    )
  else:
    eval_config = None

  if _REPORT_TO.value == constants.REPORT_TO_WANDB:
    wandb.login()

  finetune_instruct(
      pretrained_model_id=pretrained_model_id,
      dataset_name=_DATASET_NAME.value,
      output_dir=output_dir.local_dir,
      logging_output_dir=logging_output_dir,
      precision_mode=_PRECISION_MODE.value,
      lora_rank=_LORA_RANK.value,
      lora_alpha=_LORA_ALPHA.value,
      lora_dropout=_LORA_DROPOUT.value,
      warmup_ratio=_WARMUP_RATIO.value,
      num_epochs=_NUM_EPOCHS.value,
      warmup_steps=_WARMUP_STEPS.value,
      max_steps=_MAX_STEPS.value,
      max_seq_length=_MAX_SEQ_LENGTH.value,
      learning_rate=_LEARNING_RATE.value,
      instruct_column_in_dataset=_INSTRUCT_COLUMN_IN_DATASET.value,
      per_device_train_batch_size=_PER_DEVICE_TRAIN_BATCH_SIZE.value,
      optim=_OPTIMIZER.value,
      weight_decay=_WEIGHT_DECAY.value,
      gradient_accumulation_steps=_GRADIENT_ACCUMULATION_STEPS.value,
      enable_gradient_checkpointing=_ENABLE_GRADIENT_CHECKPOINTING.value,
      enable_peft=_ENABLE_PEFT.value,
      template=_TEMPLATE.value,
      lr_scheduler_type=_LR_SCHEDULER_TYPE.value,
      save_steps=_SAVE_STEPS.value,
      logging_steps=_LOGGING_STEPS.value,
      train_split_name=_TRAIN_SPLIT_NAME.value,
      eval_config=eval_config,
      report_to=_REPORT_TO.value,
      access_token=_HUGGINGFACE_ACCESS_TOKEN.value,
      train_precision=_TRAIN_PRECISION.value,
      use_example_packing=_USE_EXAMPLE_PACKING.value,
      attn_implementation=_ATTN_IMPLEMENTATION.value,
      max_grad_norm=_MAX_GRAD_NORM.value,
      completion_only=_COMPLETION_ONLY.value,
      logger_level=_LOGGER_LEVEL.value,
      benchmark_out_file=_BENCHMARK_OUT_FILE.value,
      tuning_data_stats_file=_TUNING_DATA_STATS_FILE.value,
      target_modules=_TARGET_MODULES.value,
  )
  # Frees the model from GPU.
  utils.force_gc()

  output_dir.upload_to_gcs(skip_if_exists=True)


if __name__ == '__main__':
  app.run(main)
