"""Common libraries for PEFT."""

from collections.abc import Mapping, Sequence
import datetime
import gc
import os
from typing import Any

from absl import logging
import accelerate
from accelerate import DistributedType
from accelerate import PartialState
import peft
from peft import PeftModel
from peft import prepare_model_for_kbit_training
import torch
import transformers
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import FbgemmFp8Config
import trl

from util import dataset_validation_util
from util import constants

_LLAMA_3_1_405B_MODEL_ID = "Meta-Llama-3.1-405B"
_LOCAL_MERGED_MODEL_DIR = "/tmp/merged_model"
_GEMMA2_MODEL = "gemma-2"


def load_model(
    pretrained_model_name_or_path: str,
    tokenizer: AutoTokenizer,
    precision_mode: str = None,
    gradient_checkpointing: bool = False,
    gradient_checkpointing_kwargs: Mapping[str, Any] | None = None,
    access_token: str | None = None,
    attn_implementation: str | None = None,
    train_precision: str | None = None,
    device_map: str | None = None,
    is_training: bool = True,
) -> AutoModelForCausalLM:
  """Loads models from the local dir if specified or from huggingface."""
  # The `distributed_type` we got through `PartialState` is incorrect for FSDP.
  # And that's why `Accelerator` is used here.
  # See b/357138252 for more details.
  accelerator = accelerate.Accelerator()
  logging.info("using distributed_type %s", accelerator.distributed_type)

  if device_map is None:
    if accelerator.distributed_type == DistributedType.MULTI_GPU:
      # https://github.com/artidoro/qlora/issues/186#issuecomment-1943045599
      # and b/342038175.
      device_map = {"": accelerator.process_index}
    elif accelerator.distributed_type == DistributedType.DEEPSPEED:
      # Deepspeed Zero3 does not allow setting device_map.
      # https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/modeling_utils.py#L2941-L2943
      device_map = None
    elif accelerator.distributed_type == DistributedType.FSDP:
      if precision_mode in [
          constants.PRECISION_MODE_4,
          constants.PRECISION_MODE_8,
      ]:
        device_map = trl.get_kbit_device_map()
      else:
        device_map = None
    elif (
        accelerator.distributed_type == DistributedType.NO
        and torch.cuda.device_count() > 1
    ):
      # Setting device map to None to avoid using model parallelism (MP) when
      # there are multiple GPUs, which can have very inefficient GPU utilization
      # (b/342252819). This setting should trigger torch's nn.DataParallel
      # instead, which has better GPU utilization.
      device_map = None
    else:
      device_map = "auto"
  logging.info("using device_map %s", device_map)

  if train_precision == constants.PRECISION_MODE_32:
    train_dtype = torch.float32
  elif train_precision == constants.PRECISION_MODE_16:
    train_dtype = torch.float16
  elif train_precision == constants.PRECISION_MODE_16B:
    train_dtype = torch.bfloat16
  else:
    train_dtype = "auto"

  quantization_config = None
  # Note: use_cache is False when enable gradient checkpointing.
  if precision_mode == constants.PRECISION_MODE_32:
    torch_dtype = torch.float32
  elif precision_mode == constants.PRECISION_MODE_16:
    torch_dtype = torch.float16
  elif precision_mode == constants.PRECISION_MODE_16B:
    torch_dtype = torch.bfloat16
  elif precision_mode == constants.PRECISION_MODE_8:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True, int8_threshold=0
    )
    torch_dtype = train_dtype
  elif precision_mode == constants.PRECISION_MODE_4:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=train_dtype,
    )
    # `bnb_4bit_quant_storage` must be set when using FSDP.
    # https://huggingface.co/docs/bitsandbytes/main/en/fsdp_qlora
    if accelerator.distributed_type == DistributedType.FSDP:
      quantization_config.bnb_4bit_quant_storage = train_dtype
    torch_dtype = train_dtype
  else:
    raise ValueError(f"Invalid precision mode: {precision_mode}")
  logging.info("using torch_type=%s", torch_dtype)

  model_kwargs = {
      "use_cache": not gradient_checkpointing,
      "device_map": device_map,
      "torch_dtype": torch_dtype,
      "quantization_config": quantization_config,
      "trust_remote_code": False,
      "token": access_token,
      "attn_implementation": attn_implementation,
  }
  if _GEMMA2_MODEL in pretrained_model_name_or_path:
    # The cache_implementation for Gemma 2 is set to hybrid by default. This
    # param is only supported by Gemma 2. The default 'hybrid' value causes an
    # issue when use_cache is set to False. So we have to use 'None' in such
    # cases.
    # https://github.com/huggingface/transformers/commit/238b13478df209ab534f2195a397dc64a3930883
    model_kwargs["cache_implementation"] = (
        None if gradient_checkpointing else "hybrid"
    )

  model = AutoModelForCausalLM.from_pretrained(
      pretrained_model_name_or_path, **model_kwargs
  )

  if precision_mode in (constants.PRECISION_MODE_4, constants.PRECISION_MODE_8):
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
    )

  if gradient_checkpointing:
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
    )

  # Flash attention only supports fp16 or bf16 [1].
  # prepare_model_for_kbit_training will force cast some layers to float32 [2]
  #
  # [1]: https://github.com/Dao-AILab/flash-attention/issues/882
  # [2]: https://github.com/huggingface/peft/blob/v0.10.0/src/peft/utils/other.py#L79-L81 # pylint: disable=line-too-long
  if attn_implementation == "flash_attention_2" and precision_mode in (
      constants.PRECISION_MODE_4,
      constants.PRECISION_MODE_8,
  ):
    for _, param in model.named_parameters():
      if param.dtype == torch.float32:
        param.data = param.data.to(torch_dtype)

  if is_training:
    # KV cache is useless during training
    # https://stackoverflow.com/a/77408076
    model.config.use_cache = False

  if dataset_validation_util.should_add_pad_token(
      pretrained_model_name_or_path
  ):
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    if is_training:
      # The following is needed since we added a new token that needs to be
      # learned.
      # https://github.com/QwenLM/Qwen/issues/405#issuecomment-1751680291
      model.enable_input_require_grads()

  return model


def _merge_causal_language_model_with_lora_internal(
    pretrained_model_name_or_path: str,
    merge_precision_mode: str,
    finetuned_lora_model_dir: str,
    merged_model_output_dir: str,
    access_token: str | None = None,
) -> None:
  """Internal function to merges the base model with the lora adapter."""
  logging.info("loading tokenizer...")
  tokenizer = dataset_validation_util.load_tokenizer(
      pretrained_model_name_or_path
  )

  # Note: merging peft adapter requires loading model in 16 bits, so merging
  # is done on CPU on purpose in case one GPU cannot hold the base model.
  logging.info("loading model %s...", pretrained_model_name_or_path)
  device_map = "cpu"
  model = load_model(
      pretrained_model_name_or_path=pretrained_model_name_or_path,
      tokenizer=tokenizer,
      precision_mode=merge_precision_mode,
      access_token=access_token,
      device_map=device_map,
      is_training=False,
  )

  logging.info("loading LoRA model...")
  model = PeftModel.from_pretrained(
      model, finetuned_lora_model_dir, device_map=device_map
  )

  logging.info("merging base model with finetuned LoRA model...")
  model = model.merge_and_unload()

  logging.info("saving model to %s...", merged_model_output_dir)
  model.save_pretrained(
      merged_model_output_dir,
      safe_serialization=False,
      is_main_process=PartialState().is_main_process,
  )

  logging.info("saving tokenizer to %s...", merged_model_output_dir)
  tokenizer.save_pretrained(
      merged_model_output_dir,
      is_main_process=PartialState().is_main_process,
  )


def merge_causal_language_model_with_lora(
    pretrained_model_name_or_path: str,
    precision_mode: str,
    finetuned_lora_model_dir: str,
    merged_model_output_dir: str,
    access_token: str | None = None,
) -> None:
  """Merges the base model with the lora adapter."""

  # Set merge related variables.
  if precision_mode == constants.PRECISION_MODE_FP8:
    # Merge as FP16. FP8 requires conversion after merge.
    merge_precision_mode = constants.PRECISION_MODE_16
    local_merged_model_dir = _LOCAL_MERGED_MODEL_DIR
  else:
    merge_precision_mode = precision_mode
    local_merged_model_dir = merged_model_output_dir

  if PartialState().is_main_process:
    logging.info("Starting merging job...")
    _merge_causal_language_model_with_lora_internal(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        merge_precision_mode=merge_precision_mode,
        finetuned_lora_model_dir=finetuned_lora_model_dir,
        merged_model_output_dir=local_merged_model_dir,
        access_token=access_token,
    )
    logging.info("merging job is done")

  # Wait for all processes to sync here.
  PartialState().wait_for_everyone()

  if precision_mode == constants.PRECISION_MODE_FP8:
    convert_model_to_fp8(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        merged_model_output_dir=local_merged_model_dir,
        quantized_model_output_dir=merged_model_output_dir,
        access_token=access_token,
    )


def convert_model_to_fp8(
    pretrained_model_name_or_path: str,
    merged_model_output_dir: str,
    quantized_model_output_dir: str,
    access_token: str | None = None,
) -> None:
  """Converts the model to fp8.

  Args:
    pretrained_model_name_or_path: Original base model name or path.
    merged_model_output_dir: Path to directory containing the merged model.
    quantized_model_output_dir: Path to directory to save the quantized model.
    access_token: Access token for accessing the model.
  """
  if PartialState().is_main_process:
    quantization_config = FbgemmFp8Config(
        modules_to_not_convert=_maybe_get_modules_to_not_convert_by_model_id(
            pretrained_model_name_or_path
        )
    )
    quantized_model = AutoModelForCausalLM.from_pretrained(
        merged_model_output_dir,
        device_map="cpu",
        quantization_config=quantization_config,
        trust_remote_code=False,
        token=access_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(merged_model_output_dir)

    quantized_model.save_pretrained(quantized_model_output_dir)
    tokenizer.save_pretrained(quantized_model_output_dir)
  PartialState().wait_for_everyone()


def force_gc():
  """Collects garbage immediately to release unused CPU/GPU resources."""
  gc.collect()
  torch.cuda.empty_cache()


def init_partial_state(
    timeout: datetime.timedelta = datetime.timedelta(seconds=600),
) -> None:
  """Initializes the partial state with timeout."""
  # This needs to be called before any other PartialState() calls, and
  # TrainingArguments needs `use_configured_state`. See b/357970482#comment3
  # for more details.
  PartialState(timeout=timeout)


def print_library_versions():
  if PartialState().is_main_process:
    logging.info("======================")
    logging.info("library versions")
    logging.info("======================")
    logging.info("accelerate: %s", accelerate.__version__)
    logging.info("peft: %s", peft.__version__)
    logging.info("transformers: %s", transformers.__version__)
    logging.info("trl: %s", trl.__version__)
  PartialState().wait_for_everyone()


def get_final_checkpoint_path(output_dir: str) -> str:
  """Returns the final checkpoint path."""
  return os.path.join(output_dir, constants.FINAL_CHECKPOINT_DIRNAME)


def _maybe_get_modules_to_not_convert_by_model_id(
    pretrained_model_name_or_path: str,
) -> Sequence[str] | None:
  """Returns the modules to not convert for the model."""
  if _LLAMA_3_1_405B_MODEL_ID in pretrained_model_name_or_path:
    return _get_llama_3_1_405b_modules_to_not_convert()
  else:
    return None


def _get_llama_3_1_405b_modules_to_not_convert() -> Sequence[str]:
  """Returns the modules to not convert for Llama 3.1 405B model."""
  modules_to_not_convert = ["lm_head"]
  for idx in range(126):
    for proj_name in ["k_proj", "o_proj", "q_proj", "v_proj"]:
      modules_to_not_convert.append(f"model.layers.{idx}.self_attn.{proj_name}")
  for proj_name in ["down_proj", "gate_proj", "up_proj"]:
    modules_to_not_convert.append(f"model.layers.0.mlp.{proj_name}")
    modules_to_not_convert.append(f"model.layers.125.mlp.{proj_name}")
  return tuple(modules_to_not_convert)
