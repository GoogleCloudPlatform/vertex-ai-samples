"""Common libraries for PEFT."""

import dataclasses
import datetime
import gc
import multiprocessing as mp
import os
import subprocess
from typing import Any, Dict, Optional, Sequence

from absl import logging
import accelerate
from accelerate import DistributedType
from accelerate import PartialState
from google.protobuf import json_format
from kfp.pipeline_spec import pipeline_spec_pb2
import numpy as np
import peft
from peft import PeftModel
from peft import prepare_model_for_kbit_training
import pynvml
import torch
import transformers
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import FbgemmFp8Config
from transformers.integrations import is_deepspeed_zero3_enabled
import trl

from util import dataset_validation_util
from util import constants
from util import fileutils


_MODELS_REQUIRING_PAD_TOKEN = ("llama", "falcon", "mistral", "mixtral")
_MODELS_REQUIRING_EOS_TOEKN = ("gemma-2b", "gemma-7b")
_LLAMA_3_1_405B_MODEL_ID = "Meta-Llama-3.1-405B"
_LOCAL_MERGED_MODEL_DIR = "/tmp/merged_model"



class GcsOrLocalDirectory(os.PathLike):
  """A class to represent a directory with upload support if GCS path is given.

  This class is used to represent a directory. It can be used for a temporary
  local directory and for uploading files to the GCS directory later if the
  given path is a GCS directory. If the given path is a local directory, a call
  to gcs_dir attribute will raise an error. This class has multi-node and
  multi-process support with accelerate.

  Attributes:
    local_dir: The local directory to store the files.
    gcs_dir: The path to the GCS directory.
  """

  def __init__(
      self,
      path: str,
      check_empty: bool = False,
      upload_from_all_nodes: bool = False,
  ):
    """Initializes the GcsOrLocalDirectory.

    Args:
      path: The path to the directory.
      check_empty: If True, check if the GCS directory is empty. No-op for local
        directory.
      upload_from_all_nodes: If True, upload the local directory to GCS from all
        nodes.
    """
    if len(path) > 1:
      path = path.rstrip("/")

    self._upload_from_all_nodes = upload_from_all_nodes

    if path.startswith(constants.GCS_URI_PREFIX) or path.startswith(
        constants.GCSFUSE_URI_PREFIX
    ):
      self._is_gcs_path = True
      self._local_dir = _get_local_dir_from_gcs_dir(path)
      self._gcs_dir = fileutils.force_gcs_path(path)
      os.makedirs(self.local_dir, exist_ok=True)

      with PartialState().main_process_first():
        if (
            check_empty
            and PartialState().is_main_process
            and not _is_gcs_dir_empty(self._gcs_dir)
        ):
          raise ValueError(f"{self._gcs_dir} needs to be empty.")
    else:
      self._is_gcs_path = False
      self._local_dir = path
      self._gcs_dir = path

  def __fspath__(self) -> str:
    return self.local_dir

  @property
  def local_dir(self) -> str:
    return self._local_dir

  @property
  def gcs_dir(self) -> str:
    """Returns the GCS directory path.

    Returns:
      The GCS directory path.

    Raises:
      ValueError: If the path is not a GCS path.
    """
    if not self._is_gcs_path:
      raise ValueError(f"{self._gcs_dir} is not a GCS path.")
    return self._gcs_dir

  def upload_to_gcs(
      self,
      skip_if_exists: bool = True,
      force_upload: bool = False,
  ):
    """Uploads the local directory to GCS."""
    if not self._is_gcs_path:
      logging.info(
          "Not uploading to GCS since %s is not a GCS path.", self.local_dir
      )
      return

    if not os.listdir(self.local_dir):
      logging.info("Not uploading to GCS since %s is empty.", self.local_dir)
      return

    target = os.path.dirname(self.gcs_dir) + "/"
    # Avoid race condition uploading the same file from multiple processes.
    with PartialState().main_process_first():
      if not PartialState().is_local_main_process:
        # Non local main processes don't upload.
        pass
      elif self._upload_from_all_nodes or PartialState().is_main_process:
        logging.info("Uploading %s to %s...", self.local_dir, target)
        cmd = [
            "gsutil",
            "-m",
            "cp",
            "-r",
        ]
        if skip_if_exists:
          cmd.append("-n")
        if force_upload:
          cmd.append("-f")
        cmd.extend([self.local_dir, target])
        subprocess.check_output(cmd)
        logging.info("%s uploaded.", self.local_dir)


def _get_local_dir_from_gcs_dir(path: str) -> str:
  return os.path.join(
      constants.LOCAL_OUTPUT_DIR,
      dataset_validation_util.force_gcs_fuse_path(path)[1:],
  )


def _is_gcs_dir_empty(path: str) -> bool:
  """Checks if a GCS directory is empty.

  Args:
    path: The GCS directory path.

  Returns:
    True if the directory is empty.

  Raises:
    subprocess.CalledProcessError: If the gsutil command failure reason is not
      because the dir is empty.
  """
  path = path.rstrip("/") + "/"
  try:
    subprocess.check_output(["gsutil", "ls", path], stderr=subprocess.STDOUT)
  except subprocess.CalledProcessError as e:
    if (
        str(e.output, encoding="utf-8")
        == "CommandException: One or more URLs matched no objects.\n"
    ):
      return True
    else:
      logging.info(str(e.output, encoding="utf-8"))
      raise
  else:
    return False


def load_tokenizer(
    pretrained_model_id: str,
    padding_side: Optional[str] = None,
    access_token: Optional[str] = None,
) -> AutoTokenizer:
  """Loads tokenizer based on `pretrained_model_id`."""
  tokenizer_kwargs = {}
  if should_add_eos_token(pretrained_model_id):
    tokenizer_kwargs["add_eos_token"] = True
  if padding_side:
    tokenizer_kwargs["padding_side"] = padding_side

  with PartialState().local_main_process_first():
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_id,
        trust_remote_code=False,
        use_fast=True,
        token=access_token,
        **tokenizer_kwargs,
    )

  if should_add_pad_token(pretrained_model_id):
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

  return tokenizer


def load_model(
    pretrained_model_id: str,
    tokenizer: AutoTokenizer,
    precision_mode: str = None,
    enable_gradient_checkpointing: bool = False,
    gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None,
    access_token: Optional[str] = None,
    attn_implementation: Optional[str] = None,
    train_precision: Optional[str] = None,
    device_map: Optional[str] = None,
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

  model = AutoModelForCausalLM.from_pretrained(
      pretrained_model_id,
      use_cache=not enable_gradient_checkpointing,
      device_map=device_map,
      torch_dtype=torch_dtype,
      quantization_config=quantization_config,
      trust_remote_code=True,
      token=access_token,
      attn_implementation=attn_implementation,
  )

  if precision_mode in (constants.PRECISION_MODE_4, constants.PRECISION_MODE_8):
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=enable_gradient_checkpointing,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
    )

  if enable_gradient_checkpointing:
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

  if should_add_pad_token(pretrained_model_id):
    model.resize_token_embeddings(len(tokenizer))
    if is_training:
      # The following is needed since we added a new token that needs to be
      # learned.
      # https://github.com/QwenLM/Qwen/issues/405#issuecomment-1751680291
      model.enable_input_require_grads()

  return model


def _merge_causal_language_model_with_lora_internal(
    pretrained_model_id: str,
    merge_precision_mode: str,
    finetuned_lora_model_dir: str,
    merged_model_output_dir: str,
    access_token: Optional[str] = None,
) -> None:
  """Internal function to merges the base model with the lora adapter."""
  logging.info("loading tokenizer...")
  tokenizer = load_tokenizer(pretrained_model_id)

  # Note: merging peft adapter requires loading model in 16 bits, so merging
  # is done on CPU on purpose in case one GPU cannot hold the base model.
  logging.info("loading model %s...", pretrained_model_id)
  device_map = "cpu"
  model = load_model(
      pretrained_model_id=pretrained_model_id,
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


def merge_causal_language_model_with_lora_fsdp(
    pretrained_model_id: str,
    merge_precision_mode: str,
    finetuned_lora_model_dir: str,
    merged_model_output_dir: str,
    access_token: Optional[str] = None,
) -> None:
  """Merges the base model with the lora adapter for FSDP.

  Only the main process should call this function.

  Args:
    pretrained_model_id: Predefined base model name or path to directory
      containing model checkpoints.
    merge_precision_mode: Precision mode for saving model weights.
    finetuned_lora_model_dir: Path to directory containing PEFT-finetuned model
      weights.
    merged_model_output_dir: Path to directory to save the merged model.
    access_token: Access token for accessing the model.
  """
  assert PartialState().is_main_process
  _merge_causal_language_model_with_lora_internal(
      pretrained_model_id=pretrained_model_id,
      merge_precision_mode=merge_precision_mode,
      finetuned_lora_model_dir=finetuned_lora_model_dir,
      merged_model_output_dir=merged_model_output_dir,
      access_token=access_token,
  )


def merge_causal_language_model_with_lora(
    pretrained_model_id: str,
    precision_mode: str,
    finetuned_lora_model_dir: str,
    merged_model_output_dir: str,
    access_token: Optional[str] = None,
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
    # When deepspeed Zero3 is enabled, users are not allowed to specify
    # `device_map` when loading the model (even on CPU).
    #
    # To work-around this, we kick off another process (from the
    # is_main_process) and set up the environment to avoid using Deepspeed when
    # doing the merging.
    if is_deepspeed_zero3_enabled():
      ctx = mp.get_context("spawn")
      os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
      merge_job = ctx.Process(
          target=_merge_causal_language_model_with_lora_internal,
          args=(
              pretrained_model_id,
              merge_precision_mode,
              finetuned_lora_model_dir,
              local_merged_model_dir,
          ),
          kwargs={
              "access_token": access_token,
          },
      )
      merge_job.start()
      merge_job.join()
      os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"
    else:
      _merge_causal_language_model_with_lora_internal(
          pretrained_model_id=pretrained_model_id,
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
        pretrained_model_name_or_path=pretrained_model_id,
        merged_model_output_dir=local_merged_model_dir,
        quantized_model_output_dir=merged_model_output_dir,
        access_token=access_token,
    )


def convert_model_to_fp8(
    pretrained_model_name_or_path: str,
    merged_model_output_dir: str,
    quantized_model_output_dir: str,
    access_token: Optional[str] = None,
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


@dataclasses.dataclass
class TuningDataStats:
  tuning_dataset_example_count: int
  total_billable_token_count: int
  tuning_step_count: int


def get_dataset_stats(
    dataset: Any,
    tokenizer: transformers.PreTrainedTokenizer,
    column: str,
    effective_batch_size: int,
) -> TuningDataStats:
  """Calculates dataset statistics, e.g., total number of tokens."""
  tokenized_dataset = dataset.map(lambda x: tokenizer(x[column]))
  inputs = tokenized_dataset["input_ids"]
  tuning_dataset_example_count = int(len(inputs))
  total_billable_token_count = int(np.sum([len(ex) for ex in inputs]))
  tuning_step_count = (
      tuning_dataset_example_count + effective_batch_size - 1
  ) // effective_batch_size
  return TuningDataStats(
      tuning_dataset_example_count,
      total_billable_token_count,
      tuning_step_count,
  )


def force_gc():
  """Collects garbage immediately to release unused CPU/GPU resources."""
  gc.collect()
  torch.cuda.empty_cache()


def should_add_pad_token(model_id: str) -> bool:
  """Returns whether the model requires adding a special pad token."""
  return any(s.lower() in model_id.lower() for s in _MODELS_REQUIRING_PAD_TOKEN)


def should_add_eos_token(model_id: str) -> bool:
  """Returns whether the model requires adding a special eos token."""
  return any(m in model_id for m in _MODELS_REQUIRING_EOS_TOEKN)


def write_kfp_outputs(
    executor_input: str, output_artifacts: Dict[str, str]
) -> None:
  """Writes KFP outputs given a dict of output artifact names and URIs."""
  # Only the main process writes to avoid race condition.
  if PartialState().is_main_process:
    executor_input = json_format.Parse(
        executor_input, pipeline_spec_pb2.ExecutorInput()
    )
    outputs = executor_input.outputs
    # set all artifacts
    for name, uri in output_artifacts.items():
      artifact_list = outputs.artifacts.get(name)
      if not artifact_list or not artifact_list.artifacts:
        raise ValueError(f"Artifact name={name} does not exist.")
      artifact_list.artifacts[0].uri = uri

    # write output file
    executor_output = pipeline_spec_pb2.ExecutorOutput(
        artifacts=outputs.artifacts
    )
    os.makedirs(os.path.dirname(outputs.output_file), exist_ok=True)
    with open(outputs.output_file, "w") as f:
      f.write(json_format.MessageToJson(executor_output, indent=None))

  # Wait for the main process to finish before moving on to the next task.
  PartialState().wait_for_everyone()


def upload_local_dir_to_gcs(local_dir: str, gcs_path: str):
  """Uploads local dir to GCS."""

  if PartialState().is_main_process:
    logging.info("uploading %s to %s...", local_dir, gcs_path)
    subprocess.check_output([
        "gsutil",
        "-m",
        "cp",
        "-r",
        local_dir,
        gcs_path,
    ])
    logging.info("%s uploaded.", local_dir)

  PartialState().wait_for_everyone()


def write_first_party_model_metadata(output_dir: str, docker_uri: str) -> None:
  """Multi-process friendly version of fileutils.write_first_party_model_metadata."""
  if PartialState().is_main_process:
    fileutils.write_first_party_model_metadata(output_dir, docker_uri)
  PartialState().wait_for_everyone()


@dataclasses.dataclass
class GpuStats:
  """Holds information about GPU usage stats.

  For memory related, see
  https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management
  """

  # total memory
  total_mem: float
  # memory occupied.
  occupied: float
  # memory reserved, but not used.
  unused: float
  # nvidia-smi usually reports more memory usages than pytorch (for driver,
  # kernel and etc). `smi_diff` tracks this difference.
  smi_diff: float
  # Gpu utilization.
  util: float

  # Allows unpacking operation like
  # total_mem, occupied, unused, smi_diff, util = GpuStats(...)
  # See https://stackoverflow.com/a/70753113
  def __iter__(self):
    return iter(dataclasses.astuple(self))


def gpu_stats() -> GpuStats:
  """Reports GPU memory usage and utilization."""
  # See https://pytorch.org/docs/stable/notes/cuda.html#memory-management
  bytes_per_gb = 1024.0**3
  device = torch.cuda.current_device()
  occupied = torch.cuda.memory_allocated(device) / bytes_per_gb
  reserved = torch.cuda.memory_reserved(device) / bytes_per_gb
  unused = reserved - occupied

  def smi_mem(device):
    try:
      pynvml.nvmlInit()
      handle = pynvml.nvmlDeviceGetHandleByIndex(device)
      info = pynvml.nvmlDeviceGetMemoryInfo(handle)
      return info.used / bytes_per_gb
    except pynvml.NVMLError:
      return 0.0

  mem_used_smi = smi_mem(device)
  smi_diff = mem_used_smi - reserved

  util = torch.cuda.utilization(device)
  return GpuStats(mem_used_smi, occupied, unused, smi_diff, util)


def gpu_stats_str(stats: Optional[GpuStats] = None) -> str:
  if stats is None:
    stats = gpu_stats()
  total, occupied, unused, smi_diff, util = stats
  return (
      f"GPU memory: {total:.2f}({occupied=:.2f}, {unused=:.2f},"
      f" {smi_diff=:.2f}) GB. Utilization: {util:.2f}%"
  )


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
) -> Optional[Sequence[str]]:
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
