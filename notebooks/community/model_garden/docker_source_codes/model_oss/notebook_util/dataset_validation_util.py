"""Functions for dataset validation.

This tool is used to validate the dataset against the given template.
"""

from collections.abc import Callable
import json
import multiprocessing
import os
import subprocess
import sys
from typing import Any, Union
from absl import logging
import accelerate
import datasets
import transformers

GCS_URI_PREFIX = "gs://"
GCSFUSE_URI_PREFIX = "/gcs/"
LOCAL_BASE_MODEL_DIR = "/tmp/base_model_dir"
LOCAL_TEMPLATE_DIR = "/tmp/template_dir"
_TEMPLATE_DIRNAME = "templates"
_VERTEX_AI_SAMPLES_GITHUB_REPO_NAME = "vertex-ai-samples"
_VERTEX_AI_SAMPLES_GITHUB_TEMPLATE_DIR = (
    "community-content/vertex_model_garden/model_oss/peft/train/vmg/templates"
)
_MODELS_REQUIRING_PAD_TOKEN = ("llama", "falcon", "mistral", "mixtral")
_MODELS_REQUIRING_EOS_TOEKN = ("gemma-2b", "gemma-7b")
_DESCRIPTION_KEY = "description"
_SOURCE_KEY = "source"
_PROMPT_INPUT_KEY = "prompt_input"
_PROMPT_NO_INPUT_KEY = "prompt_no_input"
_RESPONSE_SEPARATOR = "response_separator"
_INSTRUCTION_SEPARATOR = "instruction_separator"
_CHAT_TEMPLATE_KEY = "chat_template"
_KNOWN_KEYS = (
    _DESCRIPTION_KEY,
    _SOURCE_KEY,
    _PROMPT_INPUT_KEY,
    _PROMPT_NO_INPUT_KEY,
    _RESPONSE_SEPARATOR,
    _INSTRUCTION_SEPARATOR,
    _CHAT_TEMPLATE_KEY,
)


def is_gcs_path(input_path: str) -> bool:
  """Checks if the input path is a Google Cloud Storage (GCS) path.

  Args:
      input_path: The input path to be checked.

  Returns:
      True if the input path is a GCS path, False otherwise.
  """
  return input_path is not None and input_path.startswith(GCS_URI_PREFIX)


def force_gcs_fuse_path(gcs_uri: str) -> str:
  """Converts gs:// uris to their /gcs/ equivalents. No-op for other uris.

  Args:
    gcs_uri: The GCS URI to convert.

  Returns:
    The converted GCS URI.
  """
  if is_gcs_path(gcs_uri):
    return GCSFUSE_URI_PREFIX + gcs_uri[len(GCS_URI_PREFIX) :]
  else:
    return gcs_uri


def download_gcs_uri_to_local(
    gcs_uri: str,
    destination_dir: str = LOCAL_BASE_MODEL_DIR,
    check_path_exists: bool = True,
) -> str:
  """Downloads GCS URI to local.

  If GCS URI is a directory, gs://some/folder is downloaded to
  /destination_dir/folder. If GCS URI is a file, gs://some/file is downloaded to
  /destination_dir/file.

  Args:
    gcs_uri: GCS URI to download.
    destination_dir: Local directory directory.
    check_path_exists: Whether to check if the path exists.

  Returns:
    Local path to target folder/file.
  """
  target = os.path.join(
      destination_dir,
      os.path.basename(os.path.normpath(gcs_uri)),
  )
  if check_path_exists and os.path.exists(target):
    logging.info("File %s already exists.", target)
    return target
  if accelerate.PartialState().is_local_main_process:
    logging.info(
        "Downloading file(s) from %s to %s...", gcs_uri, destination_dir
    )
    if not os.path.exists(destination_dir):
      os.mkdir(destination_dir)
    subprocess.check_output([
        "gsutil",
        "-m",
        "cp",
        "-r",
        gcs_uri,
        destination_dir,
    ])
    logging.info("Downloaded file(s) from %s to %s.", gcs_uri, destination_dir)
  # Make sure ALL processes process to next step after data downloading is done.
  # It matters for the main process to wait for other processes as well.
  accelerate.PartialState().wait_for_everyone()
  return target


def get_template(template_path: str) -> dict[str, str]:
  """Gets the template dictionary given the file path.

  Args:
    template_path: Path to the template file.

  Returns:
    A dictionary of the template.

  Raises:
    ValueError: If the template file does not exist or contains unknown keys.
  """
  if is_gcs_path(template_path):
    template_path = force_gcs_fuse_path(template_path)
  elif not os.path.isfile(template_path):
    template_path = os.path.join(
        os.path.dirname(__file__),
        _TEMPLATE_DIRNAME,
        template_path + ".json",
    )
  if not os.path.isfile(template_path):
    raise ValueError(f"Template file {template_path} does not exist.")
  with open(template_path, "r") as f:
    template_json: dict[str, str] = json.load(f)
    for key in template_json:
      if key not in _KNOWN_KEYS:
        raise ValueError(f"Unknown key {key} in template {template_path}.")
    return template_json


def get_response_separator(template_json: dict[str, str]) -> Union[str, None]:
  return template_json.get(_RESPONSE_SEPARATOR, None)


def get_instruction_separator(
    template_json: dict[str, str],
) -> Union[str, None]:
  return template_json.get(_INSTRUCTION_SEPARATOR, None)


def _format_template_fn(
    template: str,
    input_column: str,
    tokenizer: transformers.PreTrainedTokenizer | None = None,
) -> Callable[[dict[str, str]], dict[str, str]]:
  """Formats a dataset example according to a template.

  Args:
    template: Name of the JSON template file under `templates/` or GCS path to
      the template file.
    input_column: The input column in the dataset to be used or updated by the
      template. If it does not exist, the template's `prompt_no_input` will be
      used, and the input_column will be created.
    tokenizer: The tokenizer to use for chat_template templates.

  Returns:
    A function that formats data according to the template.
  """
  template_json = get_template(template)

  if _CHAT_TEMPLATE_KEY not in template_json:

    def format_fn(example: dict[str, str]) -> dict[str, str]:
      format_dict = {key: value for key, value in example.items()}
      if format_dict.get(input_column):
        format_str = template_json[_PROMPT_INPUT_KEY]
      elif _PROMPT_NO_INPUT_KEY in template_json:
        format_str = template_json[_PROMPT_NO_INPUT_KEY]
      else:
        raise KeyError(
            f"The template {os.path.basename(template)} does not contain"
            f" {_PROMPT_INPUT_KEY} or {_PROMPT_NO_INPUT_KEY} key."
        )
      try:
        return {input_column: format_str.format(**format_dict)}
      except KeyError as e:
        raise KeyError(
            f"The template {os.path.basename(template)} contains a key {e} in"
            f" {_PROMPT_INPUT_KEY} or {_PROMPT_NO_INPUT_KEY} that does not"
            " exist in the dataset example. The dataset example looks like"
            f" {format_dict}."
        ) from e

    return format_fn
  elif (
      _PROMPT_INPUT_KEY in template_json
      or _PROMPT_NO_INPUT_KEY in template_json
  ):
    raise ValueError(
        f"chat_template templates do not support {_PROMPT_INPUT_KEY} or"
        f" {_PROMPT_NO_INPUT_KEY} templates."
    )
  else:
    if tokenizer is None:
      raise ValueError("A tokenizer is required for chat_template templates.")
    # Assign HuggingFace jinja template.
    tokenizer.chat_template = template_json[_CHAT_TEMPLATE_KEY]

    def format_fn(example: dict[str, str]) -> dict[str, str]:
      try:
        return {
            input_column: tokenizer.apply_chat_template(
                example[input_column],
                tokenize=False,
                add_generation_prompt=False,
            )
        }
      except KeyError as e:
        raise KeyError(
            f"The template {os.path.basename(template)} contains a key {e} in"
            f" {_CHAT_TEMPLATE_KEY} that does not exist in the dataset example."
        ) from e

    return format_fn


def _get_split_string(
    split: str,
    dataset_percent: int | None = None,
    dataset_k_rows: int | None = None,
) -> str:
  """Gets the formatted split string for the dataset.

  This is used to format the split string as per
  https://huggingface.co/docs/datasets/v2.21.0/loading#slice-splits. Also, this
  function will only be used to load the partial dataset for validating the
  dataset against the template.

  Args:
    split: Split of the dataset.
    dataset_percent: The percentage of the dataset to load.
    dataset_k_rows: The top k sequences to load from the dataset.

  Returns:
    A formatted split string.
  """
  # Validate the dataset_percent and dataset_k_rows values.
  if dataset_percent and dataset_k_rows:
    raise ValueError(
        "You can set either validate_percentage_of_dataset or"
        " validate_k_rows_of_dataset, but not both."
    )

  if dataset_percent:
    logging.info("Loading %d percent of the dataset...", dataset_percent)
    return f"{split}[:{dataset_percent}%]"

  if dataset_k_rows:
    logging.info("Loading top %d rows of the dataset...", dataset_k_rows)
    return f"{split}[:{dataset_k_rows}]"

  return split


def _github_template_path(template: str) -> str:
  """Generates the path to the template in the Vertex AI Samples GitHub repo.

  Args:
    template: Name of the template.

  Returns:
    The path to the template in the Vertex AI Samples GitHub repo.
  """
  # vertex-ai-samples directory may lie under separate directory depending on
  # the scratch_dir parameter in the notebook execution environment.
  vertex_ai_samples_abs_path = os.getcwd().split(
      _VERTEX_AI_SAMPLES_GITHUB_REPO_NAME
  )[0]
  return os.path.join(
      vertex_ai_samples_abs_path,
      _VERTEX_AI_SAMPLES_GITHUB_REPO_NAME,
      _VERTEX_AI_SAMPLES_GITHUB_TEMPLATE_DIR,
      template + ".json",
  )


def _get_dataset(
    dataset_name: str,
    split: str,
    num_proc: int | None = None,
) -> datasets.DatasetDict:
  """Gets a dataset.

  Args:
    dataset_name: Name of the dataset or path to a custom dataset.
    split: Split of the dataset.
    num_proc: Number of processors to use.

  Returns:
    A dataset.
  """
  dataset_name = force_gcs_fuse_path(dataset_name)
  if os.path.isfile(dataset_name):
    # Custom dataset.
    return datasets.load_dataset(
        "json",
        data_files=[dataset_name],
        split=split,
        num_proc=num_proc,
    )
  # HF dataset.
  return datasets.load_dataset(dataset_name, split=split, num_proc=num_proc)


def should_add_pad_token(model_id: str) -> bool:
  """Returns whether the model requires adding a special pad token.

  Args:
    model_id: The name of the model.

  Returns:
    True if the model requires adding a special pad token, False otherwise.
  """
  model_config = transformers.AutoConfig.from_pretrained(model_id)
  if model_config.model_type is None:
    return False
  return any(
      s.lower() in model_config.model_type.lower()
      for s in _MODELS_REQUIRING_PAD_TOKEN
  )


def should_add_eos_token(model_id: str) -> bool:
  """Returns whether the model requires adding a special eos token.

  Args:
    model_id: The name of the model.

  Returns:
    True if the model requires adding a special eos token, False otherwise.
  """
  return any(m in model_id for m in _MODELS_REQUIRING_EOS_TOEKN)


def load_tokenizer(
    pretrained_model_id: str,
    padding_side: str | None = None,
    access_token: str | None = None,
) -> transformers.AutoTokenizer:
  """Loads tokenizer based on `pretrained_model_id`.

  Args:
    pretrained_model_id: The name of the pretrained model.
    padding_side: The side to pad the input on.
    access_token: The access token to use for the tokenizer.

  Returns:
    The tokenizer.
  """
  tokenizer_kwargs = {}
  if should_add_eos_token(pretrained_model_id):
    tokenizer_kwargs["add_eos_token"] = True
  if padding_side:
    tokenizer_kwargs["padding_side"] = padding_side

  with accelerate.PartialState().local_main_process_first():
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_id,
        trust_remote_code=False,
        use_fast=True,
        token=access_token,
        **tokenizer_kwargs,
    )

  if should_add_pad_token(pretrained_model_id):
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

  return tokenizer


def _get_indices_for_valid_length(
    dataset: Any,
    input_column: str,
    max_sequence_length: int,
    tokenizer: transformers.PreTrainedTokenizer,
    context_name: str = "the dataset",
) -> tuple[list[int], int, int]:
  """Gets indices of examples shorter than or equal to max_seq_length.

  Args:
    dataset: The dataset to check.
    input_column: The input column in the dataset.
    max_sequence_length: The maximum sequence length.
    tokenizer: The tokenizer.
    context_name: A name for the dataset used in log messages.

  Returns:
    A tuple of (indices_to_keep, original_length, dropped_samples).
  """
  if not dataset:
    return [], 0, 0

  original_length = len(dataset)
  indices_to_keep = [
      i
      for i, entry in enumerate(dataset)
      if len(tokenizer(entry[input_column])["input_ids"]) <= max_sequence_length
  ]
  dropped_samples = original_length - len(indices_to_keep)

  if dropped_samples > 0:
    examples_removed_percent = (dropped_samples * 100) / original_length
    logging.info(
        "(%.2f%%) of examples token length is <= max-seq-length(%d); (%.2f%%) >"
        " max-seq-length in %s. %d example(s) were longer than max-seq-length.",
        100 - examples_removed_percent,
        max_sequence_length,
        examples_removed_percent,
        context_name,
        dropped_samples,
    )
  else:
    logging.info(
        "No samples were dropped from %s because all samples are"
        " shorter than max_sequence_length=%d.",
        context_name,
        max_sequence_length,
    )
  return indices_to_keep, original_length, dropped_samples


def get_filtered_dataset(
    dataset: Any,
    input_column: str,
    max_seq_length: int,
    tokenizer: transformers.PreTrainedTokenizer,
    example_removed_threshold: float = 50.0,
) -> Any:
  """Returns the dataset by removing examples that are longer than max_seq_length.

  Args:
    dataset: The dataset to filter.
    input_column: The input column in the dataset to be used.
    max_seq_length: The maximum sequence length.
    tokenizer: The tokenizer.
    example_removed_threshold: The percent threshold for the number of examples
      removed from the dataset. It should be in the range of [0, 100].

  Returns:
    The filtered dataset.

  Raises:
    ValueError: If more than `example_removed_threshold` of the dataset is
      filtered out.
  """
  indices_to_keep, original_length, dropped_samples = (
      _get_indices_for_valid_length(
          dataset, input_column, max_seq_length, tokenizer, "the dataset"
      )
  )

  if (
      original_length > 0
      and dropped_samples / original_length * 100 > example_removed_threshold
  ):
    examples_removed_percent = (dropped_samples * 100) / original_length
    raise ValueError(
        f"More than {examples_removed_percent:.2f}% of the dataset is filtered"
        " out. This may be due to small value of"
        f" max-seq-length({max_seq_length}) or incorrect template. Please"
        " increase the max-seq-length or check the template."
    )

  filtered_dataset = dataset.select(indices_to_keep)
  print(f"Some formatted examples from the dataset are: {filtered_dataset[:5]}")
  return filtered_dataset


def format_dataset(
    dataset: datasets.Dataset,
    input_column: str,
    template: str = None,
    tokenizer: transformers.PreTrainedTokenizer | None = None,
) -> datasets.Dataset:
  """Takes a raw dataset and formats it using a template and tokenizer.

  Args:
    dataset: The raw (unprocessed) dataset to format.
    input_column: The input column in the dataset to be used or updaded by the
      template. If it does not exist, the template's `prompt_no_input` will be
      used, and the input_column will be created.
    template: Name of the JSON template file under `templates/` or GCS path to
      the template file.
    tokenizer: The tokenizer to use for chat_template templates.

  Returns:
    A dataset compatible with the template.
  """
  return dataset.map(
      _format_template_fn(
          template,
          input_column=input_column,
          tokenizer=tokenizer,
      )
  )


def load_dataset_with_template(
    dataset_name: str,
    split: str,
    input_column: str,
    template: str = None,
    tokenizer: transformers.PreTrainedTokenizer | None = None,
) -> tuple[Any, Any]:
  """Loads dataset with templates.

  Args:
    dataset_name: Name of the dataset or path to a custom dataset.
    split: Split of the dataset.
    input_column: The input column in the dataset to be used or updaded by the
      template. If it does not exist, the template's `prompt_no_input` will be
      used, and the input_column will be created.
    template: Name of the JSON template file under `templates/` or GCS path to
      the template file.
    tokenizer: The tokenizer to use for chat_template templates.

  Returns:
    The raw dataset and the dataset compatible with the template.
  """
  raw = _get_dataset(dataset_name, split=split)
  if template:
    templated = format_dataset(raw, input_column, template, tokenizer)
  else:
    templated = None

  return raw, templated


def drop_long_sequences(
    dataset: Any,
    dataset_with_template: Any,
    input_column: str,
    max_sequence_length: int,
    tokenizer: transformers.PreTrainedTokenizer,
    dataset_dropped_threshold: float,
    is_train: bool,
) -> tuple[Any, Any, int]:
  """Drops examples longer than max_seq_length from the dataset.

  Args:
    dataset: The dataset to filter.
    dataset_with_template: The dataset with template to filter.
    input_column: The input column in the dataset to be used.
    max_sequence_length: The maximum sequence length.
    tokenizer: The tokenizer.
    dataset_dropped_threshold: The threshold for the number of samples dropped
      from the dataset.
    is_train: Whether the dataset is for training.

  Returns:
    A tuple of (filtered_dataset, filtered_dataset_with_template,
    dropped_samples).
  """

  context_name = f"the {'train' if is_train else 'eval'} dataset"
  indices_to_keep, original_length, dropped_samples = (
      _get_indices_for_valid_length(
          dataset_with_template,
          input_column,
          max_sequence_length,
          tokenizer,
          context_name,
      )
  )

  samples_removed_percent = (
      (dropped_samples / original_length) * 100 if original_length > 0 else 0.0
  )
  if (
      original_length > 0
      and samples_removed_percent > dataset_dropped_threshold
  ):
    logging.error(
        "%.2f%% of the samples were dropped from %s after filtering for"
        " max_sequence_length=%d. The threshold is %.2f%%.",
        samples_removed_percent,
        context_name,
        max_sequence_length,
        dataset_dropped_threshold,
    )
    
    # handling library when available.
    sys.exit(1)

  filtered_dataset = dataset.select(indices_to_keep)
  filtered_dataset_with_template = dataset_with_template.select(indices_to_keep)
  return filtered_dataset, filtered_dataset_with_template, dropped_samples


def validate_dataset_with_template(
    dataset_name: str,
    split: str,
    input_column: str,
    template: str,
    tokenizer: transformers.PreTrainedTokenizer | None = None,
    max_seq_length: int | None = None,
    use_multiprocessing: bool = False,
    validate_percentage_of_dataset: int | None = None,
    validate_k_rows_of_dataset: int | None = None,
    example_removed_threshold: float = 50.0,
) -> Any:
  """Validates dataset with templates.

  This function will be used to load the dataset and validate it against the
  template. In case of validation, we also allow the users to load the dataset
  partially by allowing them to read x% or top k rows of the dataset. To
  validate the dataset, the template file must be available in the GCS bucket
  and the dataset must be available either in the GCS bucket or Hugging Face.

  Args:
    dataset_name: Name of the dataset or path to a custom dataset.
    split: Split of the dataset.
    input_column: The input column in the dataset to be used or updaded by the
      template. If it does not exist, the template's `prompt_no_input` will be
      used, and the input_column will be created.
    template: Name of the JSON template file under `templates/` or GCS path to
      the template file.
    tokenizer: The tokenizer to use for chat_template templates.
    max_seq_length: The maximum sequence length.
    use_multiprocessing: If True, it will use multiprocessing to load the
      dataset.
    validate_percentage_of_dataset: The percentage of the dataset to load.
    validate_k_rows_of_dataset: The top k sequences to load from the dataset.
    example_removed_threshold: The threshold for the number of examples removed
      from the dataset.

  Returns:
    None if the validation is successful, otherwise returns the error message.
  """
  if not template:
    raise ValueError("template is required for validate_dataset.")

  if not dataset_name:
    raise ValueError("dataset_name is empty.")

  if not split:
    raise ValueError("split is empty.")

  split = _get_split_string(
      split,
      validate_percentage_of_dataset,
      validate_k_rows_of_dataset,
  )

  num_proc = multiprocessing.cpu_count() if use_multiprocessing else 1

  # gcsfuse cannot be used from the notebook runtime env. Hence, we have
  # to download dataset and template from gcs to local.
  if is_gcs_path(dataset_name):
    dataset_name = download_gcs_uri_to_local(dataset_name, LOCAL_BASE_MODEL_DIR)

  if is_gcs_path(template):
    template_path = download_gcs_uri_to_local(template, LOCAL_TEMPLATE_DIR)
  elif os.path.isfile(_github_template_path(template)):
    template_path = _github_template_path(template)
  else:
    raise ValueError(
        f"Template file {template} does not exist. To validate the"
        " dataset, please provide a valid GCS path for the template or a valid"
        " template name from"
        f" https://github.com/GoogleCloudPlatform/{_VERTEX_AI_SAMPLES_GITHUB_REPO_NAME}/tree/main/{_VERTEX_AI_SAMPLES_GITHUB_TEMPLATE_DIR}."
    )

  dataset = format_dataset(
      _get_dataset(dataset_name, split, num_proc),
      input_column,
      template_path,
      tokenizer,
  )

  if tokenizer is not None:
    get_filtered_dataset(
        dataset=dataset,
        input_column=input_column,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        example_removed_threshold=example_removed_threshold,
    )
  print(
      "Dataset {} is compatible with the {} template.".format(
          os.path.basename(dataset_name), os.path.basename(template)
      )
  )
