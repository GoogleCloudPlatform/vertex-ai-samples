"""Functions for dataset validation.

This tool is used to validate the dataset against the given template.
"""

import json
import multiprocessing
import os
import subprocess
from typing import Any, Callable, Dict, Union
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
    gcs_uri: str, destination_dir: str = LOCAL_BASE_MODEL_DIR
) -> str:
  """Downloads GCS URI to local.

  If GCS URI is a directory, gs://some/folder is downloaded to
  /destination_dir/folder. If GCS URI is a file, gs://some/file is downloaded to
  /destination_dir/file.

  Args:
    gcs_uri: GCS URI to download.
    destination_dir: Local directory directory.

  Returns:
    Local path to target folder/file.
  """
  target = os.path.join(
      destination_dir,
      os.path.basename(os.path.normpath(gcs_uri)),
  )
  if os.path.exists(target):
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


def get_template(template_path: str) -> Dict[str, str]:
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


def get_response_separator(template_json: Dict[str, str]) -> Union[str, None]:
  return template_json.get(_RESPONSE_SEPARATOR, None)


def get_instruction_separator(
    template_json: Dict[str, str],
) -> Union[str, None]:
  return template_json.get(_INSTRUCTION_SEPARATOR, None)


def _format_template_fn(
    template: str,
    input_column: str,
    tokenizer: transformers.PreTrainedTokenizer | None = None,
) -> Callable[[Dict[str, str]], Dict[str, str]]:
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

    def format_fn(example: Dict[str, str]) -> Dict[str, str]:
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

    def format_fn(example: Dict[str, str]) -> Dict[str, str]:
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
  return any(s.lower() in model_id.lower() for s in _MODELS_REQUIRING_PAD_TOKEN)


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


def get_filtered_dataset(
    dataset: Any,
    input_column: str,
    max_seq_length: int,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Any:
  """Returns the dataset by removing examples that are longer than max_seq_length.

  Args:
    dataset: The dataset to filter.
    input_column: The input column in the dataset to be used.
    max_seq_length: The maximum sequence length.
    tokenizer: The tokenizer.
  """
  actual_dataset_length = len(dataset)
  filtered_dataset = dataset.filter(
      lambda x: len(tokenizer(x[input_column])["input_ids"]) <= max_seq_length
  )
  filtered_dataset_length = len(filtered_dataset)
  if actual_dataset_length != filtered_dataset_length:
    examples_removed_percent = (
        (actual_dataset_length - filtered_dataset_length)
        * 100
        / actual_dataset_length
    )
    logging.info(
        "(%.2f%%) of examples token length is <= max-seq-length(%d); (%.2f%%) >"
        " max-seq-length. Filtering out %d example(s) which are longer than"
        " max-seq-length.",
        100 - examples_removed_percent,
        max_seq_length,
        examples_removed_percent,
        actual_dataset_length - filtered_dataset_length,
    )

  return filtered_dataset


def load_dataset_with_template(
    dataset_name: str,
    split: str,
    input_column: str,
    template: str = None,
    tokenizer: transformers.PreTrainedTokenizer | None = None,
) -> Any:
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
    A dataset compatible with the template.
  """
  dataset = _get_dataset(dataset_name, split=split)
  if template:
    dataset = dataset.map(
        _format_template_fn(
            template,
            input_column=input_column,
            tokenizer=tokenizer,
        )
    )

  return dataset


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

  dataset = _get_dataset(dataset_name, split, num_proc).map(
      _format_template_fn(
          template_path,
          input_column=input_column,
          tokenizer=tokenizer,
      )
  )

  if tokenizer is not None:
    get_filtered_dataset(
        dataset=dataset,
        input_column=input_column,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
    )
  print(
      "Dataset {} is compatible with the {} template.".format(
          os.path.basename(dataset_name), os.path.basename(template)
      )
  )
