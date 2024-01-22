"""Common utility functions for setting up and initializing the model and the handler."""

import logging
import os
from typing import Any

import torch

from util import constants
from util import fileutils


def get_model_id(default_model_id: str) -> str:
  """Gets a model id or a local model path.

  Args:
    default_model_id: Default model id for the corresponding model set in the
      handler.

  Returns:
    str: model id or a local model path.
  """
  # The model id can be either:
  # 1) a huggingface model card id, like "Salesforce/blip", or
  # 2) a GCS path to the model files, like "gs://foo/bar".
  # If it's a model card id, the model will be loaded from huggingface.
  model_id = (
      default_model_id
      if os.environ.get("MODEL_ID") is None
      else os.environ["MODEL_ID"]
  )

  # Else it will be downloaded from GCS to local first.
  # Since the transformers from_pretrained API can't read from GCS.
  if model_id.startswith(constants.GCS_URI_PREFIX):
    gcs_path = model_id[len(constants.GCS_URI_PREFIX) :]
    local_model_dir = os.path.join(constants.LOCAL_MODEL_DIR, gcs_path)
    logging.info("Download %s to %s", model_id, local_model_dir)
    fileutils.download_gcs_dir_to_local(model_id, local_model_dir)
    model_id = local_model_dir

  return model_id


def get_map_location(context: Any) -> str:
  """Gets model map location.

  Args:
    context: Torchserve worker context.

  Returns:
    str: Mapping location.
  """
  properties = context.system_properties
  return (
      "cuda"
      if torch.cuda.is_available() and properties.get("gpu_id") is not None
      else "cpu"
  )


def get_model_device(map_location: str, context: Any) -> torch.device:
  """Gets model accelerator device.

  Args:
    map_location: Model map location.
    context: TorchServe worker context.

  Returns:
    torch.Device: Device to load the model into.
  """
  properties = context.system_properties
  return torch.device(
      map_location + ":" + str(properties.get("gpu_id"))
      if torch.cuda.is_available() and properties.get("gpu_id") is not None
      else map_location
  )
