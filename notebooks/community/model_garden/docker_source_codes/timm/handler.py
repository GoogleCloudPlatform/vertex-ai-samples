"""Custom handler for TIMM models."""

import logging
import os
from typing import Any

from google.cloud import storage
import timm
import torch
from ts.torch_handler.base_handler import load_label_mapping
from ts.torch_handler.image_classifier import ImageClassifier


GCS_PREFIX = "gs://"
DOWNLOAD_DIR = "/tmp/download"


def download_gcs_file(gcs_uri: str, local_dir: str) -> str:
  """Download a GCS file to a local directory.

  Arguments:
    gcs_uri: A string of file path on GCS.
    local_dir: A string of local directory path.

  Returns:
    Local path to downloaded file.
  """
  if not gcs_uri.startswith(GCS_PREFIX):
    raise ValueError(f"{gcs_uri} is not a GCS path starting with gs://.")

  file_name = os.path.basename(gcs_uri)
  local_file_path = os.path.join(local_dir, file_name)
  os.makedirs(local_dir, exist_ok=True)
  client = storage.Client()
  with open(local_file_path, "wb") as f:
    client.download_blob_to_file(gcs_uri, f)
  return local_file_path


class TimmHandler(ImageClassifier):
  """Custom handler for TIMM models."""

  def initialize(self, context: Any):
    """Custom initialize."""

    properties = context.system_properties
    self.map_location = (
        "cuda"
        if torch.cuda.is_available() and properties.get("gpu_id") is not None
        else "cpu"
    )
    self.device = torch.device(
        self.map_location + ":" + str(properties.get("gpu_id"))
        if torch.cuda.is_available() and properties.get("gpu_id") is not None
        else self.map_location
    )
    self.manifest = context.manifest

    # Load timm model by model name.
    self.model_name = os.environ["MODEL_NAME"]
    # Whether to use timm pretrained weights, MODEL_PT_PATH overrides this.
    timm_pretrained = True if os.environ.get("TIMM_PRETRAINED") else False
    # Load custom checkpoint, it overrides TIMM_PRETRAINED model.
    self.model_pt_path = os.environ.get("MODEL_PT_PATH")
    if self.model_pt_path and self.model_pt_path.startswith(GCS_PREFIX):
      self.model_pt_path = download_gcs_file(self.model_pt_path, DOWNLOAD_DIR)

    if self.model_pt_path and self.model_pt_path.endswith(".pt"):
      logging.info(
          "Load model with .pt in jit mode, not working for all timm models"
          " yet."
      )
      self.model = self._load_torchscript_model(self.model_pt_path)
    else:
      logging.info("Load model with .pth in eager mode.")
      self.model = timm.create_model(
          self.model_name, pretrained=timm_pretrained
      )
      if self.model_pt_path and (
          self.model_pt_path.endswith(".pth")
          or self.model_pt_path.endswith(".pth.tar")
      ):
        checkpoint = torch.load(self.model_pt_path, map_location=self.device)
        state_dict = checkpoint["state_dict"]
        self.model.load_state_dict(state_dict)
      self.model.to(self.device)
    self.model.eval()

    mapping_file_path = os.environ.get("INDEX_TO_NAME_FILE")
    if mapping_file_path:
      if mapping_file_path.startswith(GCS_PREFIX):
        mapping_file_path = download_gcs_file(mapping_file_path, DOWNLOAD_DIR)
      self.mapping = load_label_mapping(mapping_file_path)

    self.initialized = True

  # NOTE: Preprocess and postprocess are implemented by ImageClassifier.