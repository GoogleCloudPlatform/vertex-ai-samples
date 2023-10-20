"""Custom handler for SAM huggingface/transformers models."""

import base64
import io
import logging
import os
from typing import Any, List, Optional, Tuple

from google.cloud import storage
import numpy as np
from PIL import Image
import pycocotools.mask as mask_util
import torch
from transformers import pipeline
from ts.torch_handler.base_handler import BaseHandler


SAM_VIT_BASE = "facebook/sam-vit-base"
SAM_VIT_LARGE = "facebook/sam-vit-large"
SAM_VIT_HUGE = "facebook/sam-vit-huge"

DEFAULT_MODEL_ID = "facebook/sam-vit-large"
MASK_GENERATION = "mask-generation"

GCS_PREFIX = "gs://"
DOWNLOAD_DIR = "/tmp/download"


def is_gcs_path(input_path: str) -> bool:
  return input_path.startswith(GCS_PREFIX)


def download_gcs_dir(gcs_dir: str, local_dir: str):
  """Download files in a GCS directory to a local directory.

  For example:
    download_gcs_dir(gs://bucket/foo, /tmp/bar)
    gs://bucket/foo/a -> /tmp/bar/a
    gs://bucket/foo/b/c -> /tmp/bar/b/c

  Arguments:
    gcs_dir: A string of directory path on GCS.
    local_dir: A string of local directory path.
  """
  if not is_gcs_path(gcs_dir):
    raise ValueError(f"{gcs_dir} is not a GCS path starting with gs://.")

  bucket_name = gcs_dir.split("/")[2]
  prefix = gcs_dir[len(GCS_PREFIX + bucket_name) :].strip("/")
  client = storage.Client()
  blobs = client.list_blobs(bucket_name, prefix=prefix)
  for blob in blobs:
    if blob.name[-1] == "/":
      continue
    file_path = blob.name[len(prefix) :].strip("/")
    local_file_path = os.path.join(local_dir, file_path)
    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
    blob.download_to_filename(local_file_path)


class TransformersHandler(BaseHandler):
  """Custom handler for huggingface/transformers models."""

  def initialize(self, context: Any):
    """Custom initialize."""

    # vv-docker:google3-begin(internal)
    # TODO(b/287051908): Move handler functions to common utils for
    # everyone to use.
    # vv-docker:google3-end
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

    # The model id is can be either:
    # 1) a huggingface model card id, like "Salesforce/blip", or
    # 2) a GCS path to the model files, like "gs://foo/bar".
    # If it's a model card id, the model will be loaded from huggingface.
    self.model_id = (
        DEFAULT_MODEL_ID
        if os.environ.get("MODEL_ID") is None
        else os.environ["MODEL_ID"]
    )
    # Else it will be downloaded from GCS to local first.
    # Since the transformers from_pretrained API can't read from GCS.
    if self.model_id.startswith(GCS_PREFIX):
      gcs_path = self.model_id[len(GCS_PREFIX) :]
      local_model_dir = os.path.join(DOWNLOAD_DIR, gcs_path)
      logging.info(f"Download {self.model_id} to {local_model_dir}")
      download_gcs_dir(self.model_id, local_model_dir)
      self.model_id = local_model_dir

    self.task = (
        MASK_GENERATION
        if os.environ.get("TASK") is None
        else os.environ["TASK"]
    )
    logging.info(
        f"Handler initializing task:{self.task}, model:{self.model_id}"
    )

    if self.task == MASK_GENERATION:
      self.pipeline = pipeline(
          task="mask-generation", model=self.model_id, device=self.device
      )
    else:
      raise ValueError(f"Invalid TASK: {self.task}")

    self.initialized = True
    logging.info("Handler initialization done.")

  def _image_to_base64(self, image: Image.Image) -> str:
    """Convert a PIL image to a base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return image_str

  def _base64_to_image(self, image_str: str) -> Image.Image:
    """Convert a base64 string to a PIL image."""
    image = Image.open(io.BytesIO(base64.b64decode(image_str)))
    return image

  def preprocess(
      self, data: Any
  ) -> Tuple[Optional[List[str]], Optional[List[Image.Image]]]:
    """Preprocess input data."""
    texts = None
    images = None
    if "point" in data[0]:
      texts = [item["point"] for item in data]
    if "image" in data[0]:
      images = [self._base64_to_image(item["image"]) for item in data]
    return texts, images

  def inference(self, data: Any, *args, **kwargs) -> List[Any]:
    """Run the inference."""
    _, images = data
    preds = []
    for img in images:
      if self.task == MASK_GENERATION:
        outputs = self.pipeline(img, points_per_batch=64)
        masks = np.array([m.tolist() for m in outputs["masks"]])
        preds.append(masks)
      else:
        raise ValueError(f"Invalid TASK: {self.task}")
    return preds

  def handle(self, data: Any, context: Any) -> List[Any]:  # pylint: disable=unused-argument
    """Runs preprocess, inference, and post-processing."""
    model_input = self.preprocess(data)
    model_out = self.inference(model_input)
    output = self.postprocess(model_out)
    return output

  def postprocess(self, inference_result: List[Any]) -> List[Any]:
    """Post process inference result."""
    response_list = []
    for inference_item in inference_result:
      if self.task == MASK_GENERATION:
        logging.info(inference_item)
        masks_rle = [
            mask_util.encode(np.asfortranarray(mask)) for mask in inference_item
        ]
        logging.info(masks_rle)
        for rle in masks_rle:
          rle["counts"] = rle["counts"].decode("utf-8")
        response = {"masks_rle": masks_rle}
        response_list.append(response)

    return response_list
