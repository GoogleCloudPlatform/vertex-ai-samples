"""Custom handler for OpenCLIP model."""

# pylint:disable=g-importing-member
import enum
import logging
import os
from typing import Any, Dict, List

import open_clip
import torch
from ts.torch_handler.base_handler import BaseHandler

from util import constants
from util import fileutils
from util import image_format_converter


# Supported checkpoint&model pairs:
# https://github.com/mlfoundations/open_clip#pretrained-model-interface
_DEFAULT_MODEL = "RN50"
_BIOMED_CLIP_MODEL = "microsoft/BiomedCLIP"
_ZERO_CLASSIFICATION = "zero-shot-image-classification"
_FEATURE_EMBEDDING = "feature-embedding"
_VALID_TASKS = frozenset([_ZERO_CLASSIFICATION, _FEATURE_EMBEDDING])

_IMAGE_KEY = "image"
_TEXT_KEY = "text"
_IMAGE_FEATURES_KEY = "image_features"
_TEXT_FEATURES_KEY = "text_features"


class OpenclipHandler(BaseHandler):
  """Custom handler for OpenCLIP."""

  @enum.unique
  class Precision(enum.Enum):
    AMP = "amp"
    AMP_BF16 = "amp_bf16"
    AMP_BFLOAT16 = "amp_bfloat16"
    # For the difference between floating points and "pure" floating points, see
    # https://github.com/mlfoundations/open_clip/blob/0142d279298a4ca0138316286f775fe9d7bdbb94/src/open_clip/factory.py#L232C58-L232C58
    BF16 = "bf16"
    FP16 = "fp16"
    PURE_BF16 = "pure_bf16"
    PURE_FP16 = "pure_fp16"
    FP32 = "fp32"

  _DEFAULT_PRECISION = Precision.AMP

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

    self.model_name = os.environ.get("MODEL", None)
    if not self.model_name:
      self.model_name = os.environ.get("MODEL_ID", _DEFAULT_MODEL)
    precision = os.environ.get("PRECISION", self._DEFAULT_PRECISION)
    checkpoint = os.environ.get("CHECKPOINT")
    self.task = os.environ.get("TASK", _FEATURE_EMBEDDING)
    if self.task not in _VALID_TASKS:
      raise ValueError(f"Invalid task: {self.task}.")
    logging.info(
        "Handler initializing task:%s, model:%s, precision:%s, checkpoint:%s",
        self.task,
        self.model_name,
        precision,
        checkpoint,
    )

    if fileutils.is_gcs_path(checkpoint):
      local_fname = os.path.join(constants.LOCAL_MODEL_DIR, "model.pt")
      fileutils.download_gcs_file_to_local(checkpoint, local_fname)
      checkpoint = local_fname

    self.model, self.preprocessor = open_clip.create_model_from_pretrained(
        self.model_name, pretrained=checkpoint, precision=precision
    )
    self.model.to(self.device)
    self.tokenizer = open_clip.get_tokenizer(self.model_name)

    self.initialized = True

  def preprocess(self, data: Any) -> List[Dict[str, Any]]:
    """Preprocess input data."""
    logging.info("preprocessing: %d instances received.", len(data))
    processed_list = []
    for item in data:
      sample = {}
      if _IMAGE_KEY in item:
        sample[_IMAGE_KEY] = self.preprocessor(
            image_format_converter.base64_to_image(item[_IMAGE_KEY])
        ).unsqueeze(0)
      if _TEXT_KEY in item:
        sample[_TEXT_KEY] = self.tokenizer(item[_TEXT_KEY])
      processed_list.append(sample)
    return processed_list

  def _biomedclip_inference(
      self, data: List[Dict[str, Any]], *args, **kwargs
  ) -> List[List[float]]:
    """Inference for BiomedCLIP model."""
    texts = torch.stack(
        [item[_TEXT_KEY][0] for item in data if _TEXT_KEY in item]
    ).to(self.map_location)
    images = torch.stack(
        [item[_IMAGE_KEY][0] for item in data if _IMAGE_KEY in item]
    ).to(self.map_location)
    if texts.shape[0] == 0 or images.shape[0] == 0:
      return []
    with torch.no_grad():
      image_features, text_features, logit_scale = self.model(images, texts)
      logits = (
          (logit_scale * image_features @ text_features.t())
          .detach()
          .softmax(dim=-1)
      )
      return logits.cpu().numpy().tolist()

  def inference(
      self, data: List[Dict[str, Any]], *args, **kwargs
  ) -> List[Dict[str, Any]]:
    if _BIOMED_CLIP_MODEL in self.model_name:
      return self._biomedclip_inference(data)
    feature_list = []
    with torch.no_grad(), torch.cuda.amp.autocast():
      for item in data:
        sample = {}
        if _IMAGE_KEY in item:
          sample[_IMAGE_FEATURES_KEY] = self.model.encode_image(
              item[_IMAGE_KEY]
          )
        if _TEXT_KEY in item:
          sample[_TEXT_FEATURES_KEY] = self.model.encode_text(item[_TEXT_KEY])
        feature_list.append(sample)
    return feature_list

  def postprocess(self, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Postprocess the image/text featreus for downstream task."""
    if _BIOMED_CLIP_MODEL in self.model_name:
      return features
    preds = []
    if self.task == _FEATURE_EMBEDDING:
      for item in features:
        preds.append({k: v.tolist() for k, v in item.items()})
    elif self.task == _ZERO_CLASSIFICATION:
      for item in features:
        image_features = item.get(_IMAGE_FEATURES_KEY, None)
        text_features = item.get(_TEXT_FEATURES_KEY, None)
        if image_features is None or text_features is None:
          raise ValueError(
              "Missing input for {} task. {} received.".format(
                  _ZERO_CLASSIFICATION, item.keys()
              )
          )
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        preds.append(text_probs.tolist())

    return preds