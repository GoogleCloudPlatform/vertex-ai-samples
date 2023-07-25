"""Custom handler for huggingface/transformers models."""

# pylint: disable=g-multiple-import
# pylint: disable=g-importing-member

import logging
import os
from typing import Any, List, Optional, Tuple

from PIL import Image
import torch
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
    BlipProcessor,
    CLIPModel,
)
from transformers import pipeline
from ts.torch_handler.base_handler import BaseHandler

from util import constants
from util import fileutils
from util import image_format_converter

DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"
SALESFORCE_BLIP = "Salesforce/blip"
SALESFORCE_BLIP2 = "Salesforce/blip2"
FLAN_T5 = "flan-t5"
BART_LARGE_CNN = "facebook/bart-large-cnn"

ZERO_CLASSIFICATION = "zero-shot-image-classification"
FEATURE_EMBEDDING = "feature-embedding"
ZERO_DETECTION = "zero-shot-object-detection"
IMAGE_CAPTIONING = "image-to-text"
VQA = "visual-question-answering"
DQA = "document-question-answering"
SUMMARIZATION = "summarization"
SUMMARIZATION_TEMPLATE = (
    "Summarize the following news article:\n{input}\nSummary:\n"
)


class TransformersHandler(BaseHandler):
  """Custom handler for huggingface/transformers models."""

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
    if self.model_id.startswith(constants.GCS_URI_PREFIX):
      gcs_path = self.model_id[len(constants.GCS_URI_PREFIX) :]
      local_model_dir = os.path.join(constants.LOCAL_MODEL_DIR, gcs_path)
      logging.info("Download %s to %s", self.model_id, local_model_dir)
      fileutils.download_gcs_dir_to_local(self.model_id, local_model_dir)
      self.model_id = local_model_dir

    self.task = (
        ZERO_CLASSIFICATION
        if os.environ.get("TASK") is None
        else os.environ["TASK"]
    )
    logging.info(
        "Handler initializing task:%s, model:%s", self.task, self.model_id
    )

    if SALESFORCE_BLIP in self.model_id:
      # pipeline() hasn't been ready for Salesforce/blip models.
      self.salesforce_blip = True
      self._create_blip_model()
    else:
      self.salesforce_blip = False
      if self.task == FEATURE_EMBEDDING:
        self.model = CLIPModel.from_pretrained(self.model_id).to(
            self.map_location
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
      elif self.task == SUMMARIZATION and FLAN_T5 in self.model_id:
        self.pipeline = pipeline(
            task=self.task,
            model=self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
      else:
        self.pipeline = pipeline(
            task=self.task, model=self.model_id, device=self.device
        )

    self.initialized = True
    logging.info("Handler initialization done.")

  def _create_blip_model(self):
    """A helper for creating BLIP and BLIP2 models."""
    if SALESFORCE_BLIP2 in self.model_id:
      self.torch_type = torch.float16
      self.processor = Blip2Processor.from_pretrained(self.model_id)
      self.model = Blip2ForConditionalGeneration.from_pretrained(
          self.model_id, torch_dtype=self.torch_type
      ).to(self.map_location)
    else:
      self.torch_type = torch.float32
      self.processor = BlipProcessor.from_pretrained(self.model_id)
      if self.task == IMAGE_CAPTIONING:
        self.model = BlipForConditionalGeneration.from_pretrained(
            self.model_id
        ).to(self.map_location)
      elif self.task == VQA:
        self.model = BlipForQuestionAnswering.from_pretrained(self.model_id).to(
            self.map_location
        )

  def _reformat_detection_result(self, data: List[Any]) -> List[Any]:
    """Reformat zero-shot-object-detection output."""
    if not data:
      return [data]
    boxes = {}
    boxes["label"] = data[0]["label"]
    boxes["boxes"] = []
    for item in data:
      box = {}
      box["score"] = item["score"]
      box.update(item["box"])
      boxes["boxes"].append(box)
    outputs = [boxes]
    return outputs

  def preprocess(
      self, data: Any
  ) -> Tuple[Optional[List[str]], Optional[List[Image.Image]]]:
    """Preprocess input data."""
    texts = None
    images = None
    if "text" in data[0]:
      texts = [item["text"] for item in data]
    if "image" in data[0]:
      images = [
          image_format_converter.base64_to_image(item["image"]) for item in data
      ]
    return texts, images

  def inference(self, data: Any, *args, **kwargs) -> List[Any]:
    """Run the inference."""
    texts, images = data
    preds = None
    if self.task == ZERO_CLASSIFICATION:
      preds = self.pipeline(images=images, candidate_labels=texts)
    elif self.task == ZERO_DETECTION:
      # The object detection pipeline doesn't support batch prediction.
      preds = self.pipeline(image=images[0], candidate_labels=texts[0])
    elif self.task == IMAGE_CAPTIONING:
      if self.salesforce_blip:
        inputs = self.processor(images[0], return_tensors="pt").to(
            self.map_location, self.torch_type
        )
        preds = self.model.generate(**inputs)
        preds = [
            self.processor.decode(preds[0], skip_special_tokens=True).strip()
        ]
      else:
        preds = self.pipeline(images=images)
    elif self.task == VQA:
      # The VQA pipelines doesn't support batch prediction.
      if self.salesforce_blip:
        inputs = self.processor(images[0], texts[0], return_tensors="pt").to(
            self.map_location, self.torch_type
        )
        preds = self.model.generate(**inputs)
        preds = [
            self.processor.decode(preds[0], skip_special_tokens=True).strip()
        ]
      else:
        preds = self.pipeline(image=images[0], question=texts[0])
    elif self.task == DQA:
      # The DQA pipelines doesn't support batch prediction.
      preds = self.pipeline(image=images[0], question=texts[0])
    elif self.task == FEATURE_EMBEDDING:
      preds = {}
      if texts:
        inputs = self.tokenizer(
            text=texts, padding=True, return_tensors="pt"
        ).to(self.map_location)
        text_features = self.model.get_text_features(**inputs)
        preds["text_features"] = text_features.detach().cpu().numpy().tolist()
      if images:
        inputs = self.processor(images=images, return_tensors="pt").to(
            self.map_location
        )
        image_features = self.model.get_image_features(**inputs)
        preds["image_features"] = image_features.detach().cpu().numpy().tolist()
      preds = [preds]
    elif self.task == SUMMARIZATION and FLAN_T5 in self.model_id:
      texts = [SUMMARIZATION_TEMPLATE.format(input=text) for text in texts]
      preds = self.pipeline(texts, max_length=130)
    elif self.task == SUMMARIZATION and self.model_id == BART_LARGE_CNN:
      preds = self.pipeline(
          texts[0], max_length=130, min_length=30, do_sample=False
      )
    else:
      raise ValueError(f"Invalid TASK: {self.task}")
    return preds

  def postprocess(self, data: Any) -> List[Any]:
    if self.task == ZERO_DETECTION:
      data = self._reformat_detection_result(data)
    return data
