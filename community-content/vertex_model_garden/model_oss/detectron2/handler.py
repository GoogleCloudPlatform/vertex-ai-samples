"""Custom handler for Detectron2 serving."""

import io
import json
import os
from typing import Any, List, Tuple

import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from google.cloud import storage
import numpy as np
import pycocotools.mask as mask_util
import torch


def get_bucket_and_blob_name(gcs_filepath: str) -> Tuple[str, str]:
  """Gets bucket and blob name from gcs path."""
  # The gcs path is of the form gs://<bucket-name>/<blob-name>
  gs_suffix = gcs_filepath.split("gs://", 1)[1]
  return tuple(gs_suffix.split("/", 1))


def download_gcs_file(src_file_path: str, dst_file_path: str):
  """Downloads gcs-file to local folder."""
  src_bucket_name, src_blob_name = get_bucket_and_blob_name(src_file_path)
  client = storage.Client()
  src_bucket = client.get_bucket(src_bucket_name)
  src_blob = src_bucket.blob(src_blob_name)
  src_blob.download_to_filename(dst_file_path)


class ModelHandler:
  """Custom model handler for Detectron2."""

  def __init__(self):
    self.error = None
    self._batch_size = 0
    self.initialized = False
    self.predictor = None
    self.test_threshold = 0.5

  def initialize(self, context: Any):
    """Initialize."""
    print("context.system_properties: ", context.system_properties)
    print("context.manifest: ", context.manifest)
    self.manifest = context.manifest
    properties = context.system_properties
    # Get threshold from environment variable.
    # This will be set by customer.
    self.test_threshold = float(os.environ.get("TEST_THRESHOLD"))
    print("test_threshold: ", self.test_threshold)
    # Get model and config file location from environment variables.
    # These will be set by customer when doing model upload.
    gcs_model_file = os.environ["MODEL_PTH_FILE"]
    gcs_config_file = os.environ["CONFIG_YAML_FILE"]
    print("Copying gcs_model_file: ", gcs_model_file)
    print("Copying gcs_config_file: ", gcs_config_file)
    # Copy these files from GCS location to local file.
    # Note(lavrai): GCSFuse path does not seem to work here for now.
    model_file = "./model.pth"
    config_file = "./cfg.yaml"
    download_gcs_file(src_file_path=gcs_model_file, dst_file_path=model_file)
    if not os.path.exists(model_file):
      raise RuntimeError("Missing model_file: %s" % model_file)
    download_gcs_file(src_file_path=gcs_config_file, dst_file_path=config_file)
    if not os.path.exists(config_file):
      raise RuntimeError("Missing config_file: %s" % config_file)

    # Set up config file.
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = model_file
    cfg.MODEL.DEVICE = (
        cfg.MODEL.DEVICE + str(properties.get("gpu_id"))
        if torch.cuda.is_available()
        else "cpu"
    )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.test_threshold

    # Build predictor from config.
    self.predictor = DefaultPredictor(cfg)
    self._batch_size = context.system_properties["batch_size"]
    self.initialized = True

  def preprocess(self, batch: List[Any]) -> List[Any]:
    """Preprocess raw input and return as list of images."""
    print("Running pre-processing.")
    images = []
    for request in batch:
      request_data = request.get("data")
      input_bytes = io.BytesIO(request_data)
      img = cv2.imdecode(np.fromstring(input_bytes.read(), np.uint8), 1)
      images.append(img)
    return images

  def inference(self, model_input: List[Any]) -> List[Any]:
    """Runs inference."""
    print("Running model-inference.")
    return [self.predictor(image) for image in model_input]

  def postprocess(self, inference_result: List[Any]) -> List[Any]:
    """Post process inference result."""
    response_list = []
    print("Num inference_items are:", len(inference_result))
    for inference_item in inference_result:
      predictions = inference_item["instances"].to("cpu")
      print("Predictions are:", predictions)
      boxes = None
      if predictions.has("pred_boxes"):
        boxes = predictions.pred_boxes.tensor.numpy().tolist()
      scores = None
      if predictions.has("scores"):
        scores = predictions.scores.numpy().tolist()
      classes = None
      if predictions.has("pred_classes"):
        classes = predictions.pred_classes.numpy().tolist()
      masks_rle = None
      if predictions.has("pred_masks"):
        # Do run length encoding, else the mask output becomes huge.
        masks_rle = [
            mask_util.encode(np.asfortranarray(mask))
            for mask in predictions.pred_masks
        ]
        for rle in masks_rle:
          rle["counts"] = rle["counts"].decode("utf-8")
      response = {
          "classes": classes,
          "scores": scores,
          "boxes": boxes,
          "masks_rle": masks_rle,
      }
      response_list.append(json.dumps(response))
    print("response_list: ", response_list)
    return response_list

  def handle(self, data: Any, context: Any) -> List[Any]:  # pylint: disable=unused-argument
    """Runs preprocess, inference, and post-processing."""
    model_input = self.preprocess(data)
    model_out = self.inference(model_input)
    output = self.postprocess(model_out)
    print("Done handling input.")
    return output


_service = ModelHandler()


def handle(data: Any, context: Any) -> List[Any]:
  if not _service.initialized:
    _service.initialize(context)
  if data is None:
    return None
  return _service.handle(data, context)