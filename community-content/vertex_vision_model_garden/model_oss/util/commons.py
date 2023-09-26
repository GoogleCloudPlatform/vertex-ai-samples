"""Common utility lib for prediction on images."""

from typing import Any, Dict, List

import numpy as np
from PIL import Image
import tensorflow as tf
import yaml

from util import image_format_converter


def get_prediction_instances(image: Image.Image) -> List[Dict[str, Any]]:
  """Gets prediction instances.

  Args:
    image: Image instance.

  Returns:
    List[Dict[str, Any]]: List of prediction instances.
  """
  instances = [{
      "encoded_image": {"b64": image_format_converter.image_to_base64(image)},
  }]
  return instances


def get_label_map(label_map_yaml_filepath: str) -> Dict[str, Any]:
  """Gets the label map from a YAML file.

  Args:
      label_map_yaml_filepath: Filepath to the label map YAML file.

  Returns:
      dict: Label map.
  """
  with tf.io.gfile.GFile(label_map_yaml_filepath, "rb") as input_file:
    label_map = yaml.safe_load(input_file.read())
  return label_map


def get_object_detection_endpoint_predictions(
    detection_endpoint: ...,
    input_image: np.ndarray,
    detection_thresh: float = 0.2,
) -> np.ndarray:
  """Gets endpoint predictions.

  Args:
      detection_endpoint: image object detection endpoint.
      input_image: Input image.
      detection_thresh: Detection threshold.

  Returns:
      Object detection predictions from endpoints.
  """
  height, width, _ = input_image.shape
  predictions = detection_endpoint.predict(
      get_prediction_instances(Image.fromarray(input_image))
  ).predictions
  detection_scores = np.array(predictions[0]["detection_scores"])
  detection_classes = np.array(predictions[0]["detection_classes"])
  detection_boxes = np.array(
      [
          [b[1] * width, b[0] * height, b[3] * width, b[2] * height]
          for b in predictions[0]["detection_boxes"]
      ]
  )
  thresh_indices = [
      x for x, val in enumerate(detection_scores) if val > detection_thresh
  ]
  preds_merge_conf = np.column_stack((
      detection_boxes[thresh_indices],
      detection_scores[thresh_indices],
  ))
  preds_merge_cls = np.column_stack(
      (preds_merge_conf, detection_classes[thresh_indices])
  )
  return preds_merge_cls
