"""Common utility lib for prediction on images."""

from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
import tensorflow as tf
import yaml

from util import image_format_converter


def convert_list_to_label_map(
    input_list: List[str],
) -> Tuple[Dict[str, Dict[int, str]], List[int]]:
  """Converts a list of labels to a dictionary and numerical encoding.

  Args:
    input_list: A list of strings representing class labels.

  Returns:
    A tuple containing:
      label_map: A dictionary mapping unique labels to integer indices.
      encoded_list: A list of integers corresponding to the labels in the input
      list.
  """
  unique_labels = set(input_list)
  label_map_reverse = {label: idx for idx, label in enumerate(unique_labels)}
  label_map = {idx: label for idx, label in enumerate(unique_labels)}
  encoded_list = [label_map_reverse[label] for label in input_list]

  return {"label_map": label_map}, encoded_list


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
    detector_endpoint: ...,
    input_image: np.ndarray,
    detection_thresh: float = 0.2,
) -> np.ndarray:
  """Gets endpoint predictions.

  Args:
      detector_endpoint: image object detection endpoint.
      input_image: Input image.
      detection_thresh: Detection threshold.

  Returns:
      Object detection predictions from endpoints.
  """
  height, width, _ = input_image.shape
  predictions = detector_endpoint.predict(
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
  return merge_boxes_and_classes(
      detection_scores, detection_boxes, detection_classes, detection_thresh
  )


def merge_boxes_and_classes(
    detection_scores: np.ndarray,
    detection_boxes: np.ndarray,
    detection_classes: np.ndarray,
    detection_thresh: float = 0.2,
) -> np.ndarray:
  """Merges prediction boxes and classes.

  Args:
      detection_scores: array of detection scores.
      detection_boxes: array of detection boxes.
      detection_classes: array of detection classes.
      detection_thresh: float indicating the detection threshold.

  Returns:
      preds_merge_cls: a numpy array containing the detection boxes, scores and
      classes.
  """
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
