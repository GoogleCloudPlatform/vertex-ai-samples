"""Image classification input and model functions for serving/inference."""

from typing import Callable, List, Mapping, Optional

import tensorflow as tf
from tensorflow.io import gfile

from tfvision.serving import automl_constants
from official.core import config_definitions as cfg
from official.vision.serving import image_classification


class ClassificationModule(image_classification.ClassificationModule):
  """classification Module."""

  def __init__(self,
               params: cfg.ExperimentConfig,
               *,
               batch_size: Optional[int] = None,
               input_image_size: List[int],
               input_type: str = automl_constants.INPUT_TYPE,
               num_channels: int = 3,
               model: Optional[tf.keras.Model] = None,
               input_name: str = automl_constants.ICN_INPUT_NAME,
               label_path: Optional[str] = None,
               key_name: str = automl_constants.INPUT_KEY_NAME):
    """Initializes a module for export.

    Args:
      params: Experiment params.
      batch_size: The batch size of the model input. Can be `int` or None.
      input_image_size: List or Tuple of size of the input image. For 2D image,
        it is [height, width].
      input_type: The input signature type.
      num_channels: The number of the image channels.
      model: A tf.keras.Model instance to be exported.
      input_name: A customized input tensor name.
      label_path: A label file path.
      key_name: A name to the automl model input key.
    """
    super().__init__(
        params=params,
        model=model,
        batch_size=batch_size,
        input_image_size=input_image_size,
        num_channels=num_channels,
        input_name=input_name,
        input_type=input_type,
    )

    self._key_name = key_name
    if label_path is not None:
      self._label = self._read_label(label_path)
    else:
      self._label = None

  def _read_label(self, label_path: str) -> tf.Tensor:
    """Reads the labels from a label file."""
    with gfile.GFile(label_path, 'r') as f:
      labels = [i.strip() for i in f.readlines()]
    labels = tf.convert_to_tensor([labels])
    return labels

  def serve(self, images: tf.Tensor, key: tf.Tensor) -> Mapping[str, tf.Tensor]:
    """Cast image to float and run inference.

    Args:
      images: uint8 Tensor of shape [batch_size, None, None, 3]
      key: string Tensor of shape [batch_size].

    Returns:
      Dictionary holding classification outputs.
    """
    with tf.device('cpu:0'):
      images = tf.cast(images, dtype=tf.float32)

      images = tf.nest.map_structure(
          tf.identity,
          tf.map_fn(
              self._build_inputs,
              elems=images,
              fn_output_signature=tf.TensorSpec(
                  shape=self._input_image_size + [3], dtype=tf.float32),
              parallel_iterations=32))

    logits = self.inference_step(images)
    if self.params.task.train_data.is_multilabel:
      probs = tf.math.sigmoid(logits)
    else:
      probs = tf.nn.softmax(logits)

    outputs = {'scores': probs, automl_constants.OUTPUT_KEY_NAME: key}
    if self._label is not None:
      outputs['labels'] = tf.tile(self._label, [tf.shape(images)[0], 1])
    return outputs

  @tf.function
  def inference_from_image_bytes(self, inputs: tf.Tensor,
                                 key: tf.Tensor) -> Mapping[str, tf.Tensor]:
    with tf.device('cpu:0'):
      images = tf.nest.map_structure(
          tf.identity,
          tf.map_fn(
              self._decode_image,
              elems=inputs,
              fn_output_signature=tf.TensorSpec(
                  shape=[None] * len(self._input_image_size) +
                  [self._num_channels],
                  dtype=tf.uint8),
              parallel_iterations=32))
      images = tf.stack(images)
    return self.serve(images, key)

  @tf.function
  def inference_from_image_tensors(
      self, inputs: tf.Tensor
  ) -> Mapping[str, tf.Tensor]:
    return self.serve(inputs, tf.zeros(tf.shape(inputs)[0], dtype=tf.string))

  def get_inference_signatures(
      self, function_keys: Mapping[str, str]
  ) -> Mapping[str, Callable[[tf.Tensor, tf.Tensor], Mapping[str, tf.Tensor]]]:
    """Gets defined function signatures.

    Args:
      function_keys: A dictionary with keys as the function to create signature
        for and values as the signature keys when returns.

    Returns:
      A dictionary with key as signature key and value as concrete functions
        that can be used for tf.saved_model.save.
    """
    signatures = {}
    for key, def_name in function_keys.items():
      # Adds input string 'key' to image_bytes input type.
      if key == automl_constants.INPUT_TYPE:
        input_images = tf.TensorSpec(
            shape=[self._batch_size], dtype=tf.string, name=self._input_name)
        input_key = tf.TensorSpec(
            shape=[self._batch_size], dtype=tf.string, name=self._key_name)
        signatures[
            def_name] = self.inference_from_image_bytes.get_concrete_function(
                input_images, input_key)
      elif key == automl_constants.IMAGE_TENSOR:
        input_signature = tf.TensorSpec(
            shape=[self._batch_size]
            + [None] * len(self._input_image_size)
            + [self._num_channels],
            dtype=tf.uint8,
            name=self._input_name,
        )
        signatures[def_name] = (
            self.inference_from_image_tensors.get_concrete_function(
                input_signature
            )
        )
      else:
        raise ValueError('Unrecognized `input_type`')
    return signatures
