"""Loads a tf-hub model."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from absl import logging
import tensorflow as tf
import tensorflow_hub as hub

from official.modeling import hyperparams
from official.vision.modeling.backbones import factory
from official.vision.ops import preprocess_ops

layers = tf.keras.layers


@tf.keras.utils.register_keras_serializable(package='Vision')
class HubModel(tf.keras.Model):
  """A tf-hub model wrapper."""

  def __init__(
      self,
      handle: str,
      input_specs: tf.keras.layers.InputSpec = layers.InputSpec(
          shape=[None, None, None, 3]
      ),
      trainable: bool = True,
      mean_rgb: Optional[float] = None,
      stddev_rgb: Optional[float] = None,
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      signature: Optional[str] = None,
      output_key: Optional[str] = None,
      **kwargs,
  ):
    """Initializes a tf-hub model.

    Args:
      handle: A handle to load a saved model via hub.load().
      input_specs: A input_spec of the input tensor.
      trainable: Controls whether this layer is trainable. Must not be set to
        True when using a signature (raises ValueError), including the use of
        legacy TF1 Hub format.
      mean_rgb: The mean rgb value used for normalization.
      stddev_rgb: The standard deviation of rgb values used for normalization.
      kernel_regularizer: A regularizer object for kernel weights.
      signature: Optional. If set, KerasLayer will use the requested signature.
        For legacy models in TF1 Hub format leaving unset means to use the
        `default` signature. When using a signature, output_key have to set.
      output_key: Name of the output item to return if the layer returns a dict.
        For legacy models in TF1 Hub format leaving unset means to return the
        `default` output.
      **kwargs: Additional keyword arguments to be passed.
    """
    self._handle = handle
    self._mean_rgb = mean_rgb
    self._stddev_rgb = stddev_rgb
    self._kernel_regularizer = kernel_regularizer
    self._signature = signature
    self._output_key = output_key

    inputs = tf.keras.Input(shape=input_specs.shape[1:])
    x = inputs
    if mean_rgb or stddev_rgb:
      x = layers.Lambda(self.re_normalize)(x)

    model = hub.KerasLayer(
        handle=handle,
        trainable=trainable,
        signature=signature,
        output_key=output_key,
    )
    if trainable and kernel_regularizer:
      if hasattr(model, 'regularization_losses'):
        logging.warning('regularization_losses already defined in the model.')

      def reg_loss(x):
        return lambda: kernel_regularizer(x)

      for v in model.trainable_variables:
        if 'kernel' in v.name:
          model.add_loss(reg_loss(v))
    x = model(x)
    if not trainable:
      # Solves backpropagation errors when loading CoCa.
      x = tf.stop_gradient(x)
    endpoints = {'0': x[:, tf.newaxis, tf.newaxis, :]}

    self._output_specs = {l: endpoints[l].get_shape() for l in endpoints}

    super().__init__(
        inputs=inputs, outputs=endpoints, trainable=trainable, **kwargs
    )

  def re_normalize(self, x: tf.Tensor) -> tf.Tensor:
    """Re-normalizes the input image.

    Tf-vision normalizes the images from [0, 255] to normal distribution. The
    tf-hub models are usually normalized to [0.0, 1.0]. This function converts
    the input image to proper scale.

    Args:
      x: The input image.

    Returns:
      The re-normalized image.
    """
    offset = tf.constant(preprocess_ops.MEAN_RGB)
    scale = tf.constant(preprocess_ops.STDDEV_RGB)
    x = x * scale + offset

    if self._mean_rgb:
      x -= self._mean_rgb
    if self._stddev_rgb:
      x /= self._stddev_rgb
    return x

  def get_config(self) -> Mapping[str, Any]:
    config_dict = {
        'handle': self._handle,
        'trainable': self.trainable,
        'mean_rgb': self._mean_rgb,
        'stddev_rgb': self._stddev_rgb,
        'kernel_regularizer': self._kernel_regularizer,
        'signature': self._signature,
        'output_key': self._output_key,
    }
    return config_dict

  @classmethod
  def from_config(cls,
                  config: Mapping[str, Any],
                  custom_objects: Optional[Any] = None) -> HubModel:
    return cls(**config)

  @property
  def output_specs(self) -> Mapping[str, tf.TensorShape]:
    """A dict of {level: TensorShape} pairs for the model output."""
    return self._output_specs


@factory.register_backbone_builder('hub_model')
def build_hub_model(
    input_specs: tf.keras.layers.InputSpec,
    backbone_config: hyperparams.Config,
    l2_regularizer: tf.keras.regularizers.Regularizer = None,
    **kwargs: Any,
) -> tf.keras.Model:  # pytype: disable=annotation-type-mismatch  # typed-keras
  """Builds ResNet backbone from a config."""
  del kwargs
  backbone_type = backbone_config.type
  backbone_cfg = backbone_config.get()
  assert backbone_type == 'hub_model', (f'Inconsistent backbone type '
                                        f'{backbone_type}')

  return HubModel(
      input_specs=input_specs,
      handle=backbone_cfg.handle,
      trainable=backbone_cfg.trainable,
      mean_rgb=backbone_cfg.mean_rgb,
      stddev_rgb=backbone_cfg.stddev_rgb,
      kernel_regularizer=l2_regularizer,
      signature=backbone_cfg.signature,
      output_key=backbone_cfg.output_key,
  )
