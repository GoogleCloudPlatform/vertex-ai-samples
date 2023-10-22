"""Detection input and model functions for serving/inference."""

import functools
import heapq
from typing import Any, Callable, Dict, List, Optional, Text

import tensorflow as tf

from tfvision.serving import automl_constants
from object_detection.utils import label_map_util
from official.core import config_definitions as cfg
from official.projects.yolo.modeling import factory as yolo_factory
from official.projects.yolo.modeling.decoders import yolo_decoder  # pylint: disable=unused-import
from official.projects.yolo.serving import model_fn as yolo_model_fn
from official.vision import configs
from official.vision.ops import box_ops
from official.vision.serving import detection as detection_module


def load_label_map_to_string_list(label_map_path: str,
                                  fill_in_gaps_and_background: bool = True
                                 ) -> List[str]:
  """Loads class labels as string list ordered by class id.

  Args:
    label_map_path: the path to label_map.pbtxt with string_int_label_map_pb2
      proto format.
    fill_in_gaps_and_background: whether to fill in gaps and background with
      respect to the id field in the proto. The id: 0 is reserved for the
      'background' class and will be added if it is missing. All other missing
      ids in range(1, max(id)) will be added with a dummy class name
      ("class_<id>") if they are missing.

  Returns:
    The class labels as text string lists in the order of the class numeric id.
  """

  labelmap = label_map_util.get_label_map_dict(
      label_map_path, fill_in_gaps_and_background=fill_in_gaps_and_background)
  heap = []
  for label_name, label_id in labelmap.items():
    heapq.heappush(heap, (label_id, label_name))
  label_list = [heapq.heappop(heap)[1] for _ in range(len(heap))]

  return label_list


class DetectionModule(detection_module.DetectionModule):
  """Detection Module."""

  def __init__(self,
               params: cfg.ExperimentConfig,
               *,
               batch_size: int,
               input_image_size: List[int],
               input_type: str = automl_constants.INPUT_TYPE,
               num_channels: int = 3,
               model: Optional[tf.keras.Model] = None,
               label_map_path: Optional[str] = None,
               input_name: str = automl_constants.IOD_INPUT_NAME,
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
      label_map_path: A labelmap proto file path.
      input_name: A customized input tensor name. This will be used as the
        signature's input image argument name.
      key_name: A name to the automl model input key.
    """
    self._key_name = key_name
    if label_map_path is not None:
      self._label_map_table = self._generate_label_map_list(label_map_path)
    else:
      self._label_map_table = None
    super().__init__(
        params=params,
        model=model,
        batch_size=batch_size,
        input_image_size=input_image_size,
        input_name=input_name,
        input_type=input_type)

  def _generate_label_map_list(self, label_map_path: str) -> tf.Tensor:
    """Generates a list of label texts from a labelmap path."""
    mapping_string = tf.convert_to_tensor(
        load_label_map_to_string_list(label_map_path))
    return tf.lookup.index_to_string_table_from_tensor(
        mapping_string, default_value=automl_constants.LOOKUP_DEFAULT_VALUE)

  def _generate_class_text_output(self, detection_classes) -> tf.Tensor:
    """Converts class index to class text."""
    if self._label_map_table is None:
      raise ValueError('_label_map_table is None.')
    indices = tf.cast(detection_classes, tf.int64)
    indices = tf.reshape(indices, [-1])
    values = self._label_map_table.lookup(indices)
    return tf.reshape(
        values, [-1, tf.array_ops.shape(detection_classes)[1]],
        name=automl_constants.DETECTION_CLASSES_AS_TEXT)

  def serve(self,
            images: tf.Tensor,
            key: Optional[tf.Tensor] = None) -> Dict[Text, tf.Tensor]:
    """Cast image to float and run inference.

    Args:
      images: uint8 Tensor of input images. For input type image tensor, the
        shape is [batch_size, None, None, 3], for image_bytes, the shape is
        [batch_size].
      key: Optional string Tensor of shape [batch_size]. If not provided
        output tensors will not contain it as well.

    Returns:
      Tensor holding detection output logits.
    """

    images, anchor_boxes, image_info = self.preprocess(images)
    input_image_shape = image_info[:, 1, :]

    # To overcome keras.Model extra limitation to save a model with layers that
    # have multiple inputs, we use `model.call` here to trigger the forward
    # path. Note that, this disables some keras magics happens in `__call__`.
    detections = self.model.call(
        images=images,
        image_shape=input_image_shape,
        anchor_boxes=anchor_boxes,
        training=False)

    if self.params.task.model.detection_generator.apply_nms:
      # For RetinaNet model, apply export_config.
      if isinstance(self.params.task.model, configs.retinanet.RetinaNet):
        export_config = self.params.task.export_config
        # Normalize detection box coordinates to [0, 1].
        if export_config.output_normalized_coordinates:
          detection_boxes = (
              detections['detection_boxes'] /
              tf.tile(image_info[:, 2:3, :], [1, 1, 2]))
          detections['detection_boxes'] = box_ops.normalize_boxes(
              detection_boxes, image_info[:, 0:1, :])

        # Cast num_detections and detection_classes to float. This allows the
        # model inference to work on chain (go/chain) as chain requires floating
        # point outputs.
        if export_config.cast_num_detections_to_float:
          detections['num_detections'] = tf.cast(
              detections['num_detections'], dtype=tf.float32)
        if export_config.cast_detection_classes_to_float:
          detections['detection_classes'] = tf.cast(
              detections['detection_classes'], dtype=tf.float32)

      final_outputs = {
          'detection_boxes': detections['detection_boxes'],
          'detection_scores': detections['detection_scores'],
          'detection_classes': detections['detection_classes'],
          'num_detections': detections['num_detections']
      }
    else:
      final_outputs = {
          'decoded_boxes': detections['decoded_boxes'],
          'decoded_box_scores': detections['decoded_box_scores']
      }

    if 'detection_masks' in detections.keys():
      final_outputs['detection_masks'] = detections['detection_masks']

    # Adding AutoML specific outputs.
    if self._label_map_table is not None:
      final_outputs.update({
          automl_constants.DETECTION_CLASSES_AS_TEXT:
              self._generate_class_text_output(detections['detection_classes'])
      })

    final_outputs.update({'image_info': image_info})
    if key is not None:
      final_outputs.update({automl_constants.OUTPUT_KEY_NAME: key})

    return final_outputs

  @tf.function
  def inference_from_image_bytes(
      self,
      inputs: tf.Tensor,
      key: tf.Tensor,
  ) -> Dict[Text, tf.Tensor]:
    """Entry point for model input.

    Raw image tensor will be decoded to the desired image format.

    Args:
      inputs: Image tensor to be feed to the model.
      key: AutoML specific input key to track image names or image ids.

    Returns:
      A dictionary of Tensor that contains model outputs.
    """
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
  def inference_from_image_bytes_wo_key(
      self, inputs: tf.Tensor) -> Dict[Text, tf.Tensor]:
    """Entry point for model inference without input key tensor.

    Raw image tensor will be decoded to the desired image format.

    Args:
      inputs: Image tensor to be feed to the model.

    Returns:
      A dictionary of Tensor that contains model outputs.
    """
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

    return self.serve(images)

  def get_inference_signatures(
      self, function_keys: Dict[Text, Text]
  ) -> Dict[Text, Callable[[tf.Tensor, tf.Tensor], Dict[Text, tf.Tensor]]]:
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
        # For each input type, create a signature without input key tensor.
        def_name_wo_key = def_name + automl_constants.NO_KEY_SIG_DEF_SUFFIX
        signatures[def_name_wo_key] = (
            self.inference_from_image_bytes_wo_key.get_concrete_function(
                input_images))
      else:
        raise ValueError('Unrecognized `input_type`')
    return signatures


class YoloDetectionModule(DetectionModule):
  """Yolo detection module for Model Garden."""

  def __init__(
      self,
      params: cfg.ExperimentConfig,
      *,
      batch_size: int,
      input_image_size: List[int],
      preprocessor: Callable[..., Any],
      inference_step: Callable[..., Any],
      input_type: str = automl_constants.INPUT_TYPE,
      num_channels: int = 3,
      model: Optional[tf.keras.Model] = None,
      label_map_path: Optional[str] = None,
      input_name: str = automl_constants.IOD_INPUT_NAME,
      key_name: str = automl_constants.INPUT_KEY_NAME,
  ):
    """Initializes a module for export.

    Args:
      params: Experiment params.
      batch_size: The batch size of the model input. Can be `int` or None.
      input_image_size: List or Tuple of size of the input image. For 2D image,
        it is [height, width].
      preprocessor: An optional callable to preprocess the inputs.
      inference_step: An optional callable to forward-pass the model.
      input_type: The input signature type.
      num_channels: The number of the image channels.
      model: A tf.keras.Model instance to be exported.
      label_map_path: A labelmap proto file path.
      input_name: A customized input tensor name. This will be used as the
        signature's input image argument name.
      key_name: A name to the automl model input key.
    """
    super().__init__(
        params=params,
        batch_size=batch_size,
        input_image_size=input_image_size,
        input_type=input_type,
        num_channels=num_channels,
        model=model,
        label_map_path=label_map_path,
        input_name=input_name,
        key_name=key_name,
    )

    self.preprocessor = preprocessor
    self.inference_step = functools.partial(inference_step, model=self.model)

  def preprocess(self, images: tf.Tensor) -> None:
    raise NotImplementedError('Use self.preprocessor instead.')

  def serve(
      self, images: tf.Tensor, key: Optional[tf.Tensor] = None
  ) -> Dict[Text, tf.Tensor]:
    """Cast image to float and run inference.

    Args:
      images: uint8 Tensor of input images. For input type image tensor, the
        shape is [batch_size, None, None, 3], for image_bytes, the shape is
        [batch_size].
      key: Optional string Tensor of shape [batch_size]. If not provided output
        tensors will not contain it as well.

    Returns:
      Tensor holding detection output logits.
    """
    images, image_info = self.preprocessor(images)
    final_outputs = self.inference_step((images, image_info))

    # Normalize detection box coordinates to [0, 1].
    detection_boxes = final_outputs['detection_boxes'] / tf.tile(
        image_info[:, 2:3, :], [1, 1, 2]
    )
    final_outputs['detection_boxes'] = box_ops.normalize_boxes(
        detection_boxes, image_info[:, 0:1, :]
    )

    # Cast num_detections and detection_classes to float. This allows the
    # model inference to work on chain (go/chain) as chain requires floating
    # point outputs.
    final_outputs['num_detections'] = tf.cast(
        final_outputs['num_detections'], dtype=tf.float32
    )
    final_outputs['detection_classes'] = tf.cast(
        final_outputs['detection_classes'], dtype=tf.float32
    )

    # Adding AutoML specific outputs.
    if self._label_map_table is not None:
      final_outputs.update(
          {
              automl_constants.DETECTION_CLASSES_AS_TEXT: (
                  self._generate_class_text_output(
                      final_outputs['detection_classes']
                  )
              )
          }
      )

    final_outputs.update({'image_info': image_info})
    if key is not None:
      final_outputs.update({automl_constants.OUTPUT_KEY_NAME: key})

    return final_outputs


def create_yolov7_export_module(
    params: cfg.ExperimentConfig,
    input_type: str,
    batch_size: int,
    input_image_size: List[int],
    num_channels: int = 3,
    input_name: Optional[str] = None,
    label_map_path: Optional[str] = None,
) -> YoloDetectionModule:
  """Creates YOLO export module for Model Garden."""
  input_specs = tf.keras.layers.InputSpec(
      shape=[batch_size] + input_image_size + [num_channels]
  )
  model = yolo_factory.build_yolov7(
      input_specs=input_specs,
      model_config=params.task.model,
      l2_regularization=None,
  )

  def preprocess_fn(image_tensor):
    def normalize_image_fn(inputs):
      image = tf.cast(inputs, dtype=tf.float32)
      return image / 255.0

    # If input_type is `tflite`, do not apply image preprocessing. Only apply
    # normalization.
    if input_type == 'tflite':
      return normalize_image_fn(image_tensor), None

    def preprocess_image_fn(inputs):
      image = normalize_image_fn(inputs)
      (image, image_info) = yolo_model_fn.letterbox(
          image,
          input_image_size,
          letter_box=params.task.validation_data.parser.letter_box,
      )
      return image, image_info

    images_spec = tf.TensorSpec(shape=input_image_size + [3], dtype=tf.float32)

    image_info_spec = tf.TensorSpec(shape=[4, 2], dtype=tf.float32)

    images, image_info = tf.nest.map_structure(
        tf.identity,
        tf.map_fn(
            preprocess_image_fn,
            elems=image_tensor,
            fn_output_signature=(images_spec, image_info_spec),
            parallel_iterations=32,
        ),
    )

    return images, image_info

  def inference_steps(inputs, model):
    images, image_info = inputs
    detection = model.call(images, training=False)
    if input_type != 'tflite':
      detection['bbox'] = yolo_model_fn.undo_info(
          detection['bbox'],
          detection['num_detections'],
          image_info,
          expand=False,
      )

    final_outputs = {
        'detection_boxes': detection['bbox'],
        'detection_scores': detection['confidence'],
        'detection_classes': detection['classes'],
        'num_detections': detection['num_detections'],
    }

    return final_outputs

  export_module = YoloDetectionModule(
      params=params,
      model=model,
      batch_size=batch_size,
      input_image_size=input_image_size,
      input_type=input_type,
      num_channels=num_channels,
      input_name=input_name,
      label_map_path=label_map_path,
      preprocessor=preprocess_fn,
      inference_step=inference_steps,
  )

  return export_module
