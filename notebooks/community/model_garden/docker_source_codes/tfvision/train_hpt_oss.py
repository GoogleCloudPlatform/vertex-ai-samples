"""TensorFlow Model Garden Vision training driver.

This is the main function to start OSS vision training dockers, and will run in
external environment.
"""

import json
import os
import time
from typing import Any

from absl import app
from absl import flags
from absl import logging
import gin
import hypertune
import tensorflow as tf

from util import constants
from util import hypertune_utils
from official.common import distribute_utils
from official.common import flags as tfm_flags
from official.core import task_factory
from official.core import train_lib
from official.core import train_utils
from official.modeling import performance
# pylint: disable=unused-import
from tfvision import registry_imports as vision_registry_imports
from official.projects.yolo.common import registry_imports as yolo_imports
from official.vision import registry_imports

if os.environ.get('ENABLE_MAX_VIT', ''):
  # pylint: disable=g-import-not-at-top
  # pylint: disable=import-error
  # pylint: disable=no-name-in-module
  from official.projects.maxvit import registry_imports as maxvit_imports
# pylint: enable=unused-import

# File type tfrecord.
_FILE_TYPE_TFRECORD = 'tfrecord'


_OBJECTIVE = flags.DEFINE_enum(
    'objective',
    constants.OBJECTIVE_IMAGE_CLASSIFICATION,
    [
        constants.OBJECTIVE_IMAGE_CLASSIFICATION,
        constants.OBJECTIVE_IMAGE_OBJECT_DETECTION,
        constants.OBJECTIVE_IMAGE_SEGMENTATION,
    ],
    'The objective of this training job.',
)

_MODEL_NAME = flags.DEFINE_string(
    'model_name',
    None,
    (
        'The model name for backbones. e.g.: the model names can be `vit-ti16`,'
        '`vit-b16`, `vit-s16`, `vit-l16`, for `deit_imagenet_pretrain`.'
    ),
)

_INIT_CHECKPOINT = flags.DEFINE_string(
    'init_checkpoint', None, 'The initial checkpoint of this training job.'
)

_BACKBONE_TRAINABLE = flags.DEFINE_bool(
    'backbone_trainable', None, 'Whether to train the backbone.'
)

_LEARNING_RATE = flags.DEFINE_float(
    'learning_rate', None, 'The learning rate of this training job.'
)

_WEIGHT_DECAY = flags.DEFINE_float(
    'weight_decay', None, 'The weight decay of this training job.'
)

_NUM_CLASSES = flags.DEFINE_integer(
    'num_classes', None, 'The number of classes.'
)

_INPUT_SIZE = flags.DEFINE_list(
    'input_size', None, 'Expected width and height of the input image.'
)

_INPUT_TRAIN_DATA_PATH = flags.DEFINE_string(
    'input_train_data_path', None, 'Input train data path.'
)

_INPUT_VALIDATION_DATA_PATH = flags.DEFINE_string(
    'input_validation_data_path', None, 'Input validation data path.'
)

_GLOBAL_BATCH_SIZE = flags.DEFINE_integer(
    'global_batch_size', None, 'Global batch size.'
)

_PREFETCH_BUFFER_SIZE = flags.DEFINE_integer(
    'prefetch_buffer_size', None, 'Prefetch buffer size.'
)

_TRAIN_STEPS = flags.DEFINE_integer('train_steps', None, 'Train steps.')


_ANCHOR_SIZE = flags.DEFINE_integer(
    'anchor_size', None, 'IOD model anchor size.'
)

_OUTPUT_SIZE = flags.DEFINE_list(
    'output_size',
    None,
    'Expected width and height of the output image for ISG models.',
)

_MAX_EVAL_WAIT_TIME = flags.DEFINE_integer(
    'max_eval_wait_time',
    0,
    (
        'Maximum duration to wait for evaluation result file after finishing'
        ' the training job in seconds. Defaults to 0, immediately looking for'
        ' the evaluation file.'
    ),
)

_LOG_LEVEL = flags.DEFINE_string('log_level', 'INFO', 'Log level.')

FLAGS = flags.FLAGS


def get_best_eval_metric(objective: str, params: Any) -> str:
  """Gets best eval metric.

  Args:
    objective: The objective of this training job.
    params: Experiment config.

  Returns:
    Eval metric to use.

  Raises:
    ValueError: If params does not have best_checkpoint_eval_metric set and the
      objective is not valid.
  """
  try:
    eval_metric_name = params.trainer.best_checkpoint_eval_metric
  except AttributeError:
    eval_metric_name = None

  if not eval_metric_name:
    # If eval metric is not given in params, use the default value.
    if objective == constants.OBJECTIVE_IMAGE_CLASSIFICATION:
      try:
        is_multilabel = params.task.train_data.is_multilabel
      except AttributeError:
        # Set default.
        is_multilabel = False
      if is_multilabel:
        eval_metric_name = (
            constants.IMAGE_CLASSIFICATION_MULTI_LABEL_BEST_EVAL_METRIC
        )
      else:
        eval_metric_name = (
            constants.IMAGE_CLASSIFICATION_SINGLE_LABEL_BEST_EVAL_METRIC
        )
    elif objective == constants.OBJECTIVE_IMAGE_OBJECT_DETECTION:
      eval_metric_name = constants.IMAGE_OBJECT_DETECTION_BEST_EVAL_METRIC
    elif objective == constants.OBJECTIVE_IMAGE_SEGMENTATION:
      eval_metric_name = constants.IMAGE_SEGMENTATION_BEST_EVAL_METRIC
    else:
      raise ValueError(
          'The objective must be {}, {}, or {}.'.format(
              constants.OBJECTIVE_IMAGE_CLASSIFICATION,
              constants.OBJECTIVE_IMAGE_OBJECT_DETECTION,
              constants.OBJECTIVE_IMAGE_SEGMENTATION,
          )
      )
  return eval_metric_name


def parse_params() -> Any:
  """Parses parameters."""
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_params)
  params = train_utils.parse_configuration(FLAGS, lock_return=False)
  if _INIT_CHECKPOINT.value:
    params.task.init_checkpoint = _INIT_CHECKPOINT.value
    if 'yolov7' in FLAGS.experiment:
      params.task.init_checkpoint_modules = ['backbone', 'decoder']
    else:
      params.task.init_checkpoint_modules = 'backbone'
  if _MODEL_NAME.value:
    if FLAGS.experiment in [
        'deit_imagenet_pretrain',
        'vit_imagenet_pretrain',
        'vit_imagenet_finetune',
    ]:
      params.task.model.backbone.vit.model_name = _MODEL_NAME.value
  if _NUM_CLASSES.value:
    params.task.model.num_classes = _NUM_CLASSES.value
  if _INPUT_SIZE.value:
    input_size = [int(elem) for elem in _INPUT_SIZE.value]
    if len(input_size) != 2:
      raise ValueError('The input size must contain 2 integers.')
    if input_size[0] < 0 or input_size[1] < 0:
      raise ValueError('The input size must be positive.')
    params.task.model.input_size = [input_size[0], input_size[1], 3]
  # If users set input train/validation data path, we assume the data are
  # converted from data converter as tfrecord. Users can use tfds by writing
  # their own config directly, and no need to override this parameter.
  if _INPUT_TRAIN_DATA_PATH.value:
    params.task.train_data.input_path = _INPUT_TRAIN_DATA_PATH.value
    params.task.train_data.file_type = _FILE_TYPE_TFRECORD
    params.task.train_data.tfds_name = ''
  if _INPUT_VALIDATION_DATA_PATH.value:
    params.task.validation_data.input_path = _INPUT_VALIDATION_DATA_PATH.value
    params.task.validation_data.file_type = _FILE_TYPE_TFRECORD
    params.task.validation_data.tfds_name = ''
  if _GLOBAL_BATCH_SIZE.value:
    params.task.train_data.global_batch_size = _GLOBAL_BATCH_SIZE.value
    params.task.validation_data.global_batch_size = _GLOBAL_BATCH_SIZE.value
  if _PREFETCH_BUFFER_SIZE.value:
    params.task.train_data.prefetch_buffer_size = _PREFETCH_BUFFER_SIZE.value
    params.task.validation_data.prefetch_buffer_size = (
        _PREFETCH_BUFFER_SIZE.value
    )

  # Use `get` method of train_utils.hyperparams.OneOfConfig to get learning
  # rate config.
  learning_rate = params.trainer.optimizer_config.learning_rate.get()

  if _TRAIN_STEPS.value:
    params.trainer.train_steps = _TRAIN_STEPS.value
    if hasattr(learning_rate, 'decay_steps'):
      learning_rate.decay_steps = _TRAIN_STEPS.value
  if (
      _BACKBONE_TRAINABLE.value is not None
      and params.task.model.backbone.type == 'hub_model'
  ):
    params.task.model.backbone.hub_model.trainable = _BACKBONE_TRAINABLE.value
  if _LEARNING_RATE.value:
    logging.info('Updating learning_rate: %s', _LEARNING_RATE.value)
    if hasattr(learning_rate, 'initial_learning_rate'):
      learning_rate.initial_learning_rate = _LEARNING_RATE.value

  if _WEIGHT_DECAY.value and 'yolo' in FLAGS.experiment:
    if 'sgd_torch' == params.trainer.optimizer_config.optimizer.type:
      params.trainer.optimizer_config.optimizer.sgd_torch.weight_decay = (
          _WEIGHT_DECAY.value
      )
    elif 'adamw' == params.trainer.optimizer_config.optimizer.type:
      params.trainer.optimizer_config.optimizer.adamw.weight_decay_rate = (
          _WEIGHT_DECAY.value
      )

  # Yolo models does not support anchor size.
  if _ANCHOR_SIZE.value and 'yolo' not in FLAGS.experiment:
    params.task.model.anchor.anchor_size = _ANCHOR_SIZE.value

  # Segmentation models will also set output size.
  if _OUTPUT_SIZE.value:
    output_size = [int(elem) for elem in _OUTPUT_SIZE.value]
    if len(output_size) != 2:
      raise ValueError('The output size must contain 2 integers.')
    if output_size[0] < 0 or output_size[1] < 0:
      raise ValueError('The output size must be positive.')
    params.task.train_data.output_size = output_size
    params.task.validation_data.output_size = output_size

  # Set default params for best checkpoints.
  params.trainer.best_checkpoint_export_subdir = constants.BEST_CKPT_DIRNAME
  params.trainer.best_checkpoint_metric_comp = constants.BEST_CKPT_METRIC_COMP
  params.trainer.best_checkpoint_eval_metric = get_best_eval_metric(
      _OBJECTIVE.value, params
  )
  return params


def wait_for_evaluation_file(
    eval_filepath: str,
    max_eval_wait_time: int,
    eval_wait_interval: int = 30,
) -> None:
  """Waits for the evaluation file to be created.

  Args:
    eval_filepath: The path to the evaluation file.
    max_eval_wait_time: The maximum amount of time to wait for the evaluation
      file to be created, in seconds.
    eval_wait_interval: The interval at which to check for the existence of the
      evaluation file, in seconds. Defaults to 30 seconds.

  Raises:
    ValueError: If the evaluation file does not exist after the maximum amount
      of time has passed.
  """
  eval_wait_start_time = time.time()
  while not tf.io.gfile.exists(eval_filepath):
    if time.time() - eval_wait_start_time >= max_eval_wait_time:
      raise ValueError('The eval file {} does not exist.'.format(eval_filepath))
    time.sleep(eval_wait_interval)
  return


def main(_):
  log_level = _LOG_LEVEL.value
  if log_level and log_level in ['FATAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']:
    logging.set_verbosity(log_level)
  params = parse_params()
  logging.info('The actual training parameters are:\n%s', params.as_dict())
  model_dir = os.path.join(
      FLAGS.model_dir,
      'trial_' + hypertune_utils.get_trial_id_from_environment(),
  )
  logging.info('model_dir in this trial is: %s', model_dir)
  if 'train' in FLAGS.mode:
    # Pure eval modes do not output yaml files. Otherwise continuous eval job
    # may race against the train job for writing the same file.
    train_utils.serialize_config(params, model_dir)

  # Sets mixed_precision policy. Using 'mixed_float16' or 'mixed_bfloat16'
  # can have significant impact on model speeds by utilizing float16 in case of
  # GPUs, and bfloat16 in the case of TPUs. loss_scale takes effect only when
  # dtype is float16
  if params.runtime.mixed_precision_dtype:
    performance.set_mixed_precision_policy(params.runtime.mixed_precision_dtype)
  distribution_strategy = distribute_utils.get_distribution_strategy(
      distribution_strategy=params.runtime.distribution_strategy,
      all_reduce_alg=params.runtime.all_reduce_alg,
      num_gpus=params.runtime.num_gpus,
      tpu_address=params.runtime.tpu,
  )
  with distribution_strategy.scope():
    task = task_factory.get_task(params.task, logging_dir=model_dir)

  train_lib.run_experiment(
      distribution_strategy=distribution_strategy,
      task=task,
      mode=FLAGS.mode,
      params=params,
      model_dir=model_dir,
  )

  train_utils.save_gin_config(FLAGS.mode, model_dir)

  eval_metric_name = get_best_eval_metric(_OBJECTIVE.value, params)

  eval_filepath = os.path.join(
      model_dir, constants.BEST_CKPT_DIRNAME, constants.BEST_CKPT_EVAL_FILENAME
  )
  logging.info('Load eval metrics from: %s.', eval_filepath)
  wait_for_evaluation_file(eval_filepath, _MAX_EVAL_WAIT_TIME.value)

  with tf.io.gfile.GFile(eval_filepath, 'rb') as f:
    eval_metric_results = json.load(f)
    logging.info('eval metrics are: %s.', eval_metric_results)
    if (
        eval_metric_name in eval_metric_results
        and constants.BEST_CKPT_STEP_NAME in eval_metric_results
    ):
      hp_metric = eval_metric_results[eval_metric_name]
      hp_step = int(eval_metric_results[constants.BEST_CKPT_STEP_NAME])
      hpt = hypertune.HyperTune()
      hpt.report_hyperparameter_tuning_metric(
          hyperparameter_metric_tag=constants.HP_METRIC_TAG,
          metric_value=hp_metric,
          global_step=hp_step,
      )
      logging.info(
          'Send HP metric: %f and steps %d to hyperparameter tuning.',
          hp_metric,
          hp_step,
      )
    else:
      logging.info(
          'Either %s or %s is not included in the evaluation results: %s.',
          eval_metric_name,
          constants.BEST_CKPT_STEP_NAME,
          eval_metric_results,
      )


if __name__ == '__main__':
  tfm_flags.define_flags()
  flags.mark_flags_as_required(['experiment', 'mode', 'model_dir'])
  app.run(main)
