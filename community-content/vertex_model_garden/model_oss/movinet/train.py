"""Main executable for MoViNet docker."""

import json
import os
from typing import Sequence, Any

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
# Import movinet libraries to register the backbone and model into tf.vision
# model garden factory.
# pylint: disable=unused-import
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model
from official.vision import registry_imports
# pylint: enable=unused-import


FLAGS = flags.FLAGS

_FILE_TYPE_TFRECORD = 'tfrecord'

_LEARNING_RATE = flags.DEFINE_float(
    'learning_rate', None, 'The learning rate of this training job.'
)

_NUM_CLASSES = flags.DEFINE_integer(
    'num_classes', None, 'The number of classes.'
)

_INIT_CHECKPOINT = flags.DEFINE_string(
    'init_checkpoint', None, 'The initial checkpoint of this training job.'
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

_SHUFFLE_BUFFER_SIZE = flags.DEFINE_integer(
    'shuffle_buffer_size', None, 'Shuffle buffer size.'
)

_TRAIN_STEPS = flags.DEFINE_integer('train_steps', None, 'Train steps.')
_LOG_LEVEL = flags.DEFINE_enum(
    'log_level',
    'INFO',
    ['FATAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
    'Log level.',
)


def parse_params() -> Any:
  """Parses parameters."""
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_params)
  params = train_utils.parse_configuration(FLAGS, lock_return=False)
  if _INIT_CHECKPOINT.value:
    params.task.init_checkpoint = _INIT_CHECKPOINT.value
    params.task.init_checkpoint_modules = 'backbone'
  if _NUM_CLASSES.value:
    params.task.model.num_classes = _NUM_CLASSES.value
    params.task.train_data.num_classes = _NUM_CLASSES.value
    params.task.validation_data.num_classes = _NUM_CLASSES.value
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
  if _SHUFFLE_BUFFER_SIZE.value:
    params.task.train_data.shuffle_buffer_size = _SHUFFLE_BUFFER_SIZE.value
  if _TRAIN_STEPS.value:
    params.trainer.train_steps = _TRAIN_STEPS.value
  if _LEARNING_RATE.value:
    logging.info('Updating learning_rate: %s', _LEARNING_RATE.value)
    # Use `get` method of train_utils.hyperparams.OneOfConfig to get learning
    # rate config.
    learning_rate = params.trainer.optimizer_config.learning_rate.get()
    if hasattr(learning_rate, 'initial_learning_rate'):
      learning_rate.initial_learning_rate = _LEARNING_RATE.value
    else:
      logging.warning('Cannot set learning rate for %s', learning_rate)
  # Set default params for best checkpoints.
  params.trainer.best_checkpoint_export_subdir = constants.BEST_CKPT_DIRNAME
  params.trainer.best_checkpoint_metric_comp = constants.BEST_CKPT_METRIC_COMP
  params.trainer.best_checkpoint_eval_metric = (
      constants.VIDEO_CLASSIFICATION_BEST_EVAL_METRIC
  )
  return params


def main(argv: Sequence[str]) -> None:
  logging.set_verbosity(_LOG_LEVEL.value)
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  params = parse_params()
  logging.info('The actual training parameters are:\n%s', params.as_dict())
  model_dir: str = os.path.join(
      FLAGS.model_dir,
      constants.TRIAL_PREFIX + hypertune_utils.get_trial_id_from_environment(),
  )
  logging.info('model_dir: %s', model_dir)

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

  # Create task and run experiment.
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

  eval_metric_name = constants.VIDEO_CLASSIFICATION_BEST_EVAL_METRIC

  eval_filepath = os.path.join(
      model_dir, constants.BEST_CKPT_DIRNAME, constants.BEST_CKPT_EVAL_FILENAME
  )
  logging.info('Load eval metrics from: %s.', eval_filepath)

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
  app.run(main)
