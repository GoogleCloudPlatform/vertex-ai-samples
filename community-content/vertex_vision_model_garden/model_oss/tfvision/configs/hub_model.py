"""Tf-hub model configuration definition for AutoML Vision ICN.."""

import os

from tfvision.configs import backbones
from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import optimization
from official.vision.configs import image_classification

_HANDLE = 'https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_m/feature_vector/2'  # pylint: disable=line-too-long
_COCA_HANDLE = None
_INPUT_SIZE = [480, 480, 3]
_MEAN_RGB = 0.0
_STDDEV_RGB = 255.0


# pylint is unable to handle dataclasses constructor arguments correctly.
# pylint: disable=unexpected-keyword-arg
@exp_factory.register_config_factory('hub_model')
def hub_model() -> cfg.ExperimentConfig:
  """Gets experimental configs for tf-hub models."""

  batch_size = 8
  train_steps = 625000
  steps_per_loop = 1250
  return cfg.ExperimentConfig(
      task=image_classification.ImageClassificationTask(
          model=image_classification.ImageClassificationModel(
              num_classes=1000,
              input_size=_INPUT_SIZE,
              backbone=backbones.Backbone(
                  type='hub_model',
                  hub_model=backbones.HubModel(
                      handle=_HANDLE, mean_rgb=_MEAN_RGB, stddev_rgb=_STDDEV_RGB
                  ),
              ),
              dropout_rate=0.0,
          ),
          losses=image_classification.Losses(
              l2_weight_decay=0.0, label_smoothing=0.1, one_hot=True
          ),
          train_data=image_classification.DataConfig(
              input_path=os.path.join(
                  image_classification.IMAGENET_INPUT_PATH_BASE, 'train*'
              ),
              aug_type=None,
              dtype='float32',
              global_batch_size=batch_size,
              is_training=True,
              decode_jpeg_only=False,
          ),
          validation_data=image_classification.DataConfig(
              input_path=os.path.join(
                  image_classification.IMAGENET_INPUT_PATH_BASE, 'valid*'
              ),
              dtype='float32',
              global_batch_size=batch_size,
              is_training=False,
              decode_jpeg_only=False,
              drop_remainder=False,
          ),
      ),
      trainer=cfg.TrainerConfig(
          best_checkpoint_eval_metric='accuracy',
          best_checkpoint_export_subdir='best_ckpt',
          best_checkpoint_metric_comp='higher',
          optimizer_config=optimization.OptimizationConfig(
              learning_rate=optimization.LrConfig(
                  type='cosine',
                  cosine=optimization.lr_cfg.CosineLrConfig(
                      decay_steps=train_steps, initial_learning_rate=0.001
                  ),
              ),
              optimizer=optimization.OptimizerConfig(
                  type='sgd', sgd=optimization.SGDConfig(momentum=0.9)
              ),
          ),
          checkpoint_interval=steps_per_loop,
          steps_per_loop=steps_per_loop,
          summary_interval=steps_per_loop,
          validation_interval=steps_per_loop,
          train_steps=train_steps,
          validation_steps=-1,
      ),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None',
      ],
  )


@exp_factory.register_config_factory('coca')
def coca() -> cfg.ExperimentConfig:
  """Gets experimental configs for tf-hub models."""

  batch_size = 8
  train_steps = 625000
  steps_per_loop = 1250
  return cfg.ExperimentConfig(
      task=image_classification.ImageClassificationTask(
          model=image_classification.ImageClassificationModel(
              num_classes=1000,
              input_size=[288, 288, 3],
              backbone=backbones.Backbone(
                  type='hub_model',
                  hub_model=backbones.HubModel(
                      handle=_COCA_HANDLE,
                      trainable=False,
                      mean_rgb=0.0,
                      stddev_rgb=255.0,
                  ),
              ),
              dropout_rate=0.0,
          ),
          losses=image_classification.Losses(
              l2_weight_decay=0.0, label_smoothing=0.1, one_hot=True
          ),
          train_data=image_classification.DataConfig(
              input_path=os.path.join(
                  image_classification.IMAGENET_INPUT_PATH_BASE, 'train*'
              ),
              aug_type=None,
              dtype='float32',
              global_batch_size=batch_size,
              is_training=True,
              decode_jpeg_only=False,
          ),
          validation_data=image_classification.DataConfig(
              input_path=os.path.join(
                  image_classification.IMAGENET_INPUT_PATH_BASE, 'valid*'
              ),
              dtype='float32',
              global_batch_size=batch_size,
              is_training=False,
              decode_jpeg_only=False,
              drop_remainder=False,
          ),
      ),
      trainer=cfg.TrainerConfig(
          best_checkpoint_eval_metric='accuracy',
          best_checkpoint_export_subdir='best_ckpt',
          best_checkpoint_metric_comp='higher',
          optimizer_config=optimization.OptimizationConfig(
              learning_rate=optimization.LrConfig(
                  type='cosine',
                  cosine=optimization.lr_cfg.CosineLrConfig(
                      decay_steps=train_steps, initial_learning_rate=0.001
                  ),
              ),
              optimizer=optimization.OptimizerConfig(
                  type='sgd', sgd=optimization.SGDConfig(momentum=0.9)
              ),
          ),
          checkpoint_interval=steps_per_loop,
          steps_per_loop=steps_per_loop,
          summary_interval=steps_per_loop,
          validation_interval=steps_per_loop,
          train_steps=train_steps,
          validation_steps=-1,
      ),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None',
      ],
  )
