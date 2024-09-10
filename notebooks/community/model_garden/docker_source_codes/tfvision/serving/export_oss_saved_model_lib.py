r"""Vision models export utility function for serving/inference."""

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import tensorflow as tf

from tfvision.serving import detection
from tfvision.serving import image_classification
from tfvision.serving import semantic_segmentation_export_module_lib as isg_export_lib
from util import constants
from official.core import config_definitions as cfg
from official.core import export_base
from official.projects.yolo.configs import yolo as yolo_config
from official.projects.yolo.configs import yolov7 as yolov7_config
from official.projects.yolo.serving import export_module_factory as yolo_export_module_factory


def export_inference_graph(
    input_type: str,
    batch_size: Optional[int],
    input_image_size: List[int],
    params: cfg.ExperimentConfig,
    checkpoint_path: str,
    export_dir: str,
    label_map_path: Optional[str] = None,
    label_path: Optional[str] = None,
    num_channels: Optional[int] = 3,
    export_module: Optional[export_base.ExportModule] = None,
    export_saved_model_subdir: Optional[str] = None,
    save_options: Optional[tf.saved_model.SaveOptions] = None,
    checkpoint: Optional[tf.train.Checkpoint] = None,
    input_name: Optional[str] = None,
    function_keys: Optional[Union[List[str], Dict[str, str]]] = None,
    objective: Optional[str] = None,
):
  """Exports inference graph for the model specified in the exp config.

  Saved model is stored at export_dir/saved_model, checkpoint is saved
  at export_dir/checkpoint, and params is saved at export_dir/params.yaml.

  Args:
    input_type: Input type must be `image_bytes`.
    batch_size: 'int', or None.
    input_image_size: List or Tuple of height and width.
    params: Experiment params.
    checkpoint_path: Trained checkpoint path or directory.
    export_dir: CNS export directory path.
    label_map_path: Labelmap proto file path.
    label_path: Label file path.
    num_channels: The number of input image channels.
    export_module: Optional export module to be used instead of using params to
      create one. If None, the params will be used to create an export module.
    export_saved_model_subdir: Optional subdirectory under export_dir to store
      saved model.
    save_options: `SaveOptions` for `tf.saved_model.save`.
    checkpoint: An optional tf.train.Checkpoint. If provided, the export module
      will use it to read the weights.
    input_name: The input tensor name, default at `None` which produces input
      tensor name `inputs`.
    function_keys: a list of string keys to retrieve pre-defined serving
      signatures. The signaute keys will be set with defaults. If a dictionary
      is provided, the values will be used as signature keys.
    objective: The objective of the training job.
  """
  if export_saved_model_subdir:
    output_saved_model_directory = os.path.join(export_dir,
                                                export_saved_model_subdir)
  else:
    output_saved_model_directory = export_dir

  if not export_module:
    if objective == constants.OBJECTIVE_IMAGE_CLASSIFICATION:
      export_module = image_classification.ClassificationModule(
          params=params,
          batch_size=batch_size,
          input_image_size=input_image_size,
          input_type=input_type,
          num_channels=num_channels,
          input_name=input_name,
          label_path=label_path,
      )
    elif objective == constants.OBJECTIVE_IMAGE_OBJECT_DETECTION:
      # If experiment is YOLO object detection, loads Yolo detection module.
      if isinstance(
          params.task, (yolo_config.YoloTask, yolov7_config.YoloV7Task)
      ):
        export_module = yolo_export_module_factory.get_export_module(
            params=params,
            batch_size=batch_size,
            input_image_size=input_image_size,
            input_type=input_type,
            num_channels=num_channels,
            input_name=input_name,
        )
      else:
        export_module = detection.DetectionModule(
            params=params,
            batch_size=batch_size,
            input_image_size=input_image_size,
            input_type=input_type,
            num_channels=num_channels,
            input_name=input_name,
            label_map_path=label_map_path,
        )
    elif objective == constants.OBJECTIVE_IMAGE_SEGMENTATION:
      export_module = isg_export_lib.OssSegmentationModule(
          params=params,
          batch_size=batch_size,
          input_image_size=input_image_size,
          input_type=input_type,
          num_channels=num_channels,
          input_name=input_name,
      )
    else:
      raise ValueError(
          'Export module not implemented for objective {}.'.format(objective)
      )

  export_base.export(
      export_module,
      function_keys=function_keys if function_keys else [input_type],
      export_savedmodel_dir=output_saved_model_directory,
      checkpoint=checkpoint,
      checkpoint_path=checkpoint_path,
      timestamped=False,
      save_options=save_options)


def get_best_oss_trial(
    model_dir: str, max_trial_count: int, evaluation_metric: str
) -> Tuple[str, Dict[str, Any]]:
  """Export models from TF checkpoints to TF saved model format.

  Args:
    model_dir: Path of directory to store checkpoints and metric summaries.
    max_trial_count: The desired total number of trials.
    evaluation_metric: The evaluation metric to use (ie. accuracy).

  Returns:
  """
  best_trial_dir = ''
  best_trial_evaluation_results = {}
  best_performance = -1
  trial_file_count = 0
  for i in range(max_trial_count):
    current_trial = i + 1
    current_trial_dir = os.path.join(model_dir, 'trial_' + str(current_trial))
    current_trial_best_ckpt_dir = os.path.join(current_trial_dir, 'best_ckpt')
    current_trial_best_ckpt_evaluation_filepath = os.path.join(
        current_trial_best_ckpt_dir, 'info.json'
    )
    if tf.io.gfile.exists(current_trial_best_ckpt_evaluation_filepath):
      trial_file_count += 1
      with tf.io.gfile.GFile(
          current_trial_best_ckpt_evaluation_filepath, 'rb'
      ) as f:
        eval_metric_results = json.load(f)
        current_performance = eval_metric_results[evaluation_metric]
        if current_performance > best_performance:
          best_performance = current_performance
          best_trial_dir = current_trial_dir
          best_trial_evaluation_results = eval_metric_results

  if not trial_file_count:
    raise ValueError('None of the best checkpoint paths exist.')

  return best_trial_dir, best_trial_evaluation_results
