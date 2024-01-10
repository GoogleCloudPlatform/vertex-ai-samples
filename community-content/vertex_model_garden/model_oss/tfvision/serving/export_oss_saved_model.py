"""Export OSS TfVision models."""
import os

from absl import app
from absl import flags
from absl import logging

from google.cloud import aiplatform as aip

# pylint: disable=line-too-long,unused-import
from tfvision import registry_imports as vision_registry_imports
from tfvision.serving import automl_constants
from tfvision.serving import export_oss_saved_model_lib as export_automl_oss_saved_model_lib
from util import constants
from official.core import exp_factory
from official.modeling import hyperparams
from official.projects.maxvit import registry_imports as maxvit_imports
from official.projects.yolo.common import registry_imports as yolo_imports
from official.vision import registry_imports
from official.vision.serving import export_saved_model_lib as export_oss_saved_model_lib
# pylint: enable=line-too-long, unused-import

_PARAMS_OVERRIDE_IOD = """
task:
  export_config:
    output_normalized_coordinates: true
    cast_num_detections_to_float: true
    cast_detection_classes_to_float: true
  model:
    detection_generator:
      nms_version: batched"""

_PARAMS_OVERRIDE_YOLO = """
task:
  export_config:
    output_normalized_coordinates: true
    cast_num_detections_to_float: true
    cast_detection_classes_to_float: true
  model:
    detection_generator:
      nms_version: v2"""


_PARAMS_OVERRIDE_ISG = """
task:
  export_config:
    rescale_output: true"""

_YOLO_KEY = 'yolo'

_OBJECTIVE = flags.DEFINE_enum(
    'objective',
    None,
    [
        constants.OBJECTIVE_IMAGE_CLASSIFICATION,
        constants.OBJECTIVE_IMAGE_OBJECT_DETECTION,
        constants.OBJECTIVE_IMAGE_SEGMENTATION,
    ],
    'The objective of this training job.',
)

# Cloud AI platform HPT related parameter
_PROJECT_NAME = flags.DEFINE_string(
    'project_name', None, 'Training vizier study name.'
)
_LOCATION = flags.DEFINE_string('location', None, 'Vizier study owner.')
_HPT_JOB_ID = flags.DEFINE_string('hpt_job_id', None, 'HPT job id.')
_HPT_RESULT_DIR = flags.DEFINE_string(
    'hpt_result_dir', None, 'HPT job result directory.'
)
_USE_BIGSTORE = flags.DEFINE_bool(
    'use_bigstore', None, 'Whether to use bigstore in hub model path.'
)

# TfVision related inputs.
_EXPERIMENT = flags.DEFINE_string(
    'experiment', None, 'experiment type, e.g. retinanet_resnetfpn_coco')
_EXPORT_DIR = flags.DEFINE_string('export_dir', None, 'The export directory.')
_CHECKPOINT_PATH = flags.DEFINE_string('checkpoint_path', None,
                                       'Checkpoint path.')
_LABEL_MAP_PATH = flags.DEFINE_string('label_map_path', None,
                                      'Path to the labelmap proto file.')
_LABEL_PATH = flags.DEFINE_string(
    'label_path', None, 'Path to the image classification label file.')
_CONFIG_FILE = flags.DEFINE_multi_string(
    'config_file',
    default=None,
    help=(
        'YAML/JSON files which specifies overrides. The override order follows'
        ' the order of args. Note that each file can be used as an override'
        ' template to override the default parameters specified in Python. If'
        ' the same parameter is specified in both `--config_file`.'
    ),
)
_INPUT_IMAGE_SIZE = flags.DEFINE_string(
    'input_image_size', '224,224',
    'The comma-separated string of two integers representing the height,width '
    'of the input to the model.')

# Fixed inputs.
_IMAGE_TYPE = flags.DEFINE_string(
    'input_type',
    'image_bytes',
    'One of `image_tensor`, `image_bytes`, `tf_example` and `tflite`.',
)
_EXPORT_SAVED_MODEL_SUBDIR = flags.DEFINE_string(
    'export_saved_model_subdir', 'saved_model',
    'The subdirectory for saved model.')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 1, 'The batch size.')
_INPUT_NAME = flags.DEFINE_string(
    'input_name',
    'encoded_image',
    (
        'Input tensor name in signature def. Default at None which'
        'produces input tensor name `inputs`.'
    ),
)
_MAX_TRIAL_COUNT = flags.DEFINE_integer(
    'max_trial_count', None, 'The desired total number of trials.'
)
_EVALUATION_METRIC = flags.DEFINE_string(
    'evaluation_metric',
    None,
    'The evaluation metric to use (e.g. accuracy).',
)


def change_handle(params: hyperparams.ParamsDict) -> hyperparams.ParamsDict:
  """Changes the prefix of the `handle` path in the `model.backbone.hub_model` sub-dictionary from gs:// to /bigstore/.

  Args:
    params: hyperparams.ParamsDict object containing experiment config
      information.

  Returns:
    params: hyperparams.ParamsDict.
  """

  params.task.model.backbone.hub_model.handle = (
      params.task.model.backbone.hub_model.handle.replace(
          'gs://', '/bigstore/', 1
      )
  )

  return params


def get_best_hpt_trials(
    project: str, location: str, hpt_job_id: str, hpt_result_dir: str
) -> str:
  """Select best trials by cloud ai platorm hyperparameter tuning.

  Args:
   project: GCP project name.
   location: Hyperparameter job location.
   hpt_job_id: Hyperparameter job id.
   hpt_result_dir: HPT job result GCS directory.

  Returns:
    Trial Id of the best performing trial.
  """

  aip.init(project=project, location=location)
  job_response = aip.HyperparameterTuningJob.get(resource_name=hpt_job_id)
  max_value = -1
  best_trial_id = -1
  trials = list(job_response._gca_resource.trials)  # pylint: disable=protected-access
  for trial in trials:
    if trial.final_measurement.metrics[0].metric_id != constants.HP_METRIC_TAG:
      continue
    if trial.final_measurement.metrics[0].value > max_value:
      best_trial_id = trial.id
      max_value = trial.final_measurement.metrics[0].value
  if best_trial_id == -1:
    raise ValueError('No valid completed trials.')
  best_model_dir = os.path.join(
      hpt_result_dir, constants.TRIAL_PREFIX + str(best_trial_id)
  )
  logging.info(
      'Best model directory: %s with performance: %s', best_model_dir, max_value
  )
  return best_model_dir


def main(_) -> None:
  if (
      _MAX_TRIAL_COUNT.present
      and _EVALUATION_METRIC.present
      and _CONFIG_FILE.present
  ):
    best_ckpt_dir, _ = export_automl_oss_saved_model_lib.get_best_oss_trial(
        _CHECKPOINT_PATH.value, _MAX_TRIAL_COUNT.value, _EVALUATION_METRIC.value
    )
    config_filepath = _CONFIG_FILE.value
  elif _CHECKPOINT_PATH.present and _CONFIG_FILE.present:
    best_ckpt_dir = _CHECKPOINT_PATH.value
    config_filepath = _CONFIG_FILE.value
  elif (
      _PROJECT_NAME.present
      and _LOCATION.present
      and _HPT_JOB_ID.present
      and _HPT_RESULT_DIR.present
  ):
    # Reads HPT results by project and location and hpt_job_id.
    best_ckpt_dir = get_best_hpt_trials(
        _PROJECT_NAME.value,
        _LOCATION.value,
        _HPT_JOB_ID.value,
        _HPT_RESULT_DIR.value,
    )
    config_filepath = [
        os.path.join(best_ckpt_dir, automl_constants.CFG_FILENAME)
    ]
  else:
    raise ValueError('No checkpoint path or HTP Job parameters given.')

  params = exp_factory.get_exp_config(_EXPERIMENT.value)
  for config_file in config_filepath or []:
    params = hyperparams.override_params_dict(
        params, config_file, is_strict=False
    )
  if _OBJECTIVE.value == constants.OBJECTIVE_IMAGE_OBJECT_DETECTION:
    if _YOLO_KEY in _EXPERIMENT.value:
      params = hyperparams.override_params_dict(
          params, _PARAMS_OVERRIDE_YOLO, is_strict=False
      )
    else:
      params = hyperparams.override_params_dict(
          params, _PARAMS_OVERRIDE_IOD, is_strict=False
      )
  elif _OBJECTIVE.value == constants.OBJECTIVE_IMAGE_SEGMENTATION:
    params = hyperparams.override_params_dict(
        params, _PARAMS_OVERRIDE_ISG, is_strict=True
    )

  if _USE_BIGSTORE.value:
    params = change_handle(params)

  params.validate()
  params.lock()

  if best_ckpt_dir and not best_ckpt_dir.endswith(
      params.trainer.best_checkpoint_export_subdir
  ):
    best_ckpt_dir = os.path.join(
        best_ckpt_dir, params.trainer.best_checkpoint_export_subdir
    )

  if (
      _LABEL_MAP_PATH.value
      or _LABEL_PATH.value
      or _OBJECTIVE.value == constants.OBJECTIVE_IMAGE_SEGMENTATION
  ):
    export_automl_oss_saved_model_lib.export_inference_graph(
        input_type=_IMAGE_TYPE.value,
        batch_size=_BATCH_SIZE.value,
        input_image_size=[int(x) for x in _INPUT_IMAGE_SIZE.value.split(',')],
        params=params,
        checkpoint_path=best_ckpt_dir,
        label_map_path=_LABEL_MAP_PATH.value,
        label_path=_LABEL_PATH.value,
        export_dir=_EXPORT_DIR.value,
        export_saved_model_subdir=_EXPORT_SAVED_MODEL_SUBDIR.value,
        input_name=_INPUT_NAME.value,
        objective=_OBJECTIVE.value,
    )
  elif _YOLO_KEY in _EXPERIMENT.value:
    export_automl_oss_saved_model_lib.export_inference_graph(
        input_type=_IMAGE_TYPE.value,
        batch_size=_BATCH_SIZE.value,
        input_image_size=[int(x) for x in _INPUT_IMAGE_SIZE.value.split(',')],
        params=params,
        checkpoint_path=best_ckpt_dir,
        export_dir=_EXPORT_DIR.value,
        export_saved_model_subdir=_EXPORT_SAVED_MODEL_SUBDIR.value,
        input_name=_INPUT_NAME.value,
        objective=_OBJECTIVE.value,
    )
  else:
    export_oss_saved_model_lib.export_inference_graph(
        input_type=_IMAGE_TYPE.value,
        batch_size=_BATCH_SIZE.value,
        input_image_size=[int(x) for x in _INPUT_IMAGE_SIZE.value.split(',')],
        params=params,
        checkpoint_path=best_ckpt_dir,
        export_dir=_EXPORT_DIR.value,
        export_saved_model_subdir=_EXPORT_SAVED_MODEL_SUBDIR.value,
        input_name=_INPUT_NAME.value,
    )


if __name__ == '__main__':
  app.run(main)
