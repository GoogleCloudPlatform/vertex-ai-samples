"""Entrypoint for peft train docker.

Dispatches to different scripts based on `task` type.

For task type in `_TASK_TO_SCRIPT`, if `--config_file` is specified, the script
will dispatch the call to `accelerate`, which is friendly for multi-GPU
environment. Otherwise, `python3` is used.
"""

import argparse
import json
import os
import subprocess
from typing import List, Optional, Sequence
from absl import app
from absl import flags
from absl import logging
from util import dataset_validation_util
from vertex_vision_model_garden_peft.train.vmg import utils
from util import constants
from util import hypertune_utils


_TEXT_TO_IMAGE_TASKS_SCRIPTS = {
    constants.TEXT_TO_IMAGE: 'text_to_image/train_text_to_image.py',
    constants.TEXT_TO_IMAGE_LORA: 'text_to_image/train_text_to_image_lora.py',
    constants.TEXT_TO_IMAGE_DREAMBOOTH: 'dreambooth/train_dreambooth.py',
    constants.TEXT_TO_IMAGE_DREAMBOOTH_LORA: (
        'dreambooth/train_dreambooth_lora.py'
    ),
    constants.TEXT_TO_IMAGE_DREAMBOOTH_LORA_SDXL: (
        'dreambooth/train_dreambooth_lora_sdxl.py'
    ),
}

_TASK_TO_SCRIPT = {
    constants.INSTRUCT_LORA: (
        'vertex_vision_model_garden_peft/train/vmg/instruct_lora.py'
    ),
    constants.MERGE_CAUSAL_LANGUAGE_MODEL_LORA: 'vertex_vision_model_garden_peft/train/vmg/merge_causal_language_model_lora.py',
    constants.QUANTIZE_MODEL: (
        'vertex_vision_model_garden_peft/train/vmg/quantize_model.py'
    ),
    constants.SEQUENCE_CLASSIFICATION_LORA: 'vertex_vision_model_garden_peft/train/vmg/sequence_classification_lora.py',
    constants.VALIDATE_DATASET_WITH_TEMPLATE: 'vertex_vision_model_garden_peft/train/vmg/validate_dataset_with_template.py',
}


def launch_script_cmd(
    script: str,
    config_file: Optional[str],
    accelerate_args: argparse.Namespace = argparse.Namespace(),
) -> List[str]:
  """Returns the command to launch the script."""
  if config_file:
    cmd = [
        'accelerate',
        'launch',
        '--config_file={}'.format(config_file),
    ]
  else:
    cmd = ['python3']

  _append_args_to_command_in_place(accelerate_args, cmd)
  cmd.append(script)

  return cmd


def _get_accelerate_args() -> argparse.Namespace:
  """Returns the accelerate args."""
  # For the format of the cluster spec, see
  # https://cloud.google.com/vertex-ai/docs/training/distributed-training#cluster-spec-format # pylint: disable=line-too-long
  cluster_spec = os.getenv('CLUSTER_SPEC', default=None)
  if not cluster_spec:
    return argparse.Namespace()
  logging.info('CLUSTER_SPEC: %s', cluster_spec)

  cluster_data = json.loads(cluster_spec)
  if (
      'workerpool1' not in cluster_data['cluster']
      or not cluster_data['cluster']['workerpool1']
  ):
    return argparse.Namespace()

  # Get primary node info
  primary_node = cluster_data['cluster']['workerpool0'][0]
  logging.info('primary node: %s', primary_node)
  primary_node_addr, primary_node_port = primary_node.split(':')
  logging.info('primary node address: %s', primary_node_addr)
  logging.info('primary node port: %s', primary_node_port)

  # Determine node rank of this machine
  workerpool = cluster_data['task']['type']
  if workerpool == 'workerpool0':
    node_rank = 0
  elif workerpool == 'workerpool1':
    # Add 1 for the primary node, since `index` is the index of workerpool1.
    node_rank = cluster_data['task']['index'] + 1
  else:
    raise ValueError(
        'Only workerpool0 and workerpool1 are supported. Unknown workerpool:'
        f' {workerpool}'
    )
  logging.info('node rank: %s', node_rank)

  # Calculate total nodes
  num_worker_nodes = len(cluster_data['cluster']['workerpool1'])
  num_nodes = num_worker_nodes + 1  # Add 1 for the primary node
  logging.info('num nodes: %s', num_nodes)

  accelerate_args = argparse.Namespace()
  accelerate_args.machine_rank = node_rank
  accelerate_args.num_machines = num_nodes
  accelerate_args.main_process_ip = primary_node_addr
  accelerate_args.main_process_port = primary_node_port
  accelerate_args.max_restarts = 0
  accelerate_args.monitor_interval = 120

  return accelerate_args


def _append_args_to_command_in_place(
    args: argparse.Namespace, command: List[str]
):
  for key, value in vars(args).items():
    # If not specified, skip.
    if value is not None:
      command.append(f'--{key}={value}')


def _get_train_cmd_and_maybe_merge_cmd(
    task: str, config_file: str, unknown: Sequence[str]
) -> Sequence[Sequence[str]]:
  """Returns the training command and maybe the merge command if applicable."""

  # Only populated when multi-node is used.
  accelerate_args = _get_accelerate_args()
  training_cmd = launch_script_cmd(
      _TASK_TO_SCRIPT[task],
      config_file,
      accelerate_args=accelerate_args,
  )

  # Training only flag.
  train_parser = argparse.ArgumentParser()
  train_parser.add_argument('--output_dir', required=True)
  training_args, unknown = train_parser.parse_known_args(unknown)
  # Checks for `hypertune_utils._ENVIRONMENT_VARIABLE_FOR_TRIAL_ID` env var and
  # appends the trial id if it exists.
  training_args.output_dir = hypertune_utils.maybe_append_trial_id(
      dataset_validation_util.force_gcs_fuse_path(training_args.output_dir)
  )

  # Merge only flags.
  merge_parser = argparse.ArgumentParser()
  merge_parser.add_argument('--merge_model_precision_mode')
  merge_parser.add_argument('--executor_input')
  merge_parser.add_argument('--restrict_model_upload_docker_uri')
  merge_parser.add_argument('--merge_base_and_lora_output_dir')
  merge_args, unknown = merge_parser.parse_known_args(unknown)

  # Common flags shared by merging and training.
  common_parser = argparse.ArgumentParser()
  common_parser.add_argument('--pretrained_model_id', required=True)
  common_parser.add_argument('--huggingface_access_token')
  common_args, remaining = common_parser.parse_known_args(unknown)

  # Add flags for training.
  _append_args_to_command_in_place(training_args, training_cmd)
  _append_args_to_command_in_place(common_args, training_cmd)
  training_cmd.extend(remaining)  # Remaining args are passed to training cmd.
  commands = [training_cmd]

  # Only the main node runs merging.
  if (
      merge_args.merge_base_and_lora_output_dir
      and getattr(accelerate_args, 'machine_rank', 0) == 0
  ):
    lora_dir = utils.get_final_checkpoint_path(training_args.output_dir)

    merge_cmd = [
        'WORLD_SIZE=1',  # To ignore other nodes in multi-node setting.
        'python3',
        _TASK_TO_SCRIPT[constants.MERGE_CAUSAL_LANGUAGE_MODEL_LORA],
        f'--finetuned_lora_model_dir={lora_dir}',
    ]
    _append_args_to_command_in_place(merge_args, merge_cmd)
    _append_args_to_command_in_place(common_args, merge_cmd)

    # Run in a conda environment.
    conda_run_cmd = [
        '/bin/bash',
        '-c',
        f'conda run -n merge {" ".join(merge_cmd)}',
    ]
    commands.append(conda_run_cmd)

  return commands


def main(unused_argv: Sequence[str]) -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('--config_file')
  parser.add_argument('--task')
  args, unknown = parser.parse_known_args()

  task = args.task

  if task in _TEXT_TO_IMAGE_TASKS_SCRIPTS:
    # Setup accelerate config before running trainer.
    config_gen_cmd = [
        'python',
        '-c',
        (
            'from accelerate.utils import write_basic_config;'
            ' write_basic_config(mixed_precision="fp16")'
        ),
    ]
    task_cmd = [
        'accelerate',
        'launch',
        _TEXT_TO_IMAGE_TASKS_SCRIPTS[task],
    ] + list(map(dataset_validation_util.force_gcs_fuse_path, unknown))
    commands = [config_gen_cmd, task_cmd]
  elif task in [constants.INSTRUCT_LORA]:
    commands = _get_train_cmd_and_maybe_merge_cmd(
        task=task, config_file=args.config_file, unknown=unknown
    )
  else:
    assert task in _TASK_TO_SCRIPT
    cmd = launch_script_cmd(_TASK_TO_SCRIPT[task], args.config_file)
    cmd.extend(unknown)
    commands = [cmd]

  for cmd in commands:
    logging.info('launching task=%s with cmd: \n%s', task, ' \\\n'.join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
  app.run(main, flags_parser=lambda _args: flags.FLAGS(_args, known_only=True))
