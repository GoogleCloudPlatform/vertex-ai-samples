"""Entrypoint for peft train docker.

Dispatches to different scripts based on `task` type.

For task type in `_TASK_TO_SCRIPT`, if `--config_file` is specified, the script
will dispatch the call to `accelerate`, which is friendly for multi-GPU
environment. Otherwise, `python3` is used.
"""

import argparse
from collections.abc import MutableSequence, Sequence
import multiprocessing
import subprocess
import sys
from absl import app
from absl import flags
from absl import logging
from util import dataset_validation_util
from util import cluster_spec
from vertex_vision_model_garden_peft.train.vmg import utils
from util import constants
from util import gcs_syncer
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
    constants.MERGE_CAUSAL_LANGUAGE_MODEL_LORA: (
        'vertex_vision_model_garden_peft/train/vmg/merge_causal_language_model_lora.py'
    ),
    constants.VALIDATE_DATASET_WITH_TEMPLATE: (
        'vertex_vision_model_garden_peft/train/vmg/validate_dataset_with_template.py'
    ),
    constants.RUN_TESTS: 'vertex_vision_model_garden_peft/tests/run_tests.py',
}


def launch_script_cmd(
    script: str,
    config_file: str | None,
    accelerate_args: argparse.Namespace = argparse.Namespace(),
) -> MutableSequence[str]:
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
  primary_node_addr, primary_node_port, node_rank, num_nodes = (
      cluster_spec.get_cluster_spec()
  )
  accelerate_args = argparse.Namespace()
  if num_nodes > 1:
    accelerate_args.machine_rank = node_rank
    accelerate_args.num_machines = num_nodes
    accelerate_args.main_process_ip = primary_node_addr
    accelerate_args.main_process_port = primary_node_port
    accelerate_args.max_restarts = 0
    accelerate_args.monitor_interval = 120

  return accelerate_args


def _append_args_to_command_in_place(
    args: argparse.Namespace, command: MutableSequence[str]
):
  for key, value in vars(args).items():
    # If not specified, skip.
    if value is not None:
      command.append(f'--{key}={value}')


def _get_train_and_maybe_merge_cmd_and_dirs_to_sync(
    task_type: str, config_file: str, unknown: Sequence[str]
) -> Sequence[Sequence[str]]:
  """Returns the training and merge command(if applicable) and dirs to sync.

  Args:
    task_type: The task type.
    config_file: The accelerate config file path.
    unknown: The unknown args which are not recognised by the parser.

  Returns:
    The bash commands to execute and the directories to sync.
  """
  dirs_to_sync = []
  # Only populated when multi-node is used.
  accelerate_args = _get_accelerate_args()
  node_rank = getattr(accelerate_args, 'machine_rank', 0)
  training_cmd = launch_script_cmd(
      _TASK_TO_SCRIPT[task_type],
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

  local_output_dir, gcs_output_dir = gcs_syncer.manage_sync_path(
      training_args.output_dir, node_rank
  )
  training_args.output_dir = local_output_dir
  if gcs_syncer.is_gcs_or_gcsfuse_path(gcs_output_dir):
    dirs_to_sync.append((local_output_dir, gcs_output_dir))

  # Merge only flags.
  merge_parser = argparse.ArgumentParser()
  merge_parser.add_argument('--merge_model_precision_mode')
  merge_parser.add_argument('--merge_base_and_lora_output_dir')
  merge_args, unknown = merge_parser.parse_known_args(unknown)

  if merge_args.merge_base_and_lora_output_dir:
    merge_local_dir, merge_gcs_dir = gcs_syncer.manage_sync_path(
        merge_args.merge_base_and_lora_output_dir, None
    )
    merge_args.merge_base_and_lora_output_dir = merge_local_dir
    if gcs_syncer.is_gcs_or_gcsfuse_path(merge_gcs_dir):
      dirs_to_sync.append((merge_local_dir, merge_gcs_dir))

  # Common flags shared by merging and training.
  common_parser = argparse.ArgumentParser()
  common_parser.add_argument('--pretrained_model_name_or_path', required=True)
  common_parser.add_argument('--huggingface_access_token')
  common_args, remaining = common_parser.parse_known_args(unknown)

  # Add flags for training.
  _append_args_to_command_in_place(training_args, training_cmd)
  _append_args_to_command_in_place(common_args, training_cmd)
  training_cmd.extend(remaining)  # Remaining args are passed to training cmd.
  commands = [training_cmd]

  # Only the main node runs merging.
  if merge_args.merge_base_and_lora_output_dir and node_rank == 0:
    lora_dir = utils.get_final_checkpoint_path(training_args.output_dir)
    lora_local_dir, lora_gcs_dir = gcs_syncer.manage_sync_path(
        lora_dir, node_rank
    )
    if gcs_syncer.is_gcs_or_gcsfuse_path(lora_gcs_dir):
      dirs_to_sync.append((lora_local_dir, lora_gcs_dir))

    merge_cmd = [
        'WORLD_SIZE=1',  # To ignore other nodes in multi-node setting.
        'python3',
        _TASK_TO_SCRIPT[constants.MERGE_CAUSAL_LANGUAGE_MODEL_LORA],
        f'--finetuned_lora_model_dir={lora_local_dir}',
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

  return commands, dirs_to_sync


def _get_merge_cmd_and_dirs_to_sync(
    task_type: str, config_file: str, unknown: Sequence[str]
) -> Sequence[Sequence[str]]:
  """Returns the merge command and dirs to sync.

  Args:
    task_type: The task type.
    config_file: The accelerate config file path.
    unknown: The unknown args which are not recognised by the parser.

  Returns:
    The bash commands to execute and the directories to sync.
  """
  # Merge only flags.
  merge_parser = argparse.ArgumentParser()
  merge_parser.add_argument('--merge_base_and_lora_output_dir')
  merge_args, unknown = merge_parser.parse_known_args(unknown)

  dirs_to_sync = []
  if merge_args.merge_base_and_lora_output_dir:
    merge_local_dir, merge_gcs_dir = gcs_syncer.manage_sync_path(
        merge_args.merge_base_and_lora_output_dir, None
    )
    merge_args.merge_base_and_lora_output_dir = merge_local_dir
    if gcs_syncer.is_gcs_or_gcsfuse_path(merge_gcs_dir):
      dirs_to_sync.append((merge_local_dir, merge_gcs_dir))

  cmd = launch_script_cmd(_TASK_TO_SCRIPT[task_type], config_file)
  _append_args_to_command_in_place(merge_args, cmd)
  cmd.extend(unknown)
  return [cmd], dirs_to_sync


def main(unused_argv: Sequence[str]) -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('--config_file')
  parser.add_argument('--task')
  parser.add_argument('--gcs_rsync_interval_secs', type=int, default=60)
  args, unknown = parser.parse_known_args()

  task = args.task
  dirs_to_sync = None

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
    commands, dirs_to_sync = _get_train_and_maybe_merge_cmd_and_dirs_to_sync(
        task_type=task, config_file=args.config_file, unknown=unknown
    )
  elif task in [constants.MERGE_CAUSAL_LANGUAGE_MODEL_LORA]:
    commands, dirs_to_sync = _get_merge_cmd_and_dirs_to_sync(
        task_type=task, config_file=args.config_file, unknown=unknown
    )
  else:
    assert task in _TASK_TO_SCRIPT
    cmd = launch_script_cmd(_TASK_TO_SCRIPT[task], args.config_file)
    cmd.extend(unknown)
    commands = [cmd]

  rsync_process = None
  mp_queue = multiprocessing.Queue(maxsize=1)
  if dirs_to_sync:
    rsync_process = gcs_syncer.setup_gcs_rsync(
        dirs_to_sync, mp_queue, args.gcs_rsync_interval_secs
    )

  for cmd in commands:
    logging.info('launching task=%s with cmd: \n%s', task, ' \\\n'.join(cmd))
    # Both absl logging and python's logging module writes to stderr by default.
    # Redirect output to stdout on purpose, such that log entries do not get
    # marked as `Error` in Cloud's Log Explorer.
    try:
      subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stdout, check=True)
    except subprocess.CalledProcessError as e:
      if rsync_process is not None and rsync_process.is_alive():
        logging.info('Terminating GCS rsync process.')
        rsync_process.terminate()
      raise e
  if rsync_process is not None:
    gcs_syncer.cleanup_gcs_rsync(rsync_process, mp_queue)


if __name__ == '__main__':
  logging.get_absl_handler().python_handler.stream = sys.stdout
  app.run(main, flags_parser=lambda _args: flags.FLAGS(_args, known_only=True))
