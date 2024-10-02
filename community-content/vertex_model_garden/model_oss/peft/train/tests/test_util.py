"""Test util class."""

import datetime
import os
import signal
import subprocess
import sys

from absl import flags
from absl import logging
from absl.testing import parameterized
import docker_command_builder as docker_cmd_builder

_DOCKER_URI = flags.DEFINE_string(
    'docker_uri', None, 'docker image uri', required=True
)

_DRY_RUN = flags.DEFINE_bool('dry_run', False, 'dry-run the commands')

_LOCAL_INPUT_DIR = flags.DEFINE_string(
    'local_input_dir',
    os.path.expanduser('~/test_input'),
    'local directory for storing input data.',
)

_LOCAL_OUTPUT_DIR = flags.DEFINE_string(
    'local_output_dir',
    '/tmp',
    'local directory for storing test output.',
)


_GCS_INPUT_DIR = flags.DEFINE_string(
    'gcs_input_dir',
    'gs://peft-docker-test',
    'GCS directory that stores model checkpoint, dataset and etc.',
)

_GCS_OUTPUT_DIR = flags.DEFINE_string(
    'gcs_output_dir',
    'gs://peft-docker-test/output',
    'GCS directory that stores test output.',
)


class TestBase(parameterized.TestCase):
  """Test base class that defines how to run commands."""

  def setUp(self):
    super().setUp()

    self.docker_builder = docker_cmd_builder.DockerCommandBuilder(
        _DOCKER_URI.value
    )
    self.docker_builder.add_mount_map(
        os.path.expanduser('~'), os.path.expanduser('~')
    )
    self.docker_builder.add_mount_map(
        self.local_input_dir(), self.local_input_dir()
    )

    self.task_cmd_builder = None

  def cmd(self):
    return self.docker_builder.build_cmd() + self.task_cmd_builder.build_cmd()

  def run_cmd(self) -> int:
    logging.info('running command: \n%s', ' \\\n'.join(self.cmd()))
    if _DRY_RUN.value:
      return 0

    p = subprocess.Popen(self.cmd(), stdout=sys.stdout, stderr=sys.stderr)
    try:
      unused_output, unused_error = p.communicate()
      return p.returncode
    except KeyboardInterrupt:
      p.send_signal(signal.SIGINT)
      return 0

  def gcs_output_dir(self):
    return _GCS_OUTPUT_DIR.value

  def local_output_dir(self):
    return _LOCAL_OUTPUT_DIR.value

  def local_input_dir(self):
    return _LOCAL_INPUT_DIR.value


def get_timestamp():
  return datetime.datetime.now(datetime.timezone.utc).strftime(
      '%Y%m%d_%H%M%S%Z'
  )


def get_test_data_path(name: str, download: bool = True) -> str:
  """Gets test data path.

  Args:
    name: name of the test data
    download: if True, then download data from GCS and returns its local path.

  Returns:
    test data path.
  """

  def _download_from_gcs(name):
    if not os.path.exists(_LOCAL_INPUT_DIR.value):
      os.mkdir(_LOCAL_INPUT_DIR.value)
    subprocess.check_output([
        'gsutil',
        '-m',
        'cp',
        '-r',
        os.path.join(_GCS_INPUT_DIR.value, name),
        _LOCAL_INPUT_DIR.value,
    ])

  if not download:
    return os.path.join(_GCS_INPUT_DIR.value, name)

  local_data = os.path.join(_LOCAL_INPUT_DIR.value, name)
  if not os.path.exists(local_data):
    _download_from_gcs(name)

  return local_data


def get_pretrained_model_id(model_id: str) -> str:
  # If `model_id` contains `/`, it is assumed to be HF model or model from GCS.
  if '/' in model_id:
    return model_id

  return get_test_data_path(model_id, download=True)
