"""Test util class."""

import copy
import dataclasses
import datetime
import inspect
import os
import signal
import subprocess
import sys
from absl import flags
from absl import logging
from absl.testing import parameterized
import command_builder
import immutabledict
import torch

_DOCKER_URI = flags.DEFINE_string('docker_uri', None, 'docker image uri')

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
    'gs://vmg-tuning-docker-test',
    'GCS directory that stores model checkpoint, dataset and etc.',
)

_GCS_OUTPUT_DIR = flags.DEFINE_string(
    'gcs_output_dir',
    'gs://vmg-tuning-docker-test/output',
    'GCS directory that stores test output.',
)

_GCS_TESTDATA_DIR = 'peft-train-image-test'

_THROUGHPUT_TEST_EXCEPTIONS = immutabledict.immutabledict({
    ('bm_deepspeed_zero3_8gpu_gemma-2-9b-it_4bit.txt', '12.0'): float('inf'),
    ('bm_fsdp_8gpu_llama3.1-70b-hf_4bit.txt', '20.0'): float('inf'),
    ('bm_deepspeed_zero2_8gpu_gemma-2-2b-it_bfloat16.txt', '12.0'): 20.0,
    ('bm_deepspeed_zero3_8gpu_gemma-2-2b-it_4bit.txt', '4.0'): 20.0,
    ('bm_deepspeed_zero3_8gpu_gemma-2-27b-it_4bit.txt', '4.0'): 20.0,
})


@dataclasses.dataclass
class BenchmarkStats:
  """Class to store the benchmark result.

  Attributes:
    peak_mem: peak memory in GB.
    throughput: throughput in tokens/sec.
  """

  peak_mem: float
  throughput: float


class TestBase(parameterized.TestCase):
  """Test base class that defines how to run commands."""

  def setUp(self):
    super().setUp()

    # Create a copy of the environment variables
    self.old_env_var = copy.deepcopy(os.environ)
    if _DOCKER_URI.value:
      self.command_builder = command_builder.DockerCommandBuilder(
          _DOCKER_URI.value
      )
    else:
      self.command_builder = command_builder.PythonCommandBuilder()

    self.command_builder.add_mount_map(
        os.path.expanduser('~'), os.path.expanduser('~')
    )
    self.command_builder.add_mount_map(
        self.local_input_dir(), self.local_input_dir()
    )

    self.task_cmd_builder = None

  def tearDown(self):
    super().tearDown()
    # Restore the original environment variables
    os.environ.clear()
    os.environ.update(self.old_env_var)

  def cmd(self):
    return self.command_builder.build_cmd() + self.task_cmd_builder.build_cmd()

  def run_cmd(self) -> int:
    return run_cmd(self.cmd(), output_file=None)

  def gcs_output_dir(self):
    return _GCS_OUTPUT_DIR.value

  def local_input_dir(self):
    """Returns local input dir in host/docker."""
    return _LOCAL_INPUT_DIR.value

  def local_output_dir(self):
    """Returns local output dir in host/docker."""
    return _LOCAL_OUTPUT_DIR.value

  def get_testcase_name(self):
    """Returns the function name at the calling site."""
    # https://docs.python.org/3/library/inspect.html#inspect.FrameInfo
    cur_frame = inspect.currentframe()
    # https://stackoverflow.com/a/17366561
    return cur_frame.f_back.f_code.co_name


def get_timestamp():
  return datetime.datetime.now(datetime.timezone.utc).strftime(
      '%Y%m%d_%H%M%S%Z'
  )


def download_from_gcs(gcs_uri: str, local_dir: str):
  if not os.path.exists(local_dir):
    os.mkdir(local_dir)
  subprocess.check_output([
      'gcloud',
      'storage',
      'cp',
      '-r',
      gcs_uri,
      local_dir,
  ])


def get_test_data_path(name: str, download: bool = True) -> str:
  """Gets test data path.

  Args:
    name: name of the test data
    download: if True, then download data from GCS and returns its local path.

  Returns:
    test data path.
  """
  if not download:
    return os.path.join(_GCS_INPUT_DIR.value, name)

  local_data = os.path.join(_LOCAL_INPUT_DIR.value, name)
  if not os.path.exists(local_data):
    # If `name` is a file in sub-folders, then create the sub-folders under
    # `_LOCAL_INPUT_DIR`.
    local_data_dir = os.path.dirname(local_data)
    if not os.path.exists(local_data_dir):
      os.makedirs(local_data_dir)

    download_from_gcs(os.path.join(_GCS_INPUT_DIR.value, name), local_data_dir)

  return local_data


def run_cmd(cmd: list[str], output_file: str = None) -> int:
  """Runs the command and returns the return code.

  Args:
    cmd: The command to run.
    output_file: The file to write the output to.

  Returns:
    The return code of the command.
  """
  logging.info('running command: \n%s', ' \\\n'.join(cmd))
  if _DRY_RUN.value:
    return 0
  stdout = sys.stdout if output_file is None else open(output_file, 'w')
  p = subprocess.Popen(cmd, stdout=stdout, stderr=sys.stderr)
  try:
    unused_output, unused_error = p.communicate()
    return_code = p.returncode
  except KeyboardInterrupt:
    p.send_signal(signal.SIGINT)
    return_code = 0
  finally:
    if output_file is not None:
      stdout.close()
  return return_code


def get_pretrained_model_name_or_path(model_id: str) -> str:
  # If `model_id` contains `/`, it is assumed to be HF model or model from GCS.
  if '/' in model_id:
    return model_id

  return get_test_data_path(model_id, download=True)


def is_gpu_h100():
  """Checks if the GPU is H100."""
  return 'H100' in torch.cuda.get_device_name()


def is_gpu_a100():
  """Checks if the GPU is A100."""
  return 'A100' in torch.cuda.get_device_name()


def _get_formatted_string(max_seq_length: int) -> str:
  """Returns the formatted string for max_seq_length.

  Args:
    max_seq_length: max sequence length to get the formatted string.

  Returns:
    formatted string for max_seq_length.
  """
  return f'{max_seq_length/1024.0:.1f}'


def get_benchmark_results(
    benchmark_file_path: str, max_seq_length: int
) -> BenchmarkStats:
  """Gets benchmark results from the benchmark file.

  Args:
    benchmark_file_path: path to the benchmark file.
    max_seq_length: max sequence length to get the benchmark results.

  Returns:
    peak_mem: peak memory in GB.
    throughput: throughput in tokens/sec.
  """
  formatted_max_seq_length = _get_formatted_string(max_seq_length)
  peak_mem, throughput = None, None
  with open(benchmark_file_path, 'r') as f:
    for line in f:
      if line.startswith(formatted_max_seq_length):
        metrics = line.split('|')
        try:
          peak_mem = float(metrics[1].strip())
        except ValueError:
          pass
        try:
          throughput = float(metrics[2].strip())
        except ValueError:
          pass
        break
    else:
      logging.error(
          'No metrics found for max_seq_length %s in %s',
          formatted_max_seq_length,
          benchmark_file_path,
      )
  return BenchmarkStats(peak_mem, throughput)


def print_benchmark_file(file_path: str) -> None:
  """Prints the contents of the file.

  Args:
    file_path: path to the file.
  """
  with open(file_path, 'r') as f:
    for line in f:
      logging.info(line.strip())


def print_benchmark_results(
    benchmark_file_path: str, benchmark_type: str
) -> None:
  """Prints the benchmark results.

  Args:
    benchmark_file_path: path to the benchmark file.
    benchmark_type: type of the benchmark.
  """
  benchmark_filename = os.path.basename(benchmark_file_path)
  logging.info('--------------------------------------------------------------')
  logging.info('%s benchmark for %s', benchmark_type, benchmark_filename)
  logging.info('--------------------------------------------------------------')
  print_benchmark_file(benchmark_file_path)


def _calculate_percent_change(
    actual_value: float, expected_value: float
) -> float:
  """Calculates the percent change between the actual and expected values.

  Args:
    actual_value: actual value to compare.
    expected_value: expected value to compare.

  Returns:
    percent change between the actual and expected values.
  """
  return ((actual_value - expected_value) / expected_value) * 100.0


def compare_benchmark_results(
    expected_benchmark_file_path: str,
    actual_benchmark_file_path: str,
    allowed_threshold: float,
    max_seq_length: int,
) -> bool:
  """Compares if the benchmark results are the similar.

  Args:
    expected_benchmark_file_path: path to the expected benchmark file.
    actual_benchmark_file_path: path to the actual benchmark file.
    allowed_threshold: allowed percent range of the benchmark results.
    max_seq_length: max sequence length to get the benchmark results.

  Returns:
    True if the benchmark results are the similar, False otherwise.
  """
  benchmark_filename = os.path.basename(expected_benchmark_file_path)
  expected_results = get_benchmark_results(
      expected_benchmark_file_path, max_seq_length
  )
  expected_peak_mem, expected_throughput = (
      expected_results.peak_mem,
      expected_results.throughput,
  )
  actual_results = get_benchmark_results(
      actual_benchmark_file_path, max_seq_length
  )
  actual_peak_mem, actual_throughput = (
      actual_results.peak_mem,
      actual_results.throughput,
  )
  formatted_max_seq_length = _get_formatted_string(max_seq_length)

  # Case 1: both peak mem and throughput are None(ideally due to OOM)
  if expected_peak_mem is None and actual_peak_mem is None:
    logging.info(
        'Both peak mem and throughput are None for max_seq_length %d.',
        max_seq_length,
    )
    return True

  check_oom_exception = _THROUGHPUT_TEST_EXCEPTIONS.get(
      (benchmark_filename, formatted_max_seq_length), 0.0
  ) == float('inf')
  # Case 2: When something strated to fail recently, or something which failed
  # before but is working now.
  if expected_peak_mem is None and actual_peak_mem is not None:
    if check_oom_exception:
      return True
    logging.error(
        'One of the failing benchmarks in %s is passing now for max_seq_length'
        ' %d. The expected peak mem and throughput are None, but the actual'
        ' peak mem is %f and actual throughput is %f',
        benchmark_filename,
        max_seq_length,
        actual_peak_mem,
        actual_throughput,
    )
    return False
  if actual_peak_mem is None and expected_peak_mem is not None:
    if check_oom_exception:
      return True
    logging.error(
        'One of the passing benchmarks in %s is failing now for max_seq_length'
        ' %d. The actual peak mem and throughput are None, but the expected'
        ' peak mem is %f and expected throughput is %f',
        benchmark_filename,
        max_seq_length,
        expected_peak_mem,
        expected_throughput,
    )
    return False
  # Case 3: When both actual peak mem and throughput lies within the range
  # of their respective expected values.
  mem_percent_change = _calculate_percent_change(
      actual_peak_mem, expected_peak_mem
  )
  throughput_percent_change = _calculate_percent_change(
      actual_throughput, expected_throughput
  )
  allowed_threshold = _THROUGHPUT_TEST_EXCEPTIONS.get(
      (benchmark_filename, formatted_max_seq_length), allowed_threshold
  )

  if abs(mem_percent_change) > allowed_threshold:
    logging.error(
        'The peak memory is changing by more than %f%% for max_seq_length %d.'
        ' Expected: %f, Actual: %f',
        allowed_threshold,
        max_seq_length,
        expected_peak_mem,
        actual_peak_mem,
    )
    return False
  if abs(throughput_percent_change) > allowed_threshold:
    logging.error(
        'The throughput is changing by more than %f%% for max_seq_length %d.'
        ' Expected throughput: %f, Actual throughput: %f',
        allowed_threshold,
        max_seq_length,
        expected_throughput,
        actual_throughput,
    )
    return False

  return True


def check_benchmark_results(
    actual_benchmark_file_path: str,
    model_family: str,
    allowed_threshold: float,
    max_seq_length: int,
) -> bool:
  """Checks the benchmark result between the actual and expected benchmark files.

  Args:
    actual_benchmark_file_path: path to the actual benchmark file.
    model_family: family of the model.
    allowed_threshold: allowed range of the benchmark results in percent.
    max_seq_length: max sequence length to get the benchmark results.

  Returns:
    True if the benchmark results are the similar, False otherwise.
  """
  benchmark_filename = os.path.basename(actual_benchmark_file_path)
  get_test_data_path(_GCS_TESTDATA_DIR)
  expected_benchmark_file_path = os.path.join(
      _LOCAL_INPUT_DIR.value,
      _GCS_TESTDATA_DIR,
      model_family,
      benchmark_filename,
  )
  print_benchmark_results(expected_benchmark_file_path, 'Expected')
  print_benchmark_results(actual_benchmark_file_path, 'Actual')

  return compare_benchmark_results(
      expected_benchmark_file_path,
      actual_benchmark_file_path,
      allowed_threshold,
      max_seq_length,
  )


def list_gcs_directories(bucket: str, directory: str) -> list[str]:
  """Lists GCS files."""
  output = subprocess.check_output([
      'gcloud',
      'storage',
      'ls',
      f'gs://{bucket}/{directory}',
  ])
  return output.decode('utf-8').splitlines()


def delete_gcs_object(gcs_directory: str):
  """Deletes GCS object."""
  subprocess.check_output([
      'gcloud',
      'storage',
      'rm',
      '-r',
      f'{gcs_directory}',
  ])
