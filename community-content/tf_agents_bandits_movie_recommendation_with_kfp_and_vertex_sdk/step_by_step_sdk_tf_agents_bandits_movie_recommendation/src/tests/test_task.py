# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The unit testing module for task."""
import argparse
import os
import unittest
from unittest import mock

from src.training import task


# Paths and configurations
DATA_PATH = "gs://[your-bucket-name]/artifacts/u.data"  # FILL IN
ROOT_DIR = "gs://[your-bucket-name]/artifacts"  # FILL IN
ARTIFACTS_DIR = "gs://[your-bucket-name]/artifacts"  # FILL IN
PROFILER_DIR = "gs://[your-bucket-name]/profiler"  # FILL IN
HPTUNING_RESULT_DIR = "[your-hptuning-result-dir]/"  # FILL IN
HPTUNING_RESULT_PATH = os.path.join(HPTUNING_RESULT_DIR,
                                    "result.json")  # FILL IN
RAW_BUCKET_NAME = "[your-hptuning-result-bucket-name]"  # FILL IN

# Hyperparameters
BATCH_SIZE = 8
TRAINING_LOOPS = 4
STEPS_PER_LOOP = 2

# MovieLens simulation environment parameters
RANK_K = 20
NUM_ACTIONS = 20
PER_ARM = False

# Agent parameters
TIKHONOV_WEIGHT = 0.001
AGENT_ALPHA = 10.0


class TestTask(unittest.TestCase):
  """Test class for the task module."""

  def setUp(self):  # pylint: disable=g-missing-super-call
    os.environ["AIP_MODEL_DIR"] = ROOT_DIR

  def test_get_args_for_hyperparameter_tuning(self):
    """Complete set of arguments for hyperparameter tuning."""
    raw_args = [
        "--run-hyperparameter-tuning",
        f"--data-path={DATA_PATH}",
        f"--batch-size={BATCH_SIZE}",
        f"--training-loops={TRAINING_LOOPS}",
        f"--steps-per-loop={STEPS_PER_LOOP}",
        f"--rank-k={RANK_K}",
        f"--num-actions={NUM_ACTIONS}",
        f"--tikhonov-weight={TIKHONOV_WEIGHT}",
        f"--agent-alpha={AGENT_ALPHA}",
    ]
    expected_args = {
        "run_hyperparameter_tuning": True,
        "train_with_best_hyperparameters": False,
        "artifacts_dir": None,
        "profiler_dir": None,
        "data_path": DATA_PATH,
        "best_hyperparameters_bucket": None,
        "best_hyperparameters_path": None,
        "batch_size": BATCH_SIZE,
        "training_loops": TRAINING_LOOPS,
        "steps_per_loop": STEPS_PER_LOOP,
        "rank_k": RANK_K,
        "num_actions": NUM_ACTIONS,
        "tikhonov_weight": TIKHONOV_WEIGHT,
        "agent_alpha": AGENT_ALPHA,
    }

    parsed_args = vars(task.get_args(raw_args))

    self.assertDictEqual(parsed_args, expected_args)

  def test_get_args_for_training_with_best_hyperparameters(self):
    """Complete set of arguments for training with best hyperparameters."""
    raw_args = [
        "--train-with-best-hyperparameters",
        f"--artifacts-dir={ARTIFACTS_DIR}",
        f"--profiler-dir={PROFILER_DIR}",
        f"--data-path={DATA_PATH}",
        f"--best-hyperparameters-bucket={RAW_BUCKET_NAME}",
        f"--best-hyperparameters-path={HPTUNING_RESULT_PATH}",
        f"--batch-size={BATCH_SIZE}",
        f"--training-loops={TRAINING_LOOPS}",
        f"--steps-per-loop={STEPS_PER_LOOP}",
        f"--rank-k={RANK_K}",
        f"--num-actions={NUM_ACTIONS}",
        f"--tikhonov-weight={TIKHONOV_WEIGHT}",
        f"--agent-alpha={AGENT_ALPHA}",
    ]
    expected_args = {
        "run_hyperparameter_tuning": False,
        "train_with_best_hyperparameters": True,
        "artifacts_dir": ARTIFACTS_DIR,
        "profiler_dir": PROFILER_DIR,
        "data_path": DATA_PATH,
        "best_hyperparameters_bucket": RAW_BUCKET_NAME,
        "best_hyperparameters_path": HPTUNING_RESULT_PATH,
        "batch_size": BATCH_SIZE,
        "training_loops": TRAINING_LOOPS,
        "steps_per_loop": STEPS_PER_LOOP,
        "rank_k": RANK_K,
        "num_actions": NUM_ACTIONS,
        "tikhonov_weight": TIKHONOV_WEIGHT,
        "agent_alpha": AGENT_ALPHA,
    }

    parsed_args = vars(task.get_args(raw_args))

    self.assertDictEqual(parsed_args, expected_args)

  def test_get_args_for_training_without_best_hyperparameters(self):
    """Complete set of arguments for training without best hyperparameters."""
    raw_args = [
        f"--artifacts-dir={ARTIFACTS_DIR}",
        f"--profiler-dir={PROFILER_DIR}",
        f"--data-path={DATA_PATH}",
        f"--batch-size={BATCH_SIZE}",
        f"--training-loops={TRAINING_LOOPS}",
        f"--steps-per-loop={STEPS_PER_LOOP}",
        f"--rank-k={RANK_K}",
        f"--num-actions={NUM_ACTIONS}",
        f"--tikhonov-weight={TIKHONOV_WEIGHT}",
        f"--agent-alpha={AGENT_ALPHA}",
    ]
    expected_args = {
        "run_hyperparameter_tuning": False,
        "train_with_best_hyperparameters": False,
        "artifacts_dir": ARTIFACTS_DIR,
        "profiler_dir": PROFILER_DIR,
        "data_path": DATA_PATH,
        "best_hyperparameters_bucket": None,
        "best_hyperparameters_path": None,
        "batch_size": BATCH_SIZE,
        "training_loops": TRAINING_LOOPS,
        "steps_per_loop": STEPS_PER_LOOP,
        "rank_k": RANK_K,
        "num_actions": NUM_ACTIONS,
        "tikhonov_weight": TIKHONOV_WEIGHT,
        "agent_alpha": AGENT_ALPHA,
    }

    parsed_args = vars(task.get_args(raw_args))

    self.assertDictEqual(parsed_args, expected_args)

  def test_hyperparameter_tuning(self):
    """Hyperparameter tuning using mocks and patch."""
    args_dict = {
        "run_hyperparameter_tuning": True,
        "train_with_best_hyperparameters": False,
        "artifacts_dir": None,
        "profiler_dir": None,
        "data_path": DATA_PATH,
        "best_hyperparameters_bucket": None,
        "best_hyperparameters_path": None,
        "batch_size": BATCH_SIZE,
        "training_loops": TRAINING_LOOPS,
        "steps_per_loop": STEPS_PER_LOOP,
        "rank_k": RANK_K,
        "num_actions": NUM_ACTIONS,
        "tikhonov_weight": TIKHONOV_WEIGHT,
        "agent_alpha": AGENT_ALPHA,
    }
    args = argparse.Namespace()
    args.__dict__.update(args_dict)
    best_hyperparameters_blob = mock.Mock()
    hypertune_client = mock.Mock()

    with mock.patch("src.training.policy_util.train") as train:
      task.execute_task(args, best_hyperparameters_blob, hypertune_client)

      train.assert_called_once()
      best_hyperparameters_blob.download_as_string.assert_not_called()
      hypertune_client.report_hyperparameter_tuning_metric.assert_called_once()

  def test_train_with_best_hyperparameters(self):
    """Training with best hyperparameters using mocks and patch."""
    args_dict = {
        "run_hyperparameter_tuning": False,
        "train_with_best_hyperparameters": True,
        "artifacts_dir": ARTIFACTS_DIR,
        "profiler_dir": PROFILER_DIR,
        "data_path": DATA_PATH,
        "best_hyperparameters_bucket": RAW_BUCKET_NAME,
        "best_hyperparameters_path": HPTUNING_RESULT_PATH,
        "batch_size": BATCH_SIZE,
        "training_loops": TRAINING_LOOPS,
        "steps_per_loop": STEPS_PER_LOOP,
        "rank_k": RANK_K,
        "num_actions": NUM_ACTIONS,
        "tikhonov_weight": TIKHONOV_WEIGHT,
        "agent_alpha": AGENT_ALPHA,
    }
    args = argparse.Namespace()
    args.__dict__.update(args_dict)
    best_hyperparameters_blob = mock.Mock()
    best_hyperparameters_blob.download_as_string.return_value = """
        {"steps-per-loop": 2.0, "training-loops": 4.0}"""
    hypertune_client = mock.Mock()

    with mock.patch("src.training.policy_util.train") as train:
      task.execute_task(args, best_hyperparameters_blob, hypertune_client)

      train.assert_called_once()

      best_hyperparameters_blob.download_as_string.assert_called_once()
      hypertune_client.report_hyperparameter_tuning_metric.assert_not_called()

  def test_train_without_best_hyperparameters(self):
    """Training without best hyperparameters using mocks and patch."""
    args_dict = {
        "run_hyperparameter_tuning": False,
        "train_with_best_hyperparameters": False,
        "artifacts_dir": ARTIFACTS_DIR,
        "profiler_dir": PROFILER_DIR,
        "data_path": DATA_PATH,
        "best_hyperparameters_bucket": None,
        "best_hyperparameters_path": None,
        "batch_size": BATCH_SIZE,
        "training_loops": TRAINING_LOOPS,
        "steps_per_loop": STEPS_PER_LOOP,
        "rank_k": RANK_K,
        "num_actions": NUM_ACTIONS,
        "tikhonov_weight": TIKHONOV_WEIGHT,
        "agent_alpha": AGENT_ALPHA,
    }
    args = argparse.Namespace()
    args.__dict__.update(args_dict)
    best_hyperparameters_blob = mock.Mock()
    hypertune_client = mock.Mock()

    with mock.patch("src.training.policy_util.train") as train:
      task.execute_task(args, best_hyperparameters_blob, hypertune_client)

      train.assert_called_once()

      best_hyperparameters_blob.download_as_string.assert_not_called()
      hypertune_client.report_hyperparameter_tuning_metric.assert_not_called()

  def test_hyperparameter_tuning_with_unpatched_policy_util(self):
    """Hyperparameter tuning using mocks (policy_util integration)."""
    args_dict = {
        "run_hyperparameter_tuning": True,
        "train_with_best_hyperparameters": False,
        "artifacts_dir": None,
        "profiler_dir": None,
        "data_path": DATA_PATH,
        "best_hyperparameters_bucket": None,
        "best_hyperparameters_path": None,
        "batch_size": BATCH_SIZE,
        "training_loops": TRAINING_LOOPS,
        "steps_per_loop": STEPS_PER_LOOP,
        "rank_k": RANK_K,
        "num_actions": NUM_ACTIONS,
        "tikhonov_weight": TIKHONOV_WEIGHT,
        "agent_alpha": AGENT_ALPHA,
    }
    args = argparse.Namespace()
    args.__dict__.update(args_dict)
    best_hyperparameters_blob = mock.Mock()
    hypertune_client = mock.Mock()

    task.execute_task(args, best_hyperparameters_blob, hypertune_client)

    best_hyperparameters_blob.download_as_string.assert_not_called()
    hypertune_client.report_hyperparameter_tuning_metric.assert_called_once()

  def test_train_with_best_hyperparameters_with_unpatched_policy_util(self):
    """Training with best hyperparameters using mocks (policy_util integration).
    """
    args_dict = {
        "run_hyperparameter_tuning": False,
        "train_with_best_hyperparameters": True,
        "artifacts_dir": ARTIFACTS_DIR,
        "profiler_dir": PROFILER_DIR,
        "data_path": DATA_PATH,
        "best_hyperparameters_bucket": RAW_BUCKET_NAME,
        "best_hyperparameters_path": HPTUNING_RESULT_PATH,
        "batch_size": BATCH_SIZE,
        "training_loops": TRAINING_LOOPS,
        "steps_per_loop": STEPS_PER_LOOP,
        "rank_k": RANK_K,
        "num_actions": NUM_ACTIONS,
        "tikhonov_weight": TIKHONOV_WEIGHT,
        "agent_alpha": AGENT_ALPHA,
    }
    args = argparse.Namespace()
    args.__dict__.update(args_dict)
    best_hyperparameters_blob = mock.Mock()
    best_hyperparameters_blob.download_as_string.return_value = """
        {"steps-per-loop": 2.0, "training-loops": 4.0}"""
    hypertune_client = mock.Mock()

    task.execute_task(args, best_hyperparameters_blob, hypertune_client)

    best_hyperparameters_blob.download_as_string.assert_called_once()
    hypertune_client.report_hyperparameter_tuning_metric.assert_not_called()

  def test_train_without_best_hyperparameters_with_unpatched_policy_util(self):
    """Training without best hp-params using mocks (policy_util integration)."""
    args_dict = {
        "run_hyperparameter_tuning": False,
        "train_with_best_hyperparameters": False,
        "artifacts_dir": ARTIFACTS_DIR,
        "profiler_dir": PROFILER_DIR,
        "data_path": DATA_PATH,
        "best_hyperparameters_bucket": None,
        "best_hyperparameters_path": None,
        "batch_size": BATCH_SIZE,
        "training_loops": TRAINING_LOOPS,
        "steps_per_loop": STEPS_PER_LOOP,
        "rank_k": RANK_K,
        "num_actions": NUM_ACTIONS,
        "tikhonov_weight": TIKHONOV_WEIGHT,
        "agent_alpha": AGENT_ALPHA,
    }
    args = argparse.Namespace()
    args.__dict__.update(args_dict)
    best_hyperparameters_blob = mock.Mock()
    hypertune_client = mock.Mock()

    task.execute_task(args, best_hyperparameters_blob, hypertune_client)

    best_hyperparameters_blob.download_as_string.assert_not_called()
    hypertune_client.report_hyperparameter_tuning_metric.assert_not_called()


if __name__ == "__main__":
  unittest.main()
