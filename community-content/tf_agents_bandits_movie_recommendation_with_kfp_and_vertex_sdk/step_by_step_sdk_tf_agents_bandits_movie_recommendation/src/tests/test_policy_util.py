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

"""The unit testing module for policy_util."""
import functools
import unittest

from src.training import policy_util
import tensorflow as tf
from tf_agents.bandits.agents import lin_ucb_agent
from tf_agents.bandits.environments import environment_utilities
from tf_agents.bandits.environments import movielens_py_environment
from tf_agents.bandits.metrics import tf_metrics as tf_bandit_metrics
from tf_agents.environments import tf_py_environment


# Paths and configurations
DATA_PATH = "gs://[your-bucket-name]/[your-dataset-dir]/u.data"  # FILL IN
ROOT_DIR = "gs://[your-bucket-name]/artifacts"  # FILL IN
ARTIFACTS_DIR = "gs://[your-bucket-name]/artifacts"  # FILL IN

# Hyperparameters
BATCH_SIZE = 8

# MovieLens simulation environment parameters
RANK_K = 20
NUM_ACTIONS = 20
PER_ARM = False

# Agent parameters
TIKHONOV_WEIGHT = 0.001
AGENT_ALPHA = 10.0

# Metric names
DEFAULT_METRIC_NAMES = frozenset({
    "NumberOfEpisodes", "AverageReturnMetric", "AverageEpisodeLengthMetric"})


class TestPolicyUtil(unittest.TestCase):
  """Test class for the policy_util module."""

  def setUp(self):  # pylint: disable=g-missing-super-call
    # Define RL environment.
    env = movielens_py_environment.MovieLensPyEnvironment(
        DATA_PATH, RANK_K, BATCH_SIZE,
        num_movies=NUM_ACTIONS, csv_delimiter="\t")
    self.environment = tf_py_environment.TFPyEnvironment(env)

    # Define RL agent/algorithm.
    self.agent = lin_ucb_agent.LinearUCBAgent(
        time_step_spec=self.environment.time_step_spec(),
        action_spec=self.environment.action_spec(),
        tikhonov_weight=TIKHONOV_WEIGHT,
        alpha=AGENT_ALPHA,
        dtype=tf.float32,
        accepts_per_arm_features=PER_ARM)

    # Define RL metric.
    optimal_reward_fn = functools.partial(
        environment_utilities.compute_optimal_reward_with_movielens_environment,
        environment=self.environment)
    self.regret_metric = tf_bandit_metrics.RegretMetric(optimal_reward_fn)

  def test_run_training_set_invalid_root_dir(self):
    """Invalid root directory for saving training artifacts."""
    with self.assertRaises(tf.errors.FailedPreconditionError):
      policy_util.train(
          agent=self.agent,
          environment=self.environment,
          training_loops=2,
          steps_per_loop=2,
          additional_metrics=[],
          run_hyperparameter_tuning=False,
          root_dir="\0",
          artifacts_dir=ARTIFACTS_DIR)

  def test_run_training_set_invalid_artifacts_dir(self):
    """Invalid artifacts directory for saving training artifacts."""
    with self.assertRaises(tf.errors.FailedPreconditionError):
      policy_util.train(
          agent=self.agent,
          environment=self.environment,
          training_loops=2,
          steps_per_loop=2,
          additional_metrics=[],
          run_hyperparameter_tuning=False,
          root_dir=ROOT_DIR,
          artifacts_dir="\0")

  def test_run_training_set_zero_training_loops(self):
    """Training with zero training loops."""
    metric_results = policy_util.train(
        agent=self.agent,
        environment=self.environment,
        training_loops=0,
        steps_per_loop=2,
        additional_metrics=[],
        run_hyperparameter_tuning=False,
        root_dir=ROOT_DIR,
        artifacts_dir=ARTIFACTS_DIR)

    self.assertIsInstance(metric_results, dict)
    self.assertFalse(metric_results)

  def test_run_training_set_negative_training_loops(self):
    """Training with a negative number of training loops."""
    metric_results = policy_util.train(
        agent=self.agent,
        environment=self.environment,
        training_loops=-1,
        steps_per_loop=2,
        additional_metrics=[],
        run_hyperparameter_tuning=False,
        root_dir=ROOT_DIR,
        artifacts_dir=ARTIFACTS_DIR)

    self.assertIsInstance(metric_results, dict)
    self.assertFalse(metric_results)

  def test_run_training_set_float_training_loops(self):
    """Training with a floating-point number of training loops."""
    with self.assertRaises(TypeError):
      policy_util.train(
          agent=self.agent,
          environment=self.environment,
          training_loops=0.5,
          steps_per_loop=2,
          additional_metrics=[],
          run_hyperparameter_tuning=False,
          root_dir=ROOT_DIR,
          artifacts_dir=ARTIFACTS_DIR)

  def test_run_training_set_zero_steps_per_loop(self):
    """Training with a zero number of steps per training loop."""
    with self.assertRaises(tf.errors.InvalidArgumentError):
      policy_util.train(
          agent=self.agent,
          environment=self.environment,
          training_loops=2,
          steps_per_loop=0,
          additional_metrics=[],
          run_hyperparameter_tuning=False,
          root_dir=ROOT_DIR,
          artifacts_dir=ARTIFACTS_DIR)

  def test_run_training_set_negative_steps_per_loop(self):
    """Training with a negative number of steps per training loop."""
    with self.assertRaises(tf.errors.InvalidArgumentError):
      policy_util.train(
          agent=self.agent,
          environment=self.environment,
          training_loops=2,
          steps_per_loop=-1,
          additional_metrics=[],
          run_hyperparameter_tuning=False,
          root_dir=ROOT_DIR,
          artifacts_dir=ARTIFACTS_DIR)

  def test_run_training_set_float_steps_per_loop(self):
    """Training with a floating-point number of steps per training loop."""
    with self.assertRaises(TypeError):
      policy_util.train(
          agent=self.agent,
          environment=self.environment,
          training_loops=2,
          steps_per_loop=0.5,
          additional_metrics=[],
          run_hyperparameter_tuning=False,
          root_dir=ROOT_DIR,
          artifacts_dir=ARTIFACTS_DIR)

  def test_run_training_set_no_additional_metrics(self):
    """Training with default metrics."""
    training_loops = 1
    metric_results = policy_util.train(
        agent=self.agent,
        environment=self.environment,
        training_loops=training_loops,
        steps_per_loop=2,
        additional_metrics=[],
        run_hyperparameter_tuning=False,
        root_dir=ROOT_DIR,
        artifacts_dir=ARTIFACTS_DIR)

    self.assertIsInstance(metric_results, dict)
    self.assertEqual(metric_results.keys(), DEFAULT_METRIC_NAMES)
    for metric_name in DEFAULT_METRIC_NAMES:
      # There are `training_loops` number of intermediate metric values.
      self.assertEqual(len(metric_results[metric_name]), training_loops)

  def test_run_training_set_additional_metrics(self):
    """Training with an additional metric."""
    training_loops = 1
    metric_results = policy_util.train(
        agent=self.agent,
        environment=self.environment,
        training_loops=training_loops,
        steps_per_loop=2,
        additional_metrics=[self.regret_metric],
        run_hyperparameter_tuning=False,
        root_dir=ROOT_DIR,
        artifacts_dir=ARTIFACTS_DIR)

    self.assertIsInstance(metric_results, dict)
    total_metric_names = DEFAULT_METRIC_NAMES.union(
        {type(self.regret_metric).__name__})
    self.assertEqual(metric_results.keys(), total_metric_names)
    for metric_name in total_metric_names:
      # There are `training_loops` number of intermediate metric values.
      self.assertEqual(len(metric_results[metric_name]), training_loops)

  def test_run_hyperparameter_tuning_set_root_dir(self):
    """Setting root directory for hyperparameter tuning.

    Hyperparameter tuning doesn't save artifacts to `root_dir`.
    """
    with self.assertRaises(ValueError):
      policy_util.train(
          agent=self.agent,
          environment=self.environment,
          training_loops=2,
          steps_per_loop=2,
          additional_metrics=[],
          run_hyperparameter_tuning=True,
          root_dir="./")

  def test_run_hyperparameter_tuning_set_artifacts_dir(self):
    """Setting artifacts directory for hyperparameter tuning.

    Hyperparameter tuning doesn't save artifacts to `artifacts_dir`.
    """
    with self.assertRaises(ValueError):
      policy_util.train(
          agent=self.agent,
          environment=self.environment,
          training_loops=2,
          steps_per_loop=2,
          additional_metrics=[],
          run_hyperparameter_tuning=True,
          artifacts_dir="./")

  def test_run_hyperparameter_tuning_set_zero_training_loops(self):
    """Hyperparameter tuning with zero training loops."""
    metric_results = policy_util.train(
        agent=self.agent,
        environment=self.environment,
        training_loops=0,
        steps_per_loop=2,
        additional_metrics=[],
        run_hyperparameter_tuning=True)

    self.assertIsInstance(metric_results, dict)
    self.assertFalse(metric_results)

  def test_run_hyperparameter_tuning_set_negative_training_loops(self):
    """Hyperparameter tuning with a negative number of training loops."""
    metric_results = policy_util.train(
        agent=self.agent,
        environment=self.environment,
        training_loops=-1,
        steps_per_loop=2,
        additional_metrics=[],
        run_hyperparameter_tuning=True)

    self.assertIsInstance(metric_results, dict)
    self.assertFalse(metric_results)

  def test_run_hyperparameter_tuning_set_float_training_loops(self):
    """Hyperparameter tuning with a floating-point number of training loops."""
    with self.assertRaises(TypeError):
      policy_util.train(
          agent=self.agent,
          environment=self.environment,
          training_loops=0.5,
          steps_per_loop=2,
          additional_metrics=[],
          run_hyperparameter_tuning=True)

  def test_run_hyperparameter_tuning_set_zero_steps_per_loop(self):
    """Hyperparameter tuning with a zero number of steps per training loop."""
    with self.assertRaises(tf.errors.InvalidArgumentError):
      policy_util.train(
          agent=self.agent,
          environment=self.environment,
          training_loops=2,
          steps_per_loop=0,
          additional_metrics=[],
          run_hyperparameter_tuning=True)

  def test_run_hyperparameter_tuning_set_negative_steps_per_loop(self):
    """Hyperparameter tuning with a negative number of steps per training loop.
    """
    with self.assertRaises(tf.errors.InvalidArgumentError):
      policy_util.train(
          agent=self.agent,
          environment=self.environment,
          training_loops=2,
          steps_per_loop=-1,
          additional_metrics=[],
          run_hyperparameter_tuning=True)

  def test_run_hyperparameter_tuning_set_float_steps_per_loop(self):
    """Hyperparameter tuning with a float number of steps per training loop."""
    with self.assertRaises(TypeError):
      policy_util.train(
          agent=self.agent,
          environment=self.environment,
          training_loops=2,
          steps_per_loop=0.5,
          additional_metrics=[],
          run_hyperparameter_tuning=True)

  def test_run_hyperparameter_tuning_set_no_additional_metrics(self):
    """Hyperparameter tuning with default metrics."""
    training_loops = 1
    metric_results = policy_util.train(
        agent=self.agent,
        environment=self.environment,
        training_loops=training_loops,
        steps_per_loop=2,
        additional_metrics=[],
        run_hyperparameter_tuning=True)

    self.assertIsInstance(metric_results, dict)
    self.assertEqual(metric_results.keys(), DEFAULT_METRIC_NAMES)
    for metric_name in DEFAULT_METRIC_NAMES:
      # There are `training_loops` number of intermediate metric values.
      self.assertEqual(len(metric_results[metric_name]), training_loops)

  def test_run_hyperparameter_tuning_set_additional_metrics(self):
    """Hyperparameter tuning with an additional metric."""
    training_loops = 1
    metric_results = policy_util.train(
        agent=self.agent,
        environment=self.environment,
        training_loops=training_loops,
        steps_per_loop=2,
        additional_metrics=[self.regret_metric],
        run_hyperparameter_tuning=True)

    self.assertIsInstance(metric_results, dict)
    total_metric_names = DEFAULT_METRIC_NAMES.union(
        {type(self.regret_metric).__name__})
    self.assertEqual(metric_results.keys(), total_metric_names)
    for metric_name in total_metric_names:
      # There are `training_loops` number of intermediate metric values.
      self.assertEqual(len(metric_results[metric_name]), training_loops)


if __name__ == "__main__":
  unittest.main()
