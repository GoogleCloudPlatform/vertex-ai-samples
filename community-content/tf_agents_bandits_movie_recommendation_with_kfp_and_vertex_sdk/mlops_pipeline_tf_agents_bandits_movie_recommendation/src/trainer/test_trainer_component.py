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

"""The unit testing module for the Trainer component."""
import unittest
from unittest import mock

from src.trainer import trainer_component
import tensorflow as tf


# Paths and configurations
TRAINING_ARTIFACTS_DIR = "gs://bucket-name/artifacts"
TFRECORD_FILE = "gs://bucket-name/dataset.tfrecord"

# Hyperparameters
BATCH_SIZE = 8
NUM_EPOCHS = 5

# MovieLens simulation environment parameters
RANK_K = 20
NUM_ACTIONS = 20

# Agent parameters
TIKHONOV_WEIGHT = 0.001
AGENT_ALPHA = 10.0

NUM_RECORDS = 3
RECORD = {
    "step_type":
        tf.io.serialize_tensor(tf.constant([0] * BATCH_SIZE, dtype=tf.int32)),
    "observation":
        tf.io.serialize_tensor(tf.constant([0] * BATCH_SIZE, dtype=tf.float32)),
    "action":
        tf.io.serialize_tensor(tf.constant([0] * BATCH_SIZE, dtype=tf.int32)),
    "policy_info": (),
    "next_step_type":
        tf.io.serialize_tensor(tf.constant([0] * BATCH_SIZE, dtype=tf.int32)),
    "reward":
        tf.io.serialize_tensor(tf.constant([0] * BATCH_SIZE, dtype=tf.float32)),
    "discount":
        tf.io.serialize_tensor(tf.constant([0] * BATCH_SIZE, dtype=tf.float32)),
}
LOSS = tf.constant(-1.0)


class TestTrainerComponent(unittest.TestCase):
  """Test class for the Trainer component."""

  def setUp(self):
    super().setUp()

    self.mock_raw_dataset = mock.MagicMock()
    self.mock_raw_dataset.map.return_value = [
        RECORD for _ in range(NUM_RECORDS)
    ]

    self.mock_tfrecord_dataset = mock.patch(
        "tensorflow.data.TFRecordDataset").start()
    self.mock_tfrecord_dataset.return_value = self.mock_raw_dataset

    self.mock_agent = mock.MagicMock()
    self.mock_agent.train.return_value = (LOSS, None)

    self.mock_linucb_agent = mock.patch(
        "tf_agents.bandits.agents.lin_ucb_agent.LinearUCBAgent").start()
    self.mock_linucb_agent.return_value = self.mock_agent

    self.mock_saver = mock.MagicMock()
    self.mock_policy_saver = mock.patch(
        "tf_agents.policies.policy_saver.PolicySaver").start()
    self.mock_policy_saver.return_value = self.mock_saver

  def tearDown(self):
    super().tearDown()
    mock.patch.stopall()

  def test_training_op_execute_all_steps(self):
    """Tests that training_op executes all steps."""
    trainer_component.train_reinforcement_learning_policy(
        training_artifacts_dir=TRAINING_ARTIFACTS_DIR,
        tfrecord_file=TFRECORD_FILE,
        num_epochs=NUM_EPOCHS,
        rank_k=RANK_K,
        num_actions=NUM_ACTIONS,
        tikhonov_weight=TIKHONOV_WEIGHT,
        agent_alpha=AGENT_ALPHA)

    # Assert that a LinearUCBAgent is created.
    agent_args = self.mock_linucb_agent.call_args[1]
    self.assertEqual(agent_args["tikhonov_weight"], TIKHONOV_WEIGHT)
    self.assertEqual(agent_args["alpha"], AGENT_ALPHA)

    # Assert that a raw dataset is constructed.
    self.mock_tfrecord_dataset.assert_called_once_with([TFRECORD_FILE])

    # Assert that the row dataset is mapped and parsed.
    self.mock_raw_dataset.map.assert_called_once()

    # Assert that the agent trains for the correct number of iterations.
    self.assertEqual(self.mock_agent.train.call_count, NUM_EPOCHS * NUM_RECORDS)

    # Assert that the Policy Saver is created and the policy is saved.
    self.mock_policy_saver.assert_called_once()
    self.mock_saver.save.assert_called_once_with(TRAINING_ARTIFACTS_DIR)

  def test_given_zero_epochs_training_op_execute_no_training(self):
    """Tests that training_op executes zero training with zero num_epochs."""
    trainer_component.train_reinforcement_learning_policy(
        training_artifacts_dir=TRAINING_ARTIFACTS_DIR,
        tfrecord_file=TFRECORD_FILE,
        num_epochs=0,
        rank_k=RANK_K,
        num_actions=NUM_ACTIONS,
        tikhonov_weight=TIKHONOV_WEIGHT,
        agent_alpha=AGENT_ALPHA)

    self.assertEqual(self.mock_agent.train.call_count, 0)

  def test_given_negative_epochs_training_op_execute_no_training(self):
    """Tests that training_op executes zero training with negative num_epochs.
    """
    trainer_component.train_reinforcement_learning_policy(
        training_artifacts_dir=TRAINING_ARTIFACTS_DIR,
        tfrecord_file=TFRECORD_FILE,
        num_epochs=-1,
        rank_k=RANK_K,
        num_actions=NUM_ACTIONS,
        tikhonov_weight=TIKHONOV_WEIGHT,
        agent_alpha=AGENT_ALPHA)

    self.assertEqual(self.mock_agent.train.call_count, 0)

  def test_given_float_epochs_training_op_raise_exception(self):
    """Tests that training_op raises an exception for float num_epochs."""
    with self.assertRaises(TypeError):
      trainer_component.train_reinforcement_learning_policy(
          training_artifacts_dir=TRAINING_ARTIFACTS_DIR,
          tfrecord_file=TFRECORD_FILE,
          num_epochs=0.5,
          rank_k=RANK_K,
          num_actions=NUM_ACTIONS,
          tikhonov_weight=TIKHONOV_WEIGHT,
          agent_alpha=AGENT_ALPHA)


if __name__ == "__main__":
  unittest.main()
