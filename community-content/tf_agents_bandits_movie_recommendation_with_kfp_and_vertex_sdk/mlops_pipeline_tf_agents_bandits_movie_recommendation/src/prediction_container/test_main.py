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

"""The unit testing module for the prediction container with FastAPI."""
import json
import os
import unittest
from unittest import mock

import numpy as np

os.environ["AIP_HEALTH_ROUTE"] = "/health"
os.environ["AIP_PREDICT_ROUTE"] = "/predict"

# Import main after the environment variables are set.
from src.prediction_container import main  # pylint: disable=g-import-not-at-top

# Path and configurations
TRAINING_ARTIFACTS_DIR = "gs://bucket-name/artifacts"
PROJECT_ID = "project-id"
LOGGER_PUBSUB_TOPIC = "logger-pubsub-topic"

# Hyperparameter
BATCH_SIZE = 8

# MovieLens simulation environment parameters
RANK_K = 20

OBSERVATION = np.zeros((BATCH_SIZE, RANK_K)).tolist()
PREDICTED_ACTION = np.zeros(BATCH_SIZE, dtype=np.int).tolist()
NUM_OBSERVATIONS = 5
REQUEST_INSTANCES = [
    {"observation": OBSERVATION} for _ in range(NUM_OBSERVATIONS)
]
PREDICTED_ACTIONS = [
    {"predicted_action": PREDICTED_ACTION} for _ in range(NUM_OBSERVATIONS)
]


class TestPredictionContainer(unittest.TestCase):
  """Test class for the prediction container."""

  def setUp(self):
    super().setUp()

    os.environ["AIP_STORAGE_URI"] = TRAINING_ARTIFACTS_DIR
    os.environ["PROJECT_ID"] = PROJECT_ID
    os.environ["LOGGER_PUBSUB_TOPIC"] = LOGGER_PUBSUB_TOPIC
    mock.patch("fastapi.FastAPI").start()

    self.mock_trained_policy = mock.MagicMock()
    self.mock_trained_policy.action.return_value = mock.MagicMock()

    self.mock_model_load = mock.patch("tensorflow.saved_model.load").start()

    self.mock_trajectories_restart = mock.patch(
        "tf_agents.trajectories.restart").start()

    self.mock_publisher = mock.MagicMock()
    self.mock_pubsub_client = mock.patch(
        "google.cloud.pubsub_v1.PublisherClient").start()
    self.mock_pubsub_client.return_value = self.mock_publisher

  def tearDown(self):
    super().tearDown()
    mock.patch.stopall()

  def test__startup_event_load_model_with_correct_path(self):
    """Tests _startup_event correctly loads a model."""
    main._startup_event()

    self.mock_model_load.assert_called_once_with(TRAINING_ARTIFACTS_DIR)

  def test__health_respond_empty_dict(self):
    """Tests _health always responds with an empty dictionary."""
    health_output = main._health()

    self.assertEqual(health_output, {})

  def test__predict_create_time_step_from_trajectories_for_each_observation(
      self):
    """Tests _predict creates a time step for each observation."""
    mock_message_logger = mock.patch(
        "src.prediction_container.main._message_logger_via_pubsub").start()

    main._predict(REQUEST_INSTANCES, self.mock_trained_policy)

    self.assertEqual(self.mock_trajectories_restart.call_count,
                     NUM_OBSERVATIONS)

    mock_message_logger.stop()

  def test__predict_query_model_for_each_observation(self):
    """Tests _predict queries the model for each observation."""
    mock_message_logger = mock.patch(
        "src.prediction_container.main._message_logger_via_pubsub").start()

    main._predict(REQUEST_INSTANCES, self.mock_trained_policy)

    self.assertEqual(self.mock_trained_policy.action.call_count,
                     NUM_OBSERVATIONS)

    mock_message_logger.stop()

  def test__predict_generate_one_prediction_for_each_observation(self):
    """Tests _predict queries the model for each observation."""
    mock_message_logger = mock.patch(
        "src.prediction_container.main._message_logger_via_pubsub").start()
    predictions = main._predict(REQUEST_INSTANCES, self.mock_trained_policy)

    self.assertEqual(len(predictions["predictions"]), NUM_OBSERVATIONS)

    mock_message_logger.stop()

  def test__predict_call__message_logger_via_pubsub(self):
    """Tests _predict indeed calls _message_logger_via_pubsub."""
    mock_message_logger = mock.patch(
        "src.prediction_container.main._message_logger_via_pubsub").start()

    main._predict(REQUEST_INSTANCES, self.mock_trained_policy)

    mock_message_logger.assert_called_once()

    mock_message_logger.stop()

  def test__message_logger_via_pubsub_convert_data_to_bytes_correctly(self):
    """Tests _message_logger_via_pubsub convert data to bytes correctly."""
    main._message_logger_via_pubsub(
        project_id=PROJECT_ID,
        logger_pubsub_topic=LOGGER_PUBSUB_TOPIC,
        observations=REQUEST_INSTANCES,
        predicted_actions=PREDICTED_ACTIONS)

    _, kwargs = self.mock_publisher.publish.call_args
    message_bytes = kwargs["data"]
    message_json = message_bytes.decode("utf-8")
    data = json.loads(message_json)
    self.assertEqual(data, {
        "observations": REQUEST_INSTANCES,
        "predicted_actions": PREDICTED_ACTIONS,
    })


if __name__ == "__main__":
  unittest.main()
