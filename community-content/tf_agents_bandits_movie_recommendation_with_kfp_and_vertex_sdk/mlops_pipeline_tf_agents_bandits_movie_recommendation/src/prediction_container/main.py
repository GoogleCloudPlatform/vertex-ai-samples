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

"""Prediction server that uses a trained policy to give predicted actions."""
import json
import os
from typing import Dict, List

import fastapi

from google.cloud import pubsub_v1

import tensorflow as tf
import tf_agents
from tf_agents import policies


app = fastapi.FastAPI()
app_vars = {"trained_policy": None}


def _startup_event() -> None:
  """Loads the trained policy at startup."""
  app_vars["trained_policy"] = tf.saved_model.load(
      os.environ["AIP_STORAGE_URI"])


@app.on_event("startup")
async def startup_event() -> None:
  """Loads the trained policy at startup."""
  _startup_event()


def _health() -> Dict[str, str]:
  """Handles server health check requests.

  Returns:
    An empty dict.
  """
  return {}


@app.get(os.environ["AIP_HEALTH_ROUTE"], status_code=200)
def health() -> Dict[str, str]:
  """Handles server health check requests.

  Returns:
    An empty dict.
  """
  return _health()


def _message_logger_via_pubsub(
    project_id: str,
    logger_pubsub_topic: str,
    observations: List[Dict[str, List[List[float]]]],
    predicted_actions: List[Dict[str, List[float]]]) -> None:
  """Send a message to the Pub/Sub topic which triggers the Logger.

  Package observations and the corresponding predicted actions in a message JSON
  and send to Pub/Sub topic.

  Args:
    project_id: GCP project ID.
    logger_pubsub_topic: Name of Pub/Sub topic that triggers the Logger.
    observations: List of `{"observation": <observation>}` in the prediction
      request.
    predicted_actions: List of `{"predicted_action": <predicted_action>}`
      corresponding to the observations.
  """
  # Create message with observations and predicted actions.
  message_json = json.dumps({
      "observations": observations,
      "predicted_actions": predicted_actions,
  })
  message_bytes = message_json.encode("utf-8")

  # Instantiate a Pub/Sub client.
  publisher = pubsub_v1.PublisherClient()

  # Get the Logger's Pub/Sub topic.
  topic_path = publisher.topic_path(project_id, logger_pubsub_topic)

  # Send message.
  publish_future = publisher.publish(topic_path, data=message_bytes)
  publish_future.result()


def _predict(
    instances: List[Dict[str, List[List[float]]]],
    trained_policy: policies.TFPolicy) -> Dict[str, List[Dict[str, List[int]]]]:
  """Gets predictions for the observations in `instances`; triggers the Logger.

  Unpacks observations in `instances` and queries the trained policy for
  predicted actions. Triggers the Logger with observations and predicted
  actions.

  Args:
    instances: List of `{"observation": <observation>}` for which to generate
      predictions.
    trained_policy: Trained policy to generate predictions.

  Returns:
    A dict with the key "predictions" mapping to a list of predicted actions
    corresponding to each observation in the prediction request.
  """
  predictions = []
  predicted_actions = []
  for index, instance in enumerate(instances):
    # Unpack observation and reconstruct TimeStep. Rewards default to 0.
    batch_size = len(instance["observation"])
    time_step = tf_agents.trajectories.restart(
        observation=instance["observation"],
        batch_size=tf.convert_to_tensor([batch_size]))
    policy_step = trained_policy.action(time_step)

    predicted_action = policy_step.action.numpy().tolist()
    predictions.append(
        {f"PolicyStep {index}": predicted_action})
    predicted_actions.append({"predicted_action": predicted_action})

  # Trigger the Logger to log prediction inputs and results.
  _message_logger_via_pubsub(
      project_id=os.environ["PROJECT_ID"],
      logger_pubsub_topic=os.environ["LOGGER_PUBSUB_TOPIC"],
      observations=instances,
      predicted_actions=predicted_actions)
  return {"predictions": predictions}


@app.post(os.environ["AIP_PREDICT_ROUTE"])
async def predict(
    request: fastapi.Request) -> Dict[str, List[Dict[str, List[int]]]]:
  """Handles prediction requests.

  Unpacks observations in prediction requests and queries the trained policy for
  predicted actions.

  Args:
    request: Incoming prediction requests that contain observations.

  Returns:
    A dict with the key "predictions" mapping to a list of predicted actions
    corresponding to each observation in the prediction request.
  """
  body = await request.json()
  instances = body["instances"]
  return _predict(instances, app_vars["trained_policy"])
