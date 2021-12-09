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
import os

from fastapi import FastAPI
from fastapi import Request

import tensorflow as tf
import tf_agents


app = FastAPI()
_model = tf.compat.v2.saved_model.load(os.environ["AIP_STORAGE_URI"])


@app.get(os.environ["AIP_HEALTH_ROUTE"], status_code=200)
def health():
  """Handles server health check requests.

  Returns:
    An empty dict.
  """
  return {}


@app.post(os.environ["AIP_PREDICT_ROUTE"])
async def predict(request: Request):
  """Handles prediction requests.

  Unpacks observations in prediction requests and queries the trained policy for
  predicted actions.

  Args:
    request: Incoming prediction requests that contain observations.

  Returns:
    A dict with the key `predictions` mapping to a list of predicted actions
    corresponding to each observation in the prediction request.
  """
  body = await request.json()
  instances = body["instances"]

  predictions = []
  for index, instance in enumerate(instances):
    # Unpack request body and reconstruct TimeStep. Rewards default to 0.
    batch_size = len(instance["observation"])
    time_step = tf_agents.trajectories.restart(
        observation=instance["observation"],
        batch_size=tf.convert_to_tensor([batch_size]))
    policy_step = _model.action(time_step)

    predictions.append(
        {f"PolicyStep {index}": policy_step.action.numpy().tolist()})

  return {"predictions": predictions}
