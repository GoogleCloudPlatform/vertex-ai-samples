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

"""The Logger component for logging prediction inputs and results."""
import base64
import dataclasses
import json
import os
import tempfile
from typing import Any, Dict, List

from google.cloud import bigquery
import tensorflow as tf
from tf_agents import trajectories
from tf_agents.bandits.environments import movielens_py_environment
from tf_agents.environments import tf_py_environment


@dataclasses.dataclass
class EnvVars:
  """A class containing environment variables and their values.

  Attributes:
    project_id: A string of the GCP project ID.
    raw_data_path: A string of the path to MovieLens 100K's "u.data" file.
    batch_size: A integer of the batch size of environment generated quantities.
    rank_k: An integer of the rank for matrix factorization in the MovieLens
      environment; also the observation dimension.
    num_actions: A integer of the number of actions (movie items) to choose
      from.
    bigquery_tmp_file: Path to a JSON file containing the training dataset.
    bigquery_dataset_id: A string of the BigQuery dataset ID as
      `project_id.dataset_id`.
    bigquery_location: A string of the BigQuery dataset region.
    bigquery_table_id: A string of the BigQuery table ID as
      `project_id.dataset_id.table_id`.
  """
  project_id: str
  raw_data_path: str
  batch_size: int
  rank_k: int
  num_actions: int
  bigquery_tmp_file: str
  bigquery_dataset_id: str
  bigquery_location: str
  bigquery_table_id: str


def get_env_vars() -> EnvVars:
  """Gets a set of environment variables necessary for `log`.

  Returns:
    A `EnvVars` of environment variables for configuring `log`.
  """
  return EnvVars(
      project_id=os.getenv("PROJECT_ID"),
      raw_data_path=os.getenv("RAW_DATA_PATH"),
      batch_size=int(os.getenv("BATCH_SIZE")),
      rank_k=int(os.getenv("RANK_K")),
      num_actions=int(os.getenv("NUM_ACTIONS")),
      bigquery_tmp_file=os.getenv("BIGQUERY_TMP_FILE"),
      bigquery_dataset_id=os.getenv("BIGQUERY_DATASET_ID"),
      bigquery_location=os.getenv("BIGQUERY_LOCATION"),
      bigquery_table_id=os.getenv("BIGQUERY_TABLE_ID"))


def replace_observation_in_time_step(
    original_time_step: trajectories.TimeStep,
    observation: tf.Tensor) -> trajectories.TimeStep:
  """Returns a `trajectories.TimeStep` with the observation field replaced.

  Args:
    original_time_step: The original `trajectories.TimeStep` in which the
      `observation` will be filled in.
    observation: A single, batched observation.

  Returns:
    A `trajectories.TimeStep` with `observation` filled into
    `original_time_step`.
  """
  return trajectories.TimeStep(
      step_type=original_time_step[0],
      reward=original_time_step[1],
      discount=original_time_step[2],
      observation=observation)


def get_trajectory_from_environment(
    environment: tf_py_environment.TFPyEnvironment,
    observation: List[List[float]],
    predicted_action: int) -> trajectories.Trajectory:
  """Gets trajectory data from `environment` based on observation and action.

  Aligns `environment` observation to `observation` so that its feedback align
  with `observation`. The `trajectories.Trajectory` object contains time step
  information before and after applying `predicted_action` and feedback in the
  form of a reward.

  In production, this function can be replaced to actually pull feedback from
  some real-world environment.

  Args:
    environment: A TF-Agents environment that holds observations, apply actions
      and returns rewards.
    observation: A single, batched observation.
    predicted_action: A predicted action corresponding to the observation.

  Returns:
    A dict holding the same data as `trajectory`.
  """
  environment.reset()

  # Align environment to observation.
  original_time_step = environment.current_time_step()
  time_step = replace_observation_in_time_step(original_time_step, observation)
  environment._time_step = time_step  # pylint: disable=protected-access

  # Apply predicted action to environment.
  environment.step(action=predicted_action)

  # Get next time step.
  next_time_step = environment.current_time_step()

  # Get trajectory as an encapsulation of all feedback from the environment.
  trajectory = trajectories.from_transition(
      time_step=time_step,
      action_step=trajectories.PolicyStep(action=predicted_action),
      next_time_step=next_time_step)
  return trajectory


def build_dict_from_trajectory(
    trajectory: trajectories.Trajectory) -> Dict[str, Any]:
  """Builds a dict from `trajectory` data.

  Args:
    trajectory: A `trajectories.Trajectory` object.

  Returns:
    A dict holding the same data as `trajectory`.
  """
  trajectory_dict = {
      "step_type": trajectory.step_type.numpy().tolist(),
      "observation": [{
          "observation_batch": batch
      } for batch in trajectory.observation.numpy().tolist()],
      "action": trajectory.action.numpy().tolist(),
      "policy_info": trajectory.policy_info,
      "next_step_type": trajectory.next_step_type.numpy().tolist(),
      "reward": trajectory.reward.numpy().tolist(),
      "discount": trajectory.discount.numpy().tolist(),
  }
  return trajectory_dict


def write_trajectories_to_file(
    dataset_file: str,
    environment: tf_py_environment.TFPyEnvironment,
    observations: List[Dict[str, List[List[float]]]],
    predicted_actions: List[Dict[str, List[float]]]) -> None:
  """Writes trajectory data to a file, each JSON in one line.

  Gets `trajectories.Trajectory` objects that encapsulate environment
  feedback eg. rewards based on `observations` and `predicted_actions`.
  Each `trajectories.Trajectory` object gets written as one line to
  `dataset_file` in JSON format. I.e., the `dataset_file` would be a
  newline-delimited JSON file.

  Args:
    dataset_file: Path to a JSON file containing the training dataset.
    environment: A TF-Agents environment that holds observations, apply actions
      and returns rewards.
    observations: List of `{"observation": <observation>}` in the prediction
      request.
    predicted_actions: List of `{"predicted_action": <predicted_action>}`
      corresponding to the observations.
  """
  with open(dataset_file, "w") as f:
    for observation, predicted_action in zip(observations, predicted_actions):
      trajectory = get_trajectory_from_environment(
          environment=environment,
          observation=tf.constant(observation["observation"]),
          predicted_action=tf.constant(predicted_action["predicted_action"]))
      trajectory_dict = build_dict_from_trajectory(trajectory)
      f.write(json.dumps(trajectory_dict) + "\n")


def append_dataset_to_bigquery(
    project_id: str,
    dataset_file: str,
    bigquery_dataset_id: str,
    bigquery_location: str,
    bigquery_table_id: str) -> None:
  """Appends training dataset to BigQuery table.

  Appends training dataset of `trajectories.Trajectory` in newline delimited
  JSON to a BigQuery dataset and table, using a BigQuery client.

  Args:
    project_id: GCP project ID. This is required because otherwise the
      BigQuery client will use the ID of the tenant GCP project created as a
      result of KFP, which doesn't have proper access to BigQuery.
    dataset_file: Path to a JSON file containing the training dataset.
    bigquery_dataset_id: A string of the BigQuery dataset ID in the format of
      "project.dataset".
    bigquery_location: A string of the BigQuery dataset location.
    bigquery_table_id: A string of the BigQuery table ID in the format of
      "project.dataset.table".
  """
  # Construct a BigQuery client object.
  client = bigquery.Client(project=project_id)

  # Construct a full Dataset object to send to the API.
  dataset = bigquery.Dataset(bigquery_dataset_id)

  # Specify the geographic location where the dataset should reside.
  dataset.location = bigquery_location

  # Create the dataset, or get the dataset if it exists.
  dataset = client.create_dataset(dataset, exists_ok=True, timeout=30)

  job_config = bigquery.LoadJobConfig(
      write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
      schema=[
          bigquery.SchemaField("step_type", "INT64", mode="REPEATED"),
          bigquery.SchemaField(
              "observation",
              "RECORD",
              mode="REPEATED",
              fields=[
                  bigquery.SchemaField("observation_batch", "FLOAT64",
                                       "REPEATED")
              ]),
          bigquery.SchemaField("action", "INT64", mode="REPEATED"),
          bigquery.SchemaField("policy_info", "FLOAT64", mode="REPEATED"),
          bigquery.SchemaField("next_step_type", "INT64", mode="REPEATED"),
          bigquery.SchemaField("reward", "FLOAT64", mode="REPEATED"),
          bigquery.SchemaField("discount", "FLOAT64", mode="REPEATED"),
      ],
      source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
  )

  with open(dataset_file, "rb") as source_file:
    load_job = client.load_table_from_file(
        source_file, bigquery_table_id, job_config=job_config)

  load_job.result()  # Wait for the job to complete.


def log_prediction_to_bigquery(event: Dict[str, Any], context) -> None:  # pylint: disable=unused-argument
  """Logs prediction inputs and results to BigQuery.

  Queries the MovieLens simulation environment for rewards and other info based
  on observations and predicted actions, and logs trajectory data to BigQuery.

  Serves as the Logger and the entrypoint of Cloud Functions. The Logger closes
  the feedback loop from prediction results to training data, and allows
  re-training of the policy with new training data.

  Note: In production, this function can be modified to hold the logic of
  gathering real-world feedback for observations and predicted actions,
  formulating trajectory data, and storing back into BigQuery.

  Args:
    event: Triggering event of this function.
    context: Trigerring context of this function.
      This is of type `functions_v1.context.Context` but not specified since
      it is not importable for a local environment that wants to run unit
      tests.
  """
  env_vars = get_env_vars()
  # Get a file path with permission for writing.
  dataset_file = os.path.join(tempfile.gettempdir(), env_vars.bigquery_tmp_file)

  data_bytes = base64.b64decode(event["data"])
  data_json = data_bytes.decode("utf-8")
  data = json.loads(data_json)
  observations = data["observations"]
  predicted_actions = data["predicted_actions"]

  # Create MovieLens simulation environment.
  env = movielens_py_environment.MovieLensPyEnvironment(
      env_vars.raw_data_path,
      env_vars.rank_k,
      env_vars.batch_size,
      num_movies=env_vars.num_actions,
      csv_delimiter="\t")
  environment = tf_py_environment.TFPyEnvironment(env)

  # Get environment feedback and write trajectory data.
  write_trajectories_to_file(
      dataset_file=dataset_file,
      environment=environment,
      observations=observations,
      predicted_actions=predicted_actions)

  # Add trajectory data as new training data to BigQuery.
  append_dataset_to_bigquery(
      project_id=env_vars.project_id,
      dataset_file=dataset_file,
      bigquery_dataset_id=env_vars.bigquery_dataset_id,
      bigquery_location=env_vars.bigquery_location,
      bigquery_table_id=env_vars.bigquery_table_id)
