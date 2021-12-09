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

"""The Generator component for generating MovieLens simulation data."""
from typing import NamedTuple


def generate_movielens_dataset_for_bigquery(
    project_id: str,
    raw_data_path: str,
    batch_size: int,
    rank_k: int,
    num_actions: int,
    driver_steps: int,
    bigquery_tmp_file: str,
    bigquery_dataset_id: str,
    bigquery_location: str,
    bigquery_table_id: str
) -> NamedTuple("Outputs", [
    ("bigquery_dataset_id", str),
    ("bigquery_location", str),
    ("bigquery_table_id", str),
]):
  """Generates BigQuery training data using a MovieLens simulation environment.

  Serves as the Generator pipeline component:
  1. Generates `trajectories.Trajectory` data by applying a random policy on
    MovieLens simulation environment.
  2. Converts `trajectories.Trajectory` data to JSON format.
  3. Loads JSON-formatted data into BigQuery.

  This function is to be built into a Kubeflow Pipelines (KFP) component. As a
  result, this function must be entirely self-contained. This means that the
  import statements and helper functions must reside within itself.

  Args:
    project_id: GCP project ID. This is required because otherwise the BigQuery
      client will use the ID of the tenant GCP project created as a result of
      KFP, which doesn't have proper access to BigQuery.
    raw_data_path: Path to MovieLens 100K's "u.data" file.
    batch_size: Batch size of environment generated quantities eg. rewards.
    rank_k: Rank for matrix factorization in the MovieLens environment; also
      the observation dimension.
    num_actions: Number of actions (movie items) to choose from.
    driver_steps: Number of steps to run per batch.
    bigquery_tmp_file: Path to a JSON file containing the training dataset.
    bigquery_dataset_id: A string of the BigQuery dataset ID in the format of
      "project.dataset".
    bigquery_location: A string of the BigQuery dataset location.
    bigquery_table_id: A string of the BigQuery table ID in the format of
      "project.dataset.table".

  Returns:
    A NamedTuple of (`bigquery_dataset_id`, `bigquery_location`,
    `bigquery_table_id`).
  """
  # pylint: disable=g-import-not-at-top
  import collections
  import json
  from typing import Any, Dict

  from google.cloud import bigquery

  from tf_agents import replay_buffers
  from tf_agents import trajectories
  from tf_agents.bandits.agents.examples.v2 import trainer
  from tf_agents.bandits.environments import movielens_py_environment
  from tf_agents.drivers import dynamic_step_driver
  from tf_agents.environments import tf_py_environment
  from tf_agents.policies import random_tf_policy

  def generate_simulation_data(
      raw_data_path: str,
      batch_size: int,
      rank_k: int,
      num_actions: int,
      driver_steps: int) -> replay_buffers.TFUniformReplayBuffer:
    """Generates `trajectories.Trajectory` data from the simulation environment.

    Constructs a MovieLens simulation environment, and generates a set of
    `trajectories.Trajectory` data using a random policy.

    Args:
      raw_data_path: Path to MovieLens 100K's "u.data" file.
      batch_size: Batch size of environment generated quantities eg. rewards.
      rank_k: Rank for matrix factorization in the MovieLens environment; also
        the observation dimension.
      num_actions: Number of actions (movie items) to choose from.
      driver_steps: Number of steps to run per batch.

    Returns:
      A replay buffer holding randomly generated`trajectories.Trajectory` data.
    """
    # Create MovieLens simulation environment.
    env = movielens_py_environment.MovieLensPyEnvironment(
        raw_data_path,
        rank_k,
        batch_size,
        num_movies=num_actions,
        csv_delimiter="\t")
    environment = tf_py_environment.TFPyEnvironment(env)

    # Define random policy for collecting data.
    random_policy = random_tf_policy.RandomTFPolicy(
        action_spec=environment.action_spec(),
        time_step_spec=environment.time_step_spec())

    # Use replay buffer and observers to keep track of Trajectory data.
    data_spec = random_policy.trajectory_spec
    replay_buffer = trainer.get_replay_buffer(data_spec, environment.batch_size,
                                              driver_steps)
    observers = [replay_buffer.add_batch]

    # Run driver to apply the random policy in the simulation environment.
    driver = dynamic_step_driver.DynamicStepDriver(
        env=environment,
        policy=random_policy,
        num_steps=driver_steps * environment.batch_size,
        observers=observers)
    driver.run()

    return replay_buffer

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

  def write_replay_buffer_to_file(
      replay_buffer: replay_buffers.TFUniformReplayBuffer,
      batch_size: int,
      dataset_file: str) -> None:
    """Writes replay buffer data to a file, each JSON in one line.

    Each `trajectories.Trajectory` object in `replay_buffer` will be written as
    one line to the `dataset_file` in JSON format. I.e., the `dataset_file`
    would be a newline-delimited JSON file.

    Args:
      replay_buffer: A `replay_buffers.TFUniformReplayBuffer` holding
        `trajectories.Trajectory` objects.
      batch_size: Batch size of environment generated quantities eg. rewards.
      dataset_file: File path. Will be overwritten if already exists.
    """
    dataset = replay_buffer.as_dataset(sample_batch_size=batch_size)
    dataset_size = replay_buffer.num_frames().numpy()

    with open(dataset_file, "w") as f:
      for example in dataset.take(count=dataset_size):
        traj_dict = build_dict_from_trajectory(example[0])
        f.write(json.dumps(traj_dict) + "\n")

  def load_dataset_into_bigquery(
      project_id: str,
      dataset_file: str,
      bigquery_dataset_id: str,
      bigquery_location: str,
      bigquery_table_id: str) -> None:
    """Loads training dataset into BigQuery table.

    Loads training dataset of `trajectories.Trajectory` in newline delimited
    JSON into a BigQuery dataset and table, using a BigQuery client.

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

  replay_buffer = generate_simulation_data(
      raw_data_path=raw_data_path,
      batch_size=batch_size,
      rank_k=rank_k,
      num_actions=num_actions,
      driver_steps=driver_steps)

  write_replay_buffer_to_file(
      replay_buffer=replay_buffer,
      batch_size=batch_size,
      dataset_file=bigquery_tmp_file)

  load_dataset_into_bigquery(project_id, bigquery_tmp_file, bigquery_dataset_id,
                             bigquery_location, bigquery_table_id)

  outputs = collections.namedtuple(
      "Outputs",
      ["bigquery_dataset_id", "bigquery_location", "bigquery_table_id"])

  return outputs(bigquery_dataset_id, bigquery_location, bigquery_table_id)


if __name__ == "__main__":
  from kfp.components import create_component_from_func

  generate_movielens_dataset_for_bigquery_op = create_component_from_func(
    func=generate_movielens_dataset_for_bigquery,
    base_image="tensorflow/tensorflow:2.5.0",
    output_component_file="component.yaml",
    packages_to_install=[
      "google-cloud-bigquery==2.20.0",
      "tensorflow==2.5.0",
      "tf-agents==0.8.0",
    ],
  )
