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

"""The Trainer component for training a policy on TFRecord files."""
# Import for the function return value type.
from typing import NamedTuple  # pylint: disable=unused-import


def train_reinforcement_learning_policy(
    training_artifacts_dir: str,
    tfrecord_file: str,
    num_epochs: int,
    rank_k: int,
    num_actions: int,
    tikhonov_weight: float,
    agent_alpha: float
) -> NamedTuple("Outputs", [
    ("training_artifacts_dir", str),
]):
  """Implements off-policy training for a policy on dataset of TFRecord files.

  The Trainer's task is to submit a remote training job to Vertex AI, with the
  training logic of a specified custom training container. The task will be
  handled by: `kfp.v2.google.experimental.run_as_aiplatform_custom_job` (which
  takes in the component made from this placeholder function)

  This function is to be built into a Kubeflow Pipelines (KFP) component. As a
  result, this function must be entirely self-contained. This means that the
  import statements and helper functions must reside within itself.

  Args:
    training_artifacts_dir: Path to store the Trainer artifacts (trained
      policy).
    tfrecord_file: Path to file to write the ingestion result TFRecords.
    num_epochs: Number of training epochs.
    rank_k: Rank for matrix factorization in the MovieLens environment; also
      the observation dimension.
    num_actions: Number of actions (movie items) to choose from.
    tikhonov_weight: LinUCB Tikhonov regularization weight of the Trainer.
    agent_alpha: LinUCB exploration parameter that multiplies the confidence
      intervals of the Trainer.

  Returns:
    A NamedTuple of (`training_artifacts_dir`).
  """
  # pylint: disable=g-import-not-at-top
  import collections
  from typing import Dict, List, NamedTuple  # pylint: disable=redefined-outer-name,reimported

  import tensorflow as tf

  from tf_agents import agents
  from tf_agents import policies
  from tf_agents import trajectories
  from tf_agents.bandits.agents import lin_ucb_agent
  from tf_agents.policies import policy_saver
  from tf_agents.specs import tensor_spec

  import logging

  per_arm = False  # Using the non-per-arm version of the MovieLens environment.

  # Mapping from feature name to serialized value
  feature_description = {
      "step_type": tf.io.FixedLenFeature((), tf.string),
      "observation": tf.io.FixedLenFeature((), tf.string),
      "action": tf.io.FixedLenFeature((), tf.string),
      "policy_info": tf.io.FixedLenFeature((), tf.string),
      "next_step_type": tf.io.FixedLenFeature((), tf.string),
      "reward": tf.io.FixedLenFeature((), tf.string),
      "discount": tf.io.FixedLenFeature((), tf.string),
  }

  def _parse_record(raw_record: tf.Tensor) -> Dict[str, tf.Tensor]:
    """Parses a serialized `tf.train.Example` proto.

    Args:
      raw_record: A serialized data record of a `tf.train.Example` proto.

    Returns:
      A dict mapping feature names to values as `tf.Tensor` objects of type
      string containing serialized protos, following `feature_description`.
    """
    return tf.io.parse_single_example(raw_record, feature_description)

  def build_trajectory(
      parsed_record: Dict[str, tf.Tensor],
      policy_info: policies.utils.PolicyInfo) -> trajectories.Trajectory:
    """Builds a `trajectories.Trajectory` object from `parsed_record`.

    Args:
      parsed_record: A dict mapping feature names to values as `tf.Tensor`
        objects of type string containing serialized protos.
      policy_info: Policy information specification.

    Returns:
      A `trajectories.Trajectory` object that contains values as de-serialized
      `tf.Tensor` objects from `parsed_record`.
    """
    return trajectories.Trajectory(
        step_type=tf.expand_dims(
            tf.io.parse_tensor(parsed_record["step_type"], out_type=tf.int32),
            axis=1),
        observation=tf.expand_dims(
            tf.io.parse_tensor(
                parsed_record["observation"], out_type=tf.float32),
            axis=1),
        action=tf.expand_dims(
            tf.io.parse_tensor(parsed_record["action"], out_type=tf.int32),
            axis=1),
        policy_info=policy_info,
        next_step_type=tf.expand_dims(
            tf.io.parse_tensor(
                parsed_record["next_step_type"], out_type=tf.int32),
            axis=1),
        reward=tf.expand_dims(
            tf.io.parse_tensor(parsed_record["reward"], out_type=tf.float32),
            axis=1),
        discount=tf.expand_dims(
            tf.io.parse_tensor(parsed_record["discount"], out_type=tf.float32),
            axis=1))

  def train_policy_on_trajectory(
      agent: agents.TFAgent,
      tfrecord_file: str,
      num_epochs: int
  ) -> NamedTuple("TrainOutputs", [
      ("policy", policies.TFPolicy),
      ("train_loss", Dict[str, List[float]]),
  ]):
    """Trains the policy in `agent` on the dataset of `tfrecord_file`.

    Parses `tfrecord_file` as `tf.train.Example` objects, packages them into
    `trajectories.Trajectory` objects, and trains the agent's policy on these
    trajectory objects.

    Args:
      agent: A TF-Agents agent that carries the policy to train.
      tfrecord_file: Path to the TFRecord file containing the training dataset.
      num_epochs: Number of epochs to train the policy.

    Returns:
      A NamedTuple of (a trained TF-Agents policy, a dict mapping from
      "epoch<i>" to lists of loss values produced at each training step).
    """
    raw_dataset = tf.data.TFRecordDataset([tfrecord_file])
    parsed_dataset = raw_dataset.map(_parse_record)

    train_loss = collections.defaultdict(list)
    for epoch in range(num_epochs):
      for parsed_record in parsed_dataset:
        trajectory = build_trajectory(parsed_record, agent.policy.info_spec)
        loss, _ = agent.train(trajectory)
        train_loss[f"epoch{epoch + 1}"].append(loss.numpy())

    train_outputs = collections.namedtuple(
        "TrainOutputs",
        ["policy", "train_loss"])
    return train_outputs(agent.policy, train_loss)

  def execute_training_and_save_policy(
      training_artifacts_dir: str,
      tfrecord_file: str,
      num_epochs: int,
      rank_k: int,
      num_actions: int,
      tikhonov_weight: float,
      agent_alpha: float) -> None:
    """Executes training for the policy and saves the policy.

    Args:
      training_artifacts_dir: Path to store the Trainer artifacts (trained
        policy).
      tfrecord_file: Path to file to write the ingestion result TFRecords.
      num_epochs: Number of training epochs.
      rank_k: Rank for matrix factorization in the MovieLens environment; also
        the observation dimension.
      num_actions: Number of actions (movie items) to choose from.
      tikhonov_weight: LinUCB Tikhonov regularization weight of the Trainer.
      agent_alpha: LinUCB exploration parameter that multiplies the confidence
        intervals of the Trainer.
    """
    # Define time step and action specs for one batch.
    time_step_spec = trajectories.TimeStep(
        step_type=tensor_spec.TensorSpec(
            shape=(), dtype=tf.int32, name="step_type"),
        reward=tensor_spec.TensorSpec(
            shape=(), dtype=tf.float32, name="reward"),
        discount=tensor_spec.BoundedTensorSpec(
            shape=(), dtype=tf.float32, name="discount", minimum=0.,
            maximum=1.),
        observation=tensor_spec.TensorSpec(
            shape=(rank_k,), dtype=tf.float32,
            name="observation"))

    action_spec = tensor_spec.BoundedTensorSpec(
        shape=(),
        dtype=tf.int32,
        name="action",
        minimum=0,
        maximum=num_actions - 1)

    # Define RL agent/algorithm.
    agent = lin_ucb_agent.LinearUCBAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        tikhonov_weight=tikhonov_weight,
        alpha=agent_alpha,
        dtype=tf.float32,
        accepts_per_arm_features=per_arm)
    agent.initialize()
    logging.info("TimeStep Spec (for each batch):\n%s\n", agent.time_step_spec)
    logging.info("Action Spec (for each batch):\n%s\n", agent.action_spec)

    # Perform off-policy training.
    policy, _ = train_policy_on_trajectory(
        agent=agent,
        tfrecord_file=tfrecord_file,
        num_epochs=num_epochs)

    # Save trained policy.
    saver = policy_saver.PolicySaver(policy)
    saver.save(training_artifacts_dir)

  execute_training_and_save_policy(
      training_artifacts_dir=training_artifacts_dir,
      tfrecord_file=tfrecord_file,
      num_epochs=num_epochs,
      rank_k=rank_k,
      num_actions=num_actions,
      tikhonov_weight=tikhonov_weight,
      agent_alpha=agent_alpha)

  outputs = collections.namedtuple(
      "Outputs",
      ["training_artifacts_dir"])

  return outputs(training_artifacts_dir)


if __name__ == "__main__":
  from kfp.components import create_component_from_func

  train_reinforcement_learning_policy_op = create_component_from_func(
    func=train_reinforcement_learning_policy,
    base_image="tensorflow/tensorflow:2.5.0",
    output_component_file="component.yaml",
    packages_to_install=[
      "tensorflow==2.5.0",
      "tf-agents==0.8.0",
    ],
  )
