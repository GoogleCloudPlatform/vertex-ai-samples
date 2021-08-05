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

"""The utility module for reinforcement learning policy."""
import collections
from typing import Callable, Dict, List, Optional, TypeVar

from tf_agents.agents import TFAgent
from tf_agents.bandits.agents.examples.v2 import trainer
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import TFEnvironment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.metrics.tf_metric import TFStepMetric
from tf_agents.policies import policy_saver


T = TypeVar("T")


def train(agent: TFAgent,
          environment: TFEnvironment,
          training_loops: int,
          steps_per_loop: int,
          additional_metrics: Optional[List[TFStepMetric]] = None,
          training_data_spec_transformation_fn: Optional[Callable[[T],
                                                                  T]] = None,
          run_hyperparameter_tuning: bool = False,
          root_dir: Optional[str] = None,
          artifacts_dir: Optional[str] = None) -> Dict[str, List[float]]:
  """Performs `training_loops` iterations of training on the agent's policy.

  Uses the `environment` as the problem formulation and source of immediate
  feedback and the agent's algorithm, to perform `training-loops` iterations
  of on-policy training on the policy. Has hyperparameter mode and regular
  training mode.
  If one or more baseline_reward_fns are provided, the regret is computed
  against each one of them. Here is example baseline_reward_fn:
  def baseline_reward_fn(observation, per_action_reward_fns):
   rewards = ... # compute reward for each arm
   optimal_action_reward = ... # take the maximum reward
   return optimal_action_reward

  Args:
    agent: An instance of `TFAgent`.
    environment: An instance of `TFEnvironment`.
    training_loops: An integer indicating how many training loops should be run.
    steps_per_loop: An integer indicating how many driver steps should be
      executed and presented to the trainer during each training loop.
    additional_metrics: Optional; list of metric objects to log, in addition to
      default metrics `NumberOfEpisodes`, `AverageReturnMetric`, and
      `AverageEpisodeLengthMetric`.
    training_data_spec_transformation_fn: Optional; function that transforms
      the data items before they get to the replay buffer.
    run_hyperparameter_tuning: Optional; whether this training logic is
      executed for the purpose of hyperparameter tuning. If so, then it does
      not save model artifacts.
    root_dir: Optional; path to the directory where training artifacts are
      written; usually used for a default or auto-generated location. Do not
      specify this argument if using hyperparameter tuning instead of training.
    artifacts_dir: Optional; path to an extra directory where training
      artifacts are written; usually used for a mutually agreed location from
      which artifacts will be loaded. Do not specify this argument if using
      hyperparameter tuning instead of training.

  Returns:
    A dict mapping metric names (eg. "AverageReturnMetric") to a list of
    intermediate metric values over `training_loops` iterations of training.
  """
  if run_hyperparameter_tuning and not (root_dir is None and
                                        artifacts_dir is None):
    raise ValueError("Do not specify `root_dir` or `artifacts_dir` when" +
                     " running hyperparameter tuning.")

  if training_data_spec_transformation_fn is None:
    data_spec = agent.policy.trajectory_spec
  else:
    data_spec = training_data_spec_transformation_fn(
        agent.policy.trajectory_spec)
  replay_buffer = trainer.get_replay_buffer(data_spec, environment.batch_size,
                                            steps_per_loop)

  # `step_metric` records the number of individual rounds of bandit interaction;
  # that is, (number of trajectories) * batch_size.
  step_metric = tf_metrics.EnvironmentSteps()
  metrics = [
      tf_metrics.NumberOfEpisodes(),
      tf_metrics.AverageEpisodeLengthMetric(batch_size=environment.batch_size)
  ]
  if additional_metrics:
    metrics += additional_metrics

  if isinstance(environment.reward_spec(), dict):
    metrics += [tf_metrics.AverageReturnMultiMetric(
        reward_spec=environment.reward_spec(),
        batch_size=environment.batch_size)]
  else:
    metrics += [
        tf_metrics.AverageReturnMetric(batch_size=environment.batch_size)]

  # Store intermediate metric results, indexed by metric names.
  metric_results = collections.defaultdict(list)

  if training_data_spec_transformation_fn is not None:
    add_batch_fn = lambda data: replay_buffer.add_batch(  # pylint: disable=g-long-lambda
        training_data_spec_transformation_fn(data))
  else:
    add_batch_fn = replay_buffer.add_batch

  observers = [add_batch_fn, step_metric] + metrics

  driver = dynamic_step_driver.DynamicStepDriver(
      env=environment,
      policy=agent.collect_policy,
      num_steps=steps_per_loop * environment.batch_size,
      observers=observers)

  training_loop = trainer.get_training_loop_fn(
      driver, replay_buffer, agent, steps_per_loop)
  if not run_hyperparameter_tuning:
    saver = policy_saver.PolicySaver(agent.policy)

  for _ in range(training_loops):
    training_loop()
    metric_utils.log_metrics(metrics)
    for metric in metrics:
      metric.tf_summaries(train_step=step_metric.result())
      metric_results[type(metric).__name__].append(metric.result().numpy())
  if not run_hyperparameter_tuning:
    saver.save(root_dir)
    saver.save(artifacts_dir)
  return metric_results
