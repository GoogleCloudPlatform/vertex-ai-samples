"""Different trainer callbacks for PEFT Trainer."""

from collections.abc import MutableMapping
import math
import time

from absl import logging
import accelerate
from transformers import TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_callback import TrainerControl
from transformers.trainer_callback import TrainerState

from util import device_stats


class TrainerStatsCallback(TrainerCallback):
  """Trainer callback to report trainer stats."""

  def __init__(self, max_seq_length, filename=None):
    self._max_seq_length = max_seq_length
    self._filename = filename

    self._partial_state = accelerate.PartialState()
    self._start_time = float('nan')
    self._prev_time = float('nan')
    self._peak_mem = 0.0
    self._avg_throughput = 0.0

  def on_log(
      self,
      args: TrainingArguments,
      state: TrainerState,
      control: TrainerControl,
      logs: MutableMapping[str, float] | None = None,
      **kwargs,
  ) -> None:
    """Calculates perplexity from train loss.

    Args:
      args: Arguments passed to the trainer.
      state: State of the trainer.
      control: Control of the trainer.
      logs: A dict of logs from the training loop.
      **kwargs: Additional keyword arguments, not used in this callback.
    """
    del kwargs  # Unused.
    if self._partial_state.is_main_process:
      train_loss = logs.get('loss') if logs is not None else None
      if train_loss is not None:
        perplexity = round(float(math.exp(train_loss)), 4)
        logs['perplexity'] = perplexity

  def on_step_end(
      self,
      args: TrainingArguments,
      state: TrainerState,
      control: TrainerControl,
      **kwargs,
  ):
    if self._partial_state.is_main_process:
      if state.global_step == 1:
        self._prev_time = time.time()
        self._prev_num_token = state.num_input_tokens_seen
        throughput = 0.0
      else:
        cur_time = time.time()
        cur_num_token = state.num_input_tokens_seen
        throughput = (cur_num_token - self._prev_num_token) / (
            cur_time - self._prev_time
        )
        self._prev_time = cur_time
        self._prev_num_token = cur_num_token
        self._avg_throughput += (throughput - self._avg_throughput) / (
            state.global_step - 1
        )

      gpu_stats = device_stats.gpu_stats()
      self._peak_mem = max(
          gpu_stats.reserved + gpu_stats.smi_diff, self._peak_mem
      )
      logging.info(
          'on_step_end: Throughput: %.2f token/s. %s, %s',
          throughput,
          device_stats.gpu_stats_str(gpu_stats),
          device_stats.cpu_stats_str(),
      )

  def on_train_begin(
      self,
      args: TrainingArguments,
      state: TrainerState,
      control: TrainerControl,
      **kwargs,
  ):
    if self._partial_state.is_main_process:
      self._start_time = time.time()
      logging.info(
          'on_train_begin: %s, %s',
          device_stats.gpu_stats_str(),
          device_stats.cpu_stats_str(),
      )

  def on_train_end(
      self,
      args: TrainingArguments,
      state: TrainerState,
      control: TrainerControl,
      **kwargs,
  ):
    if self._partial_state.is_main_process:
      train_time = time.time() - self._start_time
      throughput = state.num_input_tokens_seen / train_time
      logging.info(
          'training time %.2f s, throughput (including overhead, e.g., ckpt'
          ' saving): %.2f token/s, peak_mem: %.2f GB',
          train_time,
          throughput,
          self._peak_mem,
      )
      if self._filename:
        with open(self._filename, 'a') as out_f:
          out_f.write(
              f'{self._max_seq_length/1024.0:.1f} | {self._peak_mem:.2f} |'
              f' {self._avg_throughput:.2f}\n'
          )
