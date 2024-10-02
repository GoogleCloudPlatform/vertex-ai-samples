"""Different trainer callbacks for PEFT Trainer."""

import time

from absl import logging
import accelerate
from transformers import TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_callback import TrainerControl
from transformers.trainer_callback import TrainerState

from vertex_vision_model_garden_peft.train.vmg import utils


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
        delta_t = float('nan')
      else:
        cur_time = time.time()
        delta_t = cur_time - self._prev_time
        self._prev_time = cur_time
        self._avg_throughput += (delta_t - self._avg_throughput) / (
            state.global_step - 1
        )

      gpu_stats = utils.gpu_stats()
      self._peak_mem = max(gpu_stats.total_mem, self._peak_mem)
      logging.info(
          'on_step_end: %s, throughput: %.2f s/it',
          utils.gpu_stats_str(gpu_stats),
          delta_t,
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
      logging.info('on_train_begin: %s', utils.gpu_stats_str())

  def on_train_end(
      self,
      args: TrainingArguments,
      state: TrainerState,
      control: TrainerControl,
      **kwargs,
  ):
    if self._partial_state.is_main_process:
      train_time = time.time() - self._start_time
      logging.info(
          'training time %.2f s, throughput: %.2f s/it, peak_mem: %.2f GB',
          train_time,
          self._avg_throughput,
          self._peak_mem,
      )
      if self._filename:
        with open(self._filename, 'a') as out_f:
          out_f.write(
              f'{self._max_seq_length/1024.0:.1f}k | {self._peak_mem:.2f} |'
              f' {self._avg_throughput:.2f}\n'
          )
