# pylint: disable=W,C,R

# DO NOT MODIFY: this file is auto-generated
# See go/vmg-oss-peft-tests#command-builder-genpy


class InstructLoraCommandBuilder:

  def __init__(self):
    self._config_file = None
    self._task = None
    self._gcs_rsync_interval_secs = None
    self._pretrained_model_name_or_path = None
    self._train_dataset = None
    self._train_split = None
    self._train_template = None
    self._train_column = None
    self._output_dir = None
    self._merge_base_and_lora_output_dir = None
    self._logging_output_dir = None
    self._per_device_train_batch_size = None
    self._gradient_accumulation_steps = None
    self._lora_rank = None
    self._lora_alpha = None
    self._lora_dropout = None
    self._max_steps = None
    self._num_train_epochs = None
    self._max_seq_length = None
    self._learning_rate = None
    self._lr_scheduler_type = None
    self._precision_mode = None
    self._train_precision = None
    self._gradient_checkpointing = None
    self._example_packing = None
    self._attn_implementation = None
    self._optimizer = None
    self._warmup_ratio = None
    self._report_to = None
    self._save_steps = None
    self._logging_steps = None
    self._huggingface_access_token = None
    self._eval_dataset = None
    self._eval_column = None
    self._eval_template = None
    self._eval_split = None
    self._eval_steps = None
    self._eval_metric_name = None
    self._metric_for_best_model = None
    self._input_masking = None
    self._max_grad_norm = None
    self._logger_level = None
    self._benchmark_out_file = None
    self._tuning_data_stats_file = None
    self._enable_peft = None
    self._merge_model_precision_mode = None
    self._target_modules = None
    self._unnamed_args = None

  @property
  def config_file(self):
    return self._config_file

  @config_file.setter
  def config_file(self, val: str):
    self._config_file = val

  @property
  def task(self):
    return self._task

  @task.setter
  def task(self, val: str):
    self._task = val

  @property
  def gcs_rsync_interval_secs(self):
    return self._gcs_rsync_interval_secs

  @gcs_rsync_interval_secs.setter
  def gcs_rsync_interval_secs(self, val: str):
    self._gcs_rsync_interval_secs = val

  @property
  def pretrained_model_name_or_path(self):
    return self._pretrained_model_name_or_path

  @pretrained_model_name_or_path.setter
  def pretrained_model_name_or_path(self, val: str):
    self._pretrained_model_name_or_path = val

  @property
  def train_dataset(self):
    return self._train_dataset

  @train_dataset.setter
  def train_dataset(self, val: str):
    self._train_dataset = val

  @property
  def train_split(self):
    return self._train_split

  @train_split.setter
  def train_split(self, val: str):
    self._train_split = val

  @property
  def train_template(self):
    return self._train_template

  @train_template.setter
  def train_template(self, val: str):
    self._train_template = val

  @property
  def train_column(self):
    return self._train_column

  @train_column.setter
  def train_column(self, val: str):
    self._train_column = val

  @property
  def ckpt_dir(self):
    return self._output_dir

  @ckpt_dir.setter
  def ckpt_dir(self, val: str):
    self._output_dir = val

  @property
  def merged_model_dir(self):
    return self._merge_base_and_lora_output_dir

  @merged_model_dir.setter
  def merged_model_dir(self, val: str):
    self._merge_base_and_lora_output_dir = val

  @property
  def logging_dir(self):
    return self._logging_output_dir

  @logging_dir.setter
  def logging_dir(self, val: str):
    self._logging_output_dir = val

  @property
  def per_device_batch_size(self):
    return self._per_device_train_batch_size

  @per_device_batch_size.setter
  def per_device_batch_size(self, val: int):
    self._per_device_train_batch_size = val

  @property
  def gradient_accumulation_steps(self):
    return self._gradient_accumulation_steps

  @gradient_accumulation_steps.setter
  def gradient_accumulation_steps(self, val: int):
    self._gradient_accumulation_steps = val

  @property
  def lora_rank(self):
    return self._lora_rank

  @lora_rank.setter
  def lora_rank(self, val: int):
    self._lora_rank = val

  @property
  def lora_alpha(self):
    return self._lora_alpha

  @lora_alpha.setter
  def lora_alpha(self, val: int):
    self._lora_alpha = val

  @property
  def lora_dropout(self):
    return self._lora_dropout

  @lora_dropout.setter
  def lora_dropout(self, val: float):
    self._lora_dropout = val

  @property
  def max_steps(self):
    return self._max_steps

  @max_steps.setter
  def max_steps(self, val: int):
    self._max_steps = val

  @property
  def num_train_epochs(self):
    return self._num_train_epochs

  @num_train_epochs.setter
  def num_train_epochs(self, val: float):
    self._num_train_epochs = val

  @property
  def max_seq_length(self):
    return self._max_seq_length

  @max_seq_length.setter
  def max_seq_length(self, val: int):
    self._max_seq_length = val

  @property
  def learning_rate(self):
    return self._learning_rate

  @learning_rate.setter
  def learning_rate(self, val: float):
    self._learning_rate = val

  @property
  def lr_scheduler_type(self):
    return self._lr_scheduler_type

  @lr_scheduler_type.setter
  def lr_scheduler_type(self, val: str):
    self._lr_scheduler_type = val

  @property
  def load_precision(self):
    return self._precision_mode

  @load_precision.setter
  def load_precision(self, val: str):
    self._precision_mode = val

  @property
  def train_precision(self):
    return self._train_precision

  @train_precision.setter
  def train_precision(self, val: str):
    self._train_precision = val

  @property
  def gradient_checkpointing(self):
    return self._gradient_checkpointing

  @gradient_checkpointing.setter
  def gradient_checkpointing(self, val: bool):
    self._gradient_checkpointing = val

  @property
  def example_packing(self):
    return self._example_packing

  @example_packing.setter
  def example_packing(self, val: bool):
    self._example_packing = val

  @property
  def attn_implementation(self):
    return self._attn_implementation

  @attn_implementation.setter
  def attn_implementation(self, val: str):
    self._attn_implementation = val

  @property
  def optimizer(self):
    return self._optimizer

  @optimizer.setter
  def optimizer(self, val: str):
    self._optimizer = val

  @property
  def warmup_ratio(self):
    return self._warmup_ratio

  @warmup_ratio.setter
  def warmup_ratio(self, val: float):
    self._warmup_ratio = val

  @property
  def report_to(self):
    return self._report_to

  @report_to.setter
  def report_to(self, val: str):
    self._report_to = val

  @property
  def save_steps(self):
    return self._save_steps

  @save_steps.setter
  def save_steps(self, val: int):
    self._save_steps = val

  @property
  def logging_steps(self):
    return self._logging_steps

  @logging_steps.setter
  def logging_steps(self, val: int):
    self._logging_steps = val

  @property
  def huggingface_access_token(self):
    return self._huggingface_access_token

  @huggingface_access_token.setter
  def huggingface_access_token(self, val: str):
    self._huggingface_access_token = val

  @property
  def eval_dataset(self):
    return self._eval_dataset

  @eval_dataset.setter
  def eval_dataset(self, val: str):
    self._eval_dataset = val

  @property
  def eval_column(self):
    return self._eval_column

  @eval_column.setter
  def eval_column(self, val: str):
    self._eval_column = val

  @property
  def eval_template(self):
    return self._eval_template

  @eval_template.setter
  def eval_template(self, val: str):
    self._eval_template = val

  @property
  def eval_split(self):
    return self._eval_split

  @eval_split.setter
  def eval_split(self, val: str):
    self._eval_split = val

  @property
  def eval_steps(self):
    return self._eval_steps

  @eval_steps.setter
  def eval_steps(self, val: int):
    self._eval_steps = val

  @property
  def eval_metric_name(self):
    return self._eval_metric_name

  @eval_metric_name.setter
  def eval_metric_name(self, val: str):
    self._eval_metric_name = val

  @property
  def metric_for_best_model(self):
    return self._metric_for_best_model

  @metric_for_best_model.setter
  def metric_for_best_model(self, val: str):
    self._metric_for_best_model = val

  @property
  def input_masking(self):
    return self._input_masking

  @input_masking.setter
  def input_masking(self, val: bool):
    self._input_masking = val

  @property
  def max_grad_norm(self):
    return self._max_grad_norm

  @max_grad_norm.setter
  def max_grad_norm(self, val: float):
    self._max_grad_norm = val

  @property
  def logger_level(self):
    return self._logger_level

  @logger_level.setter
  def logger_level(self, val: str):
    self._logger_level = val

  @property
  def benchmark_out_file(self):
    return self._benchmark_out_file

  @benchmark_out_file.setter
  def benchmark_out_file(self, val: str):
    self._benchmark_out_file = val

  @property
  def tuning_data_stats_file(self):
    return self._tuning_data_stats_file

  @tuning_data_stats_file.setter
  def tuning_data_stats_file(self, val: str):
    self._tuning_data_stats_file = val

  @property
  def enable_peft(self):
    return self._enable_peft

  @enable_peft.setter
  def enable_peft(self, val: bool):
    self._enable_peft = val

  @property
  def merge_model_precision_mode(self):
    return self._merge_model_precision_mode

  @merge_model_precision_mode.setter
  def merge_model_precision_mode(self, val: str):
    self._merge_model_precision_mode = val

  @property
  def target_modules(self):
    return self._target_modules

  @target_modules.setter
  def target_modules(self, val: str):
    self._target_modules = val

  @property
  def unnamed_args(self):
    return self._unnamed_args

  @unnamed_args.setter
  def unnamed_args(self, val: list):
    self._unnamed_args = val

  def build_cmd(self) -> list[str]:
    cmd = []
    args = ''
    for k, v in self.__dict__.items():
      if k == '_unnamed_args' and v is not None:
        args += ' '.join(v)
        continue
      if v is not None:
        cmd.append(f'--{k[1:]}={v}')
    cmd.append(f'{args}')
    return cmd
