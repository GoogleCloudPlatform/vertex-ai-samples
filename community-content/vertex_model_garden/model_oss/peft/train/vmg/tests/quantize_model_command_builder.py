# pylint: disable=W,C,R

# DO NOT MODIFY: this file is auto-generated
# See go/vmg-oss-peft-tests#command-builder-genpy


class QuantizeModelCommandBuilder:

  def __init__(self):
    self._task = None
    self._pretrained_model_name_or_path = None
    self._quantization_method = None
    self._quantization_precision_mode = None
    self._quantization_dataset_name = None
    self._text_column_in_quantization_dataset = None
    self._quantization_output_dir = None
    self._device_map = None
    self._max_memory = None
    self._group_size = None
    self._desc_act = None
    self._damp_percent = None
    self._cache_examples_on_gpu = None
    self._awq_version = None

  @property
  def task(self):
    return self._task

  @task.setter
  def task(self, val: str):
    self._task = val

  @property
  def pretrained_model_name_or_path(self):
    return self._pretrained_model_name_or_path

  @pretrained_model_name_or_path.setter
  def pretrained_model_name_or_path(self, val: str):
    self._pretrained_model_name_or_path = val

  @property
  def quantization_method(self):
    return self._quantization_method

  @quantization_method.setter
  def quantization_method(self, val: str):
    self._quantization_method = val

  @property
  def quantization_precision_mode(self):
    return self._quantization_precision_mode

  @quantization_precision_mode.setter
  def quantization_precision_mode(self, val: str):
    self._quantization_precision_mode = val

  @property
  def quantization_dataset_name(self):
    return self._quantization_dataset_name

  @quantization_dataset_name.setter
  def quantization_dataset_name(self, val: str):
    self._quantization_dataset_name = val

  @property
  def text_column_in_quantization_dataset(self):
    return self._text_column_in_quantization_dataset

  @text_column_in_quantization_dataset.setter
  def text_column_in_quantization_dataset(self, val: str):
    self._text_column_in_quantization_dataset = val

  @property
  def quantization_output_dir(self):
    return self._quantization_output_dir

  @quantization_output_dir.setter
  def quantization_output_dir(self, val: str):
    self._quantization_output_dir = val

  @property
  def device_map(self):
    return self._device_map

  @device_map.setter
  def device_map(self, val: str):
    self._device_map = val

  @property
  def max_memory(self):
    return self._max_memory

  @max_memory.setter
  def max_memory(self, val: str):
    self._max_memory = val

  @property
  def group_size(self):
    return self._group_size

  @group_size.setter
  def group_size(self, val: int):
    self._group_size = val

  @property
  def desc_act(self):
    return self._desc_act

  @desc_act.setter
  def desc_act(self, val: bool):
    self._desc_act = val

  @property
  def damp_percent(self):
    return self._damp_percent

  @damp_percent.setter
  def damp_percent(self, val: float):
    self._damp_percent = val

  @property
  def cache_examples_on_gpu(self):
    return self._cache_examples_on_gpu

  @cache_examples_on_gpu.setter
  def cache_examples_on_gpu(self, val: bool):
    self._cache_examples_on_gpu = val

  @property
  def awq_version(self):
    return self._awq_version

  @awq_version.setter
  def awq_version(self, val: str):
    self._awq_version = val

  def build_cmd(self) -> str:
    cmd = []
    for k, v in self.__dict__.items():
      if v is not None:
        cmd.append(f'--{k[1:]}={v}')
    return cmd
