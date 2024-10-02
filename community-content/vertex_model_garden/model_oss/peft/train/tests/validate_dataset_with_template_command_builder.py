# pylint: disable=W,C,R

# DO NOT MODIFY: this file is auto-generated
# See go/vmg-oss-peft-tests#command-builder-genpy


class ValidateDatasetWithTemplateCommandBuilder:

  def __init__(self):
    self._task = None
    self._template = None
    self._dataset_name = None
    self._train_split_name = None
    self._instruct_column_in_dataset = None
    self._use_multiprocessing = None
    self._validate_k_rows_of_dataset = None
    self._validate_percentage_of_dataset = None

  @property
  def task(self):
    return self._task

  @task.setter
  def task(self, val: str):
    self._task = val

  @property
  def template(self):
    return self._template

  @template.setter
  def template(self, val: str):
    self._template = val

  @property
  def dataset_name(self):
    return self._dataset_name

  @dataset_name.setter
  def dataset_name(self, val: str):
    self._dataset_name = val

  @property
  def train_split_name(self):
    return self._train_split_name

  @train_split_name.setter
  def train_split_name(self, val: str):
    self._train_split_name = val

  @property
  def instruct_column_in_dataset(self):
    return self._instruct_column_in_dataset

  @instruct_column_in_dataset.setter
  def instruct_column_in_dataset(self, val: str):
    self._instruct_column_in_dataset = val

  @property
  def use_multiprocessing(self):
    return self._use_multiprocessing

  @use_multiprocessing.setter
  def use_multiprocessing(self, val: bool):
    self._use_multiprocessing = val

  @property
  def validate_k_rows_of_dataset(self):
    return self._validate_k_rows_of_dataset

  @validate_k_rows_of_dataset.setter
  def validate_k_rows_of_dataset(self, val: int):
    self._validate_k_rows_of_dataset = val

  @property
  def validate_percentage_of_dataset(self):
    return self._validate_percentage_of_dataset

  @validate_percentage_of_dataset.setter
  def validate_percentage_of_dataset(self, val: int):
    self._validate_percentage_of_dataset = val

  def build_cmd(self) -> str:
    cmd = []
    for k, v in self.__dict__.items():
      if v is not None:
        cmd.append(f'--{k[1:]}={v}')
    return cmd
