# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
"""Tests adapters of PEFT train docker."""

import inspect
import os
import time

from absl.testing import absltest
from absl.testing import parameterized
import instruct_lora_command_builder as task_cmd_builder
from safetensors import safe_open
import test_util


class AdapterTest(test_util.TestBase):

  # Needs to be accessible outside docker to check artifacts.
  _TEST_OUTPUT_DIR = os.path.expanduser('~/output')
  _MODULES_NEED_TO_BE_EXCLUDED_IN_ADAPTER = ['lm_head', 'embed_tokens']

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.test_suite_output_dir = os.path.join(
        cls._TEST_OUTPUT_DIR,
        os.path.splitext(os.path.basename(__file__))[0],
        cls.__class__.__name__,
    )

  def setUp(self):
    super().setUp()

    self.command_builder.add_env_var('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')

    self.task_cmd_builder = task_cmd_builder.InstructLoraCommandBuilder()
    self.task_cmd_builder.task = 'instruct-lora'
    self.task_cmd_builder.per_device_batch_size = 1
    self.task_cmd_builder.train_dataset = test_util.get_test_data_path(
        'peft_train_sample.jsonl'
    )
    self.task_cmd_builder.train_split = 'train'
    self.task_cmd_builder.train_column = 'input_text'
    self.task_cmd_builder.train_template = 'llama3-text-bison'
    self.task_cmd_builder.gradient_accumulation_steps = 1
    self.task_cmd_builder.lora_rank = 16
    self.task_cmd_builder.lora_alpha = 32
    self.task_cmd_builder.lora_dropout = 0.05
    self.task_cmd_builder.max_steps = 1
    self.task_cmd_builder.max_seq_length = 256
    self.task_cmd_builder.load_precision = '4bit'
    self.task_cmd_builder.gradient_checkpointing = True
    self.task_cmd_builder.attn_implementation = 'flash_attention_2'
    self.task_cmd_builder.save_steps = 10
    self.task_cmd_builder.max_steps = 3
    self.task_cmd_builder.config_file = (
        'vertex_vision_model_garden_peft/llama_fsdp_8gpu.yaml'
    )

  def setup_output_dir(self, testcase_name: str):
    testcase_output_dir = os.path.join(
        self.test_suite_output_dir, testcase_name, test_util.get_timestamp()
    )
    self.task_cmd_builder.ckpt_dir = os.path.join(
        testcase_output_dir, 'adapter'
    )
    self.task_cmd_builder.logging_dir = os.path.join(
        testcase_output_dir, 'logs'
    )

  def check_adapter_for_bad_modules(self, adapter_path):
    unwanted_modules = set()
    with safe_open(adapter_path, framework='pt', device='cpu') as f:
      for key in f.keys():
        for module in self._MODULES_NEED_TO_BE_EXCLUDED_IN_ADAPTER:
          if module in key:
            unwanted_modules.add(key)
    assert (
        not unwanted_modules
    ), f'Adapter includes unwanted modules: {unwanted_modules}'

  @parameterized.named_parameters(
      ('llama3.1-8b', 'llama3.1-8b-hf'),
      ('llama3.1-70b', 'llama3.1-70b-hf'),
      ('llama2-7b', 'llama2-7b-hf'),
  )
  def test_llama_adapters(self, model_name):
    test_function_name = inspect.stack()[0][3]
    self.setup_output_dir(f'{test_function_name}-{model_name}')
    self.task_cmd_builder.pretrained_model_name_or_path = (
        test_util.get_pretrained_model_name_or_path(model_name)
    )
    start_time = time.time()
    self.assertEqual(self.run_cmd(), 0)
    end_time = time.time()
    self.assertLess(end_time - start_time, 1000)

    adapter = os.path.join(
        self.task_cmd_builder.ckpt_dir,
        'checkpoint-final/adapter_model.safetensors',
    )
    self.check_adapter_for_bad_modules(adapter)


if __name__ == '__main__':
  absltest.main()
