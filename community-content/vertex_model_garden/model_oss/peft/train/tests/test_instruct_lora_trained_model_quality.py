# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
"""Tests to make sure trained model achieves decent quality.

Right now, the metric is loss decreasing and we'll eyeball the TB graphs.
"""

import os

from absl.testing import absltest
from absl.testing import parameterized
import instruct_lora_command_builder as task_cmd_builder
import test_util


class TrainedModelQualityTest(test_util.TestBase):

  _TEST_OUTPUT_DIR = os.path.expanduser('~/output')

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

    self.task_cmd_builder = task_cmd_builder.InstructLoraCommandBuilder()
    self.task_cmd_builder.task = 'instruct-lora'
    self.task_cmd_builder.eval_tasks = 'builtin_eval'
    self.task_cmd_builder.eval_metric_name = 'loss'
    self.task_cmd_builder.per_device_batch_size = 1
    self.task_cmd_builder.gradient_accumulation_steps = 8
    self.task_cmd_builder.lora_rank = 16
    self.task_cmd_builder.lora_alpha = 32
    self.task_cmd_builder.lora_dropout = 0.05
    self.task_cmd_builder.learning_rate = 5e-5
    self.task_cmd_builder.num_epochs = 2.0
    self.task_cmd_builder.warmup_ratio = 0.01
    self.task_cmd_builder.max_steps = -1
    self.task_cmd_builder.save_steps = 10
    self.task_cmd_builder.eval_steps = 10
    self.task_cmd_builder.max_seq_length = 4096
    self.task_cmd_builder.load_precision = '4bit'
    self.task_cmd_builder.gradient_checkpointing = True
    self.task_cmd_builder.completion_only = True
    self.task_cmd_builder.attn_implementation = 'flash_attention_2'
    self.task_cmd_builder.report_to = 'tensorboard'

  def setup_output_dir(self, testcase_name: str):
    testcase_output_dir = os.path.join(
        self.test_suite_output_dir, testcase_name
    )
    self.task_cmd_builder.ckpt_dir = os.path.join(
        testcase_output_dir, 'adapter'
    )
    self.task_cmd_builder.logging_dir = os.path.join(
        testcase_output_dir, 'logs'
    )
    self.task_cmd_builder.merged_model_dir = os.path.join(
        testcase_output_dir, 'merged'
    )

  @parameterized.named_parameters(
      ('llama3-8b', 'llama3-8b-hf'),
      ('llama3.1-8b', 'llama3.1-8b-hf'),
  )
  def test_8b_model_deepspeed(self, model_name):
    self.setup_output_dir(f'test_deepspeed_{model_name}')
    self.task_cmd_builder.pretrained_model_id = (
        test_util.get_pretrained_model_id(model_name)
    )
    self.task_cmd_builder.train_dataset = test_util.get_test_data_path(
        'peft_train_sample.jsonl'
    )
    self.task_cmd_builder.train_split_name = 'train'
    self.task_cmd_builder.instruct_column = 'input_text'
    self.task_cmd_builder.template = 'llama3-text-bison'
    self.task_cmd_builder.eval_dataset = test_util.get_test_data_path(
        'peft_eval_sample.jsonl'
    )
    self.task_cmd_builder.eval_split_name = 'train'
    self.task_cmd_builder.eval_instruct_column = (
        self.task_cmd_builder.instruct_column
    )
    self.task_cmd_builder.eval_template = self.task_cmd_builder.template

    self.docker_builder.add_env_var('CUDA_VISIBLE_DEVICES', '0')

    self.assertEqual(self.run_cmd(), 0)

  @parameterized.named_parameters(
      ('llama3-70b', 'llama3-70b-hf'),
      ('llama3.1-70b', 'llama3.1-70b-hf'),
  )
  def test_70b_model_deepspeed(self, model_name):
    self.setup_output_dir(f'test_deepspeed_{model_name}')
    self.task_cmd_builder.pretrained_model_id = (
        test_util.get_pretrained_model_id(model_name)
    )
    self.task_cmd_builder.config_file = (
        'vertex_vision_model_garden_peft/deepspeed_zero2_8gpu.yaml'
    )
    self.task_cmd_builder.train_dataset = 'timdettmers/openassistant-guanaco'
    self.task_cmd_builder.train_split_name = 'train'
    self.task_cmd_builder.instruct_column = 'text'
    self.task_cmd_builder.template = 'openassistant-guanaco'
    self.task_cmd_builder.eval_dataset = self.task_cmd_builder.train_dataset
    self.task_cmd_builder.eval_split_name = 'test'
    self.task_cmd_builder.eval_instruct_column = (
        self.task_cmd_builder.instruct_column
    )
    self.task_cmd_builder.eval_template = self.task_cmd_builder.template

    self.docker_builder.add_env_var('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')

    self.assertEqual(self.run_cmd(), 0)

  @parameterized.named_parameters(
      ('llama3-70b', 'llama3-70b-hf'),
      ('llama3.1-70b', 'llama3.1-70b-hf'),
  )
  def test_70b_model_fsdp(self, model_name):
    self.setup_output_dir(f'test_fsdp_{model_name}')
    self.task_cmd_builder.pretrained_model_id = (
        test_util.get_pretrained_model_id(model_name)
    )
    self.task_cmd_builder.config_file = (
        'vertex_vision_model_garden_peft/llama_fsdp_8gpu.yaml'
    )
    self.task_cmd_builder.train_dataset = 'timdettmers/openassistant-guanaco'
    self.task_cmd_builder.train_split_name = 'train'
    self.task_cmd_builder.instruct_column = 'text'
    self.task_cmd_builder.template = 'openassistant-guanaco'
    self.task_cmd_builder.eval_dataset = self.task_cmd_builder.train_dataset
    self.task_cmd_builder.eval_split_name = 'test'
    self.task_cmd_builder.eval_instruct_column = (
        self.task_cmd_builder.instruct_column
    )
    self.task_cmd_builder.eval_template = self.task_cmd_builder.template

    self.docker_builder.add_env_var('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')

    self.assertEqual(self.run_cmd(), 0)


if __name__ == '__main__':
  absltest.main()
