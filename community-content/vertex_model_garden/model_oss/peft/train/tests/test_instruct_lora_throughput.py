# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
"""Tests to check training throughput and GPU memory consumption."""

import os
import pathlib

from absl.testing import absltest
from absl.testing import parameterized
import instruct_lora_command_builder as task_cmd_builder
import test_util


class TrainerThroughputTest(test_util.TestBase):

  _TEST_OUTPUT_DIR = os.path.expanduser('~/throughput_tests')

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.test_suite_output_dir = os.path.join(
        cls._TEST_OUTPUT_DIR, os.path.splitext(os.path.basename(__file__))[0]
    )
    if not os.path.isdir(cls.test_suite_output_dir):
      pathlib.Path(cls.test_suite_output_dir).mkdir(parents=True)

  def setUp(self):
    super().setUp()

    self.task_cmd_builder = task_cmd_builder.InstructLoraCommandBuilder()
    self.task_cmd_builder.task = 'instruct-lora'
    self.task_cmd_builder.per_device_batch_size = 1
    self.task_cmd_builder.gradient_accumulation_steps = 1
    self.task_cmd_builder.lora_rank = 16
    self.task_cmd_builder.lora_alpha = 32
    self.task_cmd_builder.lora_dropout = 0.05
    self.task_cmd_builder.learning_rate = 5e-5
    self.task_cmd_builder.warmup_ratio = 0.01
    self.task_cmd_builder.max_steps = 10
    self.task_cmd_builder.save_steps = 1000
    self.task_cmd_builder.logging_steps = 1
    self.task_cmd_builder.gradient_checkpointing = True
    self.task_cmd_builder.attn_implementation = 'flash_attention_2'
    self.task_cmd_builder.example_packing = True
    self.task_cmd_builder.train_dataset = 'mlabonne/guanaco-llama2-1k'
    self.task_cmd_builder.train_split_name = 'train'
    self.task_cmd_builder.instruct_column = 'text'
    self.task_cmd_builder.template = 'openassistant-guanaco'
    self.task_cmd_builder.ckpt_dir = '/tmp/adapter'
    self.task_cmd_builder.logging_dir = '/tmp/logs'

  def run_cmd_and_handle_failure(self):
    ret = self.run_cmd()
    if ret != 0:
      with open(self.task_cmd_builder.benchmark_out_file, 'a') as f:
        max_seq_length = self.task_cmd_builder.max_seq_length
        f.write(f'{max_seq_length/1024.0:.1f}k | failed | n/a\n')
    return ret

  @parameterized.product(
      model_name=[
          'llama3-70b-hf',
          'llama3.1-70b-hf',
          'Mistral-7B-v0.1',
          'Mixtral-8x7B-v0.1',
          'Gemma2-9b-it',
      ],
      precision=['4bit', '8bit', 'bfloat16'],
      max_seq_length=list(range(4 * 1024, 24 * 1024 + 1, 4 * 1024)),
  )
  def test_model_single_gpu(self, model_name, precision, max_seq_length):
    self.task_cmd_builder.pretrained_model_id = (
        test_util.get_pretrained_model_id(model_name)
    )
    self.task_cmd_builder.max_seq_length = max_seq_length
    self.task_cmd_builder.load_precision = precision
    self.task_cmd_builder.benchmark_out_file = os.path.join(
        self.test_suite_output_dir, f'bm_{model_name}_{precision}.txt'
    )

    self.docker_builder.add_env_var('CUDA_VISIBLE_DEVICES', '0')

    self.assertEqual(self.run_cmd_and_handle_failure(), 0)

  @parameterized.product(
      model_name=[
          'llama3-70b-hf',
          'llama3.1-70b-hf',
          'Mistral-7B-v0.1',
          'Mixtral-8x7B-v0.1',
          'Gemma2-9b-it',
      ],
      precision=['4bit', '8bit', 'bfloat16'],
      max_seq_length=list(range(4 * 1024, 24 * 1024 + 1, 4 * 1024)),
      num_gpus=[8],
      config=['deepspeed_zero2', 'deepspeed_zero3'],
  )
  def test_model_multi_gpu_deepspeed(
      self, model_name, precision, max_seq_length, num_gpus, config
  ):
    self.assertTrue(num_gpus == 4 or num_gpus == 8)
    self.task_cmd_builder.pretrained_model_id = (
        test_util.get_pretrained_model_id(model_name)
    )
    self.task_cmd_builder.max_seq_length = max_seq_length
    self.task_cmd_builder.load_precision = precision
    self.task_cmd_builder.benchmark_out_file = os.path.join(
        self.test_suite_output_dir,
        f'bm_{config}_{num_gpus}gpu_{model_name}_{precision}.txt',
    )

    self.task_cmd_builder.config_file = (
        f'vertex_vision_model_garden_peft/{config}_{num_gpus}gpu.yaml'
    )
    self.docker_builder.add_env_var(
        'CUDA_VISIBLE_DEVICES', ','.join([str(x) for x in range(0, num_gpus)])
    )

    self.assertEqual(self.run_cmd_and_handle_failure(), 0)

  @parameterized.product(
      model_name=['llama3.1-70b-hf'],
      precision=['4bit', '8bit', 'bfloat16'],
      max_seq_length=list(range(4 * 1024, 24 * 1024 + 1, 4 * 1024)),
      num_gpus=[8],
  )
  def test_model_multi_gpu_fsdp_lora(
      self, model_name, precision, max_seq_length, num_gpus
  ):
    self.task_cmd_builder.pretrained_model_id = (
        test_util.get_pretrained_model_id(model_name)
    )
    self.task_cmd_builder.max_seq_length = max_seq_length
    self.task_cmd_builder.load_precision = precision
    self.task_cmd_builder.benchmark_out_file = os.path.join(
        self.test_suite_output_dir,
        f'bm_fsdp_{num_gpus}gpu_{model_name}_{precision}.txt',
    )
    self.task_cmd_builder.config_file = (
        'vertex_vision_model_garden_peft/llama_fsdp_8gpu.yaml'
    )

    self.docker_builder.add_env_var(
        'CUDA_VISIBLE_DEVICES', ','.join([str(x) for x in range(0, num_gpus)])
    )

    self.assertEqual(self.run_cmd_and_handle_failure(), 0)

  @parameterized.product(
      model_name=['llama3.1-70b-hf'],
      precision=['bfloat16'],
      max_seq_length=list(range(4 * 1024, 24 * 1024 + 1, 4 * 1024)),
      num_gpus=[8],
  )
  def test_model_multi_gpu_fsdp_full_finetuning(
      self, model_name, precision, max_seq_length, num_gpus
  ):
    self.task_cmd_builder.pretrained_model_id = (
        test_util.get_pretrained_model_id(model_name)
    )
    self.task_cmd_builder.max_seq_length = max_seq_length
    self.task_cmd_builder.load_precision = precision
    self.task_cmd_builder.benchmark_out_file = os.path.join(
        self.test_suite_output_dir,
        f'bm_fsdp_full_finetuning_{num_gpus}gpu_{model_name}_{precision}.txt',
    )
    self.task_cmd_builder.config_file = (
        'vertex_vision_model_garden_peft/llama_fsdp_8gpu.yaml'
    )
    self.task_cmd_builder.enable_peft = False

    self.docker_builder.add_env_var(
        'CUDA_VISIBLE_DEVICES', ','.join([str(x) for x in range(0, num_gpus)])
    )

    self.assertEqual(self.run_cmd_and_handle_failure(), 0)


if __name__ == '__main__':
  absltest.main()
