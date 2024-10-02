# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
"""Tests various features of PEFT train docker."""

import os
import time

from absl.testing import absltest
from absl.testing import parameterized
import instruct_lora_command_builder as task_cmd_builder
import test_util


class GcsUploadDownloadTest(test_util.TestBase):

  def setUp(self):
    super().setUp()

    self.docker_builder.add_env_var('CUDA_VISIBLE_DEVICES', '0')

    self.task_cmd_builder = task_cmd_builder.InstructLoraCommandBuilder()
    self.task_cmd_builder.task = 'instruct-lora'
    self.task_cmd_builder.per_device_batch_size = 1
    self.task_cmd_builder.train_dataset = test_util.get_test_data_path(
        'peft_train_sample.jsonl'
    )
    self.task_cmd_builder.train_split_name = 'train'
    self.task_cmd_builder.instruct_column = 'input_text'
    self.task_cmd_builder.template = 'llama3-text-bison'
    self.task_cmd_builder.gradient_accumulation_steps = 1
    self.task_cmd_builder.lora_rank = 16
    self.task_cmd_builder.lora_alpha = 32
    self.task_cmd_builder.lora_dropout = 0.05
    self.task_cmd_builder.max_steps = 1
    self.task_cmd_builder.max_seq_length = 256
    self.task_cmd_builder.load_precision = '4bit'
    self.task_cmd_builder.gradient_checkpointing = True
    self.task_cmd_builder.attn_implementation = 'flash_attention_2'
    self.task_cmd_builder.ckpt_dir = '/tmp'

  @parameterized.named_parameters(
      (
          'llama3_8b_gcs',
          'gs://vertex-model-garden-public-us/llama3/llama3-8b-hf',
      ),
      ('llama2_7b_hf', 'NousResearch/Llama-2-7b-hf'),
  )
  def test_model_download_single_process(self, pretrained_model_id):
    self.task_cmd_builder.pretrained_model_id = (
        test_util.get_pretrained_model_id(pretrained_model_id)
    )
    start_time = time.time()
    self.assertEqual(self.run_cmd(), 0)
    end_time = time.time()
    self.assertLess(end_time - start_time, 5 * 60.0)

  @parameterized.named_parameters(
      (
          'llama3_8b_gcs',
          'gs://vertex-model-garden-public-us/llama3/llama3-8b-hf',
      ),
      ('llama2_7b_hf', 'NousResearch/Llama-2-7b-hf'),
  )
  def test_model_download_multi_process(self, pretrained_model_id):
    self.task_cmd_builder.pretrained_model_id = (
        test_util.get_pretrained_model_id(pretrained_model_id)
    )
    self.task_cmd_builder.config_file = (
        'vertex_vision_model_garden_peft/llama_fsdp_8gpu.yaml'
    )

    self.docker_builder.add_env_var('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')

    start_time = time.time()
    self.assertEqual(self.run_cmd(), 0)
    end_time = time.time()
    self.assertLess(end_time - start_time, 5 * 60.0)

  def test_70b_model_download(self):
    self.task_cmd_builder.pretrained_model_id = (
        'gs://vertex-model-garden-public-us/llama3/llama3-70b-hf'
    )
    start_time = time.time()
    self.assertEqual(self.run_cmd(), 0)
    end_time = time.time()
    self.assertLess(end_time - start_time, 10 * 60.0)

  @parameterized.named_parameters(
      ('merged-without-upload', '/tmp/merged'),
      ('merged-and-upload-to-gcs', 'gs://vmg-test-ttl-1y/tests/merged'),
  )
  def test_model_merge(self, output_dir):
    self.task_cmd_builder.pretrained_model_id = (
        test_util.get_pretrained_model_id('llama3.1-8b-hf')
    )

    ckpt_dir = os.path.join(
        output_dir,
        f'output-{test_util.get_timestamp()}',
    )
    self.task_cmd_builder.ckpt_dir = ckpt_dir
    self.task_cmd_builder.merged_model_dir = os.path.join(ckpt_dir, 'merged')
    self.task_cmd_builder.logging_dir = '/tmp/logging'

    start_time = time.time()
    self.assertEqual(self.run_cmd(), 0)
    end_time = time.time()
    self.assertLess(end_time - start_time, 5 * 60.0)

  def test_model_fp8_conversion(self):
    self.task_cmd_builder.pretrained_model_id = (
        test_util.get_pretrained_model_id('llama3.1-8b-hf')
    )

    ckpt_dir = f'/tmp/output/output-{test_util.get_timestamp()}'
    self.task_cmd_builder.ckpt_dir = ckpt_dir
    self.task_cmd_builder.merged_model_dir = os.path.join(ckpt_dir, 'merged')
    self.task_cmd_builder.logging_dir = os.path.join(ckpt_dir, 'logging')
    self.task_cmd_builder.merge_model_precision_mode = 'float8'

    start_time = time.time()
    self.assertEqual(self.run_cmd(), 0)
    end_time = time.time()
    self.assertLess(end_time - start_time, 5 * 60.0)

  @parameterized.named_parameters(
      ('merged-without-upload', '/tmp/merged'),
      ('merged-and-upload-to-gcs', 'gs://vmg-test-ttl-1y/tests/merged'),
  )
  def test_model_merge_and_upload_deepspeed(self, merged_model_dir):
    self.task_cmd_builder.pretrained_model_id = (
        test_util.get_pretrained_model_id('llama3.1-8b-hf')
    )
    self.task_cmd_builder.config_file = (
        'vertex_vision_model_garden_peft/deepspeed_zero3_8gpu.yaml'
    )
    self.task_cmd_builder.merged_model_dir = os.path.join(
        merged_model_dir, f'merged-{test_util.get_timestamp()}'
    )

    self.docker_builder.add_env_var('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
    start_time = time.time()
    self.assertEqual(self.run_cmd(), 0)
    end_time = time.time()
    self.assertLess(end_time - start_time, 5 * 60.0)

  @parameterized.named_parameters(
      ('save-only-last', 10),
      ('save-multiple-times', 1),
  )
  def test_llama3_8b_save_and_merge_8_gpus_fsdp(self, save_steps):
    self.task_cmd_builder.pretrained_model_id = (
        test_util.get_pretrained_model_id('llama3.1-8b-hf')
    )
    self.task_cmd_builder.save_steps = save_steps
    self.task_cmd_builder.max_steps = 3
    self.task_cmd_builder.config_file = (
        'vertex_vision_model_garden_peft/llama_fsdp_8gpu.yaml'
    )
    self.task_cmd_builder.merged_model_dir = '/tmp/merged'

    self.docker_builder.add_env_var('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
    start_time = time.time()
    self.assertEqual(self.run_cmd(), 0)
    end_time = time.time()
    self.assertLess(end_time - start_time, 9 * 60.0)


class TemplateAndDataStatsTest(test_util.TestBase):

  def setUp(self):
    super().setUp()

    self.docker_builder.add_env_var('CUDA_VISIBLE_DEVICES', '0')

    self.task_cmd_builder = task_cmd_builder.InstructLoraCommandBuilder()
    self.task_cmd_builder.task = 'instruct-lora'
    self.task_cmd_builder.per_device_batch_size = 1
    self.task_cmd_builder.gradient_accumulation_steps = 1
    self.task_cmd_builder.lora_rank = 16
    self.task_cmd_builder.lora_alpha = 32
    self.task_cmd_builder.lora_dropout = 0.05
    self.task_cmd_builder.max_steps = 1
    self.task_cmd_builder.max_seq_length = 256
    self.task_cmd_builder.load_precision = '4bit'
    self.task_cmd_builder.gradient_checkpointing = True
    self.task_cmd_builder.attn_implementation = 'flash_attention_2'
    self.task_cmd_builder.ckpt_dir = '/tmp'

  def test_openai_chat_template(self):
    self.task_cmd_builder.pretrained_model_id = (
        test_util.get_pretrained_model_id('llama3.1-8b-hf')
    )
    self.task_cmd_builder.train_dataset = test_util.get_test_data_path(
        'openai-multi-chat-example-data.jsonl'
    )
    self.task_cmd_builder.train_split_name = 'train'
    self.task_cmd_builder.instruct_column = 'messages'
    self.task_cmd_builder.template = 'llama3'

    self.assertEqual(self.run_cmd(), 0)

  def test_openai_completion_template(self):
    self.task_cmd_builder.pretrained_model_id = (
        test_util.get_pretrained_model_id('llama3.1-8b-hf')
    )
    self.task_cmd_builder.train_dataset = test_util.get_test_data_path(
        'openai-completion-example-data.jsonl'
    )
    self.task_cmd_builder.train_split_name = 'train'
    self.task_cmd_builder.instruct_column = 'prompt'
    self.task_cmd_builder.template = 'openai-completion'

    self.assertEqual(self.run_cmd(), 0)

  def test_data_stats_chat_template(self):
    self.task_cmd_builder.pretrained_model_id = (
        test_util.get_pretrained_model_id('llama3.1-8b-hf')
    )
    self.task_cmd_builder.config_file = (
        'vertex_vision_model_garden_peft/llama_fsdp_8gpu.yaml'
    )
    self.task_cmd_builder.train_dataset = test_util.get_test_data_path(
        'openai-multi-chat-example-data.jsonl'
    )
    self.task_cmd_builder.train_split_name = 'train'
    self.task_cmd_builder.instruct_column = 'messages'
    self.task_cmd_builder.template = 'llama3'
    self.task_cmd_builder.tuning_data_stats_file = '/tmp/data-stats.json'
    self.docker_builder.add_env_var('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')

    self.assertEqual(self.run_cmd(), 0)

  def test_data_stats_completion_template(self):
    self.task_cmd_builder.pretrained_model_id = (
        test_util.get_pretrained_model_id('llama3.1-8b-hf')
    )
    self.task_cmd_builder.config_file = (
        'vertex_vision_model_garden_peft/llama_fsdp_8gpu.yaml'
    )
    self.task_cmd_builder.train_dataset = test_util.get_test_data_path(
        'openai-completion-example-data.jsonl'
    )
    self.task_cmd_builder.train_split_name = 'train'
    self.task_cmd_builder.instruct_column = 'prompt'
    self.task_cmd_builder.template = 'openai-completion'
    self.task_cmd_builder.tuning_data_stats_file = '/tmp/data-stats.json'

    self.docker_builder.add_env_var('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
    self.assertEqual(self.run_cmd(), 0)


class TargetModulesTest(test_util.TestBase):

  def setUp(self):
    super().setUp()

    self.docker_builder.add_env_var('CUDA_VISIBLE_DEVICES', '0')

    self.task_cmd_builder = task_cmd_builder.InstructLoraCommandBuilder()
    self.task_cmd_builder.task = 'instruct-lora'
    self.task_cmd_builder.per_device_batch_size = 1
    self.task_cmd_builder.train_dataset = test_util.get_test_data_path(
        'peft_train_sample.jsonl'
    )
    self.task_cmd_builder.train_split_name = 'train'
    self.task_cmd_builder.instruct_column = 'input_text'
    self.task_cmd_builder.template = 'llama3-text-bison'
    self.task_cmd_builder.gradient_accumulation_steps = 1
    self.task_cmd_builder.lora_rank = 16
    self.task_cmd_builder.lora_alpha = 32
    self.task_cmd_builder.lora_dropout = 0.05
    self.task_cmd_builder.max_steps = 1
    self.task_cmd_builder.max_seq_length = 256
    self.task_cmd_builder.load_precision = '4bit'
    self.task_cmd_builder.gradient_checkpointing = True
    self.task_cmd_builder.attn_implementation = 'flash_attention_2'
    self.task_cmd_builder.ckpt_dir = '/tmp'

  def test_target_modules(self):
    self.task_cmd_builder.pretrained_model_id = (
        test_util.get_pretrained_model_id('llama3.1-8b-hf')
    )
    self.task_cmd_builder.target_modules = 'q_proj, v_proj, k_proj'

    self.assertEqual(self.run_cmd(), 0)


if __name__ == '__main__':
  absltest.main()
