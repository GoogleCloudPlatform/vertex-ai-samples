# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
"""Tests quantize model task in PEFT docker."""

import os
import time

from absl.testing import absltest
import quantize_model_command_builder as task_cmd_builder
import test_util


class QuantizeModelTest(test_util.TestBase):

  def setUp(self):
    super().setUp()

    self.docker_builder.add_env_var('CUDA_VISIBLE_DEVICES', '')
    self.docker_builder.add_mount_map(
        os.path.expanduser('~'), os.path.expanduser('~')
    )

    self.task_cmd_builder = task_cmd_builder.QuantizeModelCommandBuilder()
    self.task_cmd_builder.task = 'quantize-model'
    self.task_cmd_builder.pretrained_model_id = (
        'gs://vertex-model-garden-public-us/llama3/llama3-8b-hf'
    )
    self.task_cmd_builder.quantization_method = 'awq'
    self.task_cmd_builder.quantization_precision_mode = '4bit'
    self.task_cmd_builder.quantization_dataset_name = 'pileval'
    self.task_cmd_builder.text_column_in_quantization_dataset = 'text'
    self.task_cmd_builder.quantization_output_dir = '~/llama3-8b-hf-quantized'
    self.task_cmd_builder.device_map = None
    self.task_cmd_builder.max_memory = None
    self.task_cmd_builder.group_size = 128
    self.task_cmd_builder.desc_act = False
    self.task_cmd_builder.damp_percent = 0.1
    self.task_cmd_builder.cache_examples_on_gpu = False
    self.task_cmd_builder.awq_version = 'GEMM'

  def test_llama3_8b_model_awq_quantization(self):
    start_time = time.time()
    self.assertEqual(self.run_cmd(), 0)
    end_time = time.time()
    self.assertLess(end_time - start_time, 1.5 * 60 * 60)


if __name__ == '__main__':
  absltest.main()
