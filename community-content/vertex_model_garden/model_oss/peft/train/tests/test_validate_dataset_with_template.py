"""Tests validate the dataset with template task in PEFT docker."""

from absl.testing import absltest
from absl.testing import parameterized
import test_util
import validate_dataset_with_template_command_builder as task_cmd_builder


class ValidateDatasetWithTemplateTest(test_util.TestBase):
  """Test the validate dataset with template task in PEFT docker."""

  def setUp(self):
    super().setUp()

    self.task_cmd_builder = (
        task_cmd_builder.ValidateDatasetWithTemplateCommandBuilder()
    )
    self.task_cmd_builder.task = "validate-dataset-with-template"

  @parameterized.named_parameters(
      dict(
          testcase_name="valid_rows",
          validate_top_k_rows=100,
          expected_result=0,
      ),
      dict(
          testcase_name="negative_rows",
          validate_top_k_rows=-10,
          expected_result=0,
      ),
      dict(
          testcase_name="out_of_range_rows",
          validate_top_k_rows=100000,
          expected_result=1,
      ),
  )
  def test_validate_dataset_with_template_top_k_rows(
      self,
      validate_top_k_rows,
      expected_result,
  ):
    self.task_cmd_builder.dataset_name = "timdettmers/openassistant-guanaco"
    self.task_cmd_builder.train_split_name = "train"
    self.task_cmd_builder.instruct_column_in_dataset = "text"
    self.task_cmd_builder.template = (
        "gs://cloud-nas-260507-tmp-20240724/openassistant-guanaco.json"
    )
    self.task_cmd_builder.validate_percentage_of_dataset = None
    self.task_cmd_builder.validate_k_rows_of_dataset = validate_top_k_rows
    self.task_cmd_builder.use_multiprocessing = True
    result = self.run_cmd()
    self.assertEqual(result, expected_result)

  @parameterized.named_parameters(
      dict(
          testcase_name="valid_positive_x_percent",
          validate_percentage_of_dataset=10,
          expected_result=0,
      ),
      dict(
          testcase_name="valid_negative_x_percent",
          validate_percentage_of_dataset=-10,
          expected_result=0,
      ),
      dict(
          testcase_name="invalid_positive_x_percent",
          validate_percentage_of_dataset=110,
          expected_result=1,
      ),
      dict(
          testcase_name="invalid_negative_x_percent",
          validate_percentage_of_dataset=-110,
          expected_result=1,
      ),
  )
  def test_validate_dataset_with_template_x_percent(
      self,
      validate_percentage_of_dataset,
      expected_result,
  ):
    self.task_cmd_builder.dataset_name = "timdettmers/openassistant-guanaco"
    self.task_cmd_builder.train_split_name = "train"
    self.task_cmd_builder.instruct_column_in_dataset = "text"
    self.task_cmd_builder.template = (
        "gs://cloud-nas-260507-tmp-20240724/openassistant-guanaco.json"
    )
    self.task_cmd_builder.validate_percentage_of_dataset = (
        validate_percentage_of_dataset
    )
    self.task_cmd_builder.validate_k_rows_of_dataset = None
    self.task_cmd_builder.use_multiprocessing = True
    result = self.run_cmd()
    self.assertEqual(result, expected_result)

  @parameterized.named_parameters(
      dict(
          testcase_name="invalid_default_input_column",
          dataset_name="timdettmers/openassistant-guanaco",
          split="train",
          input_column="",
          template=(
              "gs://cloud-nas-260507-tmp-20240724/openassistant-guanaco.json"
          ),
          validate_percentage_of_dataset=None,
          validate_top_k_rows=None,
          use_multiprocessing=True,
          expected_result=1,
      ),
      dict(
          testcase_name="invalid_percentage",
          dataset_name="timdettmers/openassistant-guanaco",
          split="train",
          input_column="text",
          template=(
              "gs://cloud-nas-260507-tmp-20240724/openassistant-guanaco.json"
          ),
          validate_percentage_of_dataset=110,
          validate_top_k_rows=None,
          use_multiprocessing=True,
          expected_result=1,
      ),
      dict(
          testcase_name="negative_percentage",
          dataset_name="timdettmers/openassistant-guanaco",
          split="train",
          input_column="text",
          template=(
              "gs://cloud-nas-260507-tmp-20240724/openassistant-guanaco.json"
          ),
          validate_percentage_of_dataset=-110,
          validate_top_k_rows=None,
          use_multiprocessing=True,
          expected_result=1,
      ),
      dict(
          testcase_name="empty_dataset",
          dataset_name="",
          split="train",
          input_column="text",
          template=(
              "gs://cloud-nas-260507-tmp-20240724/openassistant-guanaco.json"
          ),
          validate_percentage_of_dataset=None,
          validate_top_k_rows=None,
          use_multiprocessing=True,
          expected_result=1,
      ),
      dict(
          testcase_name="empty_split",
          dataset_name="timdettmers/openassistant-guanaco",
          split="",
          input_column="text",
          template=(
              "gs://cloud-nas-260507-tmp-20240724/openassistant-guanaco.json"
          ),
          validate_percentage_of_dataset=None,
          validate_top_k_rows=None,
          use_multiprocessing=True,
          expected_result=1,
      ),
      dict(
          testcase_name="empty_template",
          dataset_name="timdettmers/openassistant-guanaco",
          split="train",
          input_column="text",
          template="",
          validate_percentage_of_dataset=None,
          validate_top_k_rows=None,
          use_multiprocessing=True,
          expected_result=1,
      ),
      dict(
          testcase_name="wrong_gcs_template",
          dataset_name="gs://cloud-nas-260507-tmp-20240724/model-evaluation/peft_train_sample.jsonl",
          split="train",
          input_column="text",
          template="gs://cloud-nas-260507-tmp-20240724/sample-template.json",
          validate_percentage_of_dataset=None,
          validate_top_k_rows=None,
          use_multiprocessing=True,
          expected_result=1,
      ),
      dict(
          testcase_name="wrong_gcs_dataset_name",
          dataset_name="gs://cloud-nas-260507-tmp-20240724/model-evaluation/peft-train_sample.jsonl",
          split="train",
          input_column="text",
          template="gs://cloud-nas-260507-tmp-20240724/sample_template.json",
          validate_percentage_of_dataset=None,
          validate_top_k_rows=None,
          use_multiprocessing=True,
          expected_result=1,
      ),
  )
  def test_validate_dataset_with_template_invalid_input(
      self,
      dataset_name,
      split,
      input_column,
      template,
      validate_percentage_of_dataset,
      validate_top_k_rows,
      use_multiprocessing,
      expected_result,
  ):
    self.task_cmd_builder.dataset_name = dataset_name
    self.task_cmd_builder.train_split_name = split
    self.task_cmd_builder.instruct_column_in_dataset = input_column
    self.task_cmd_builder.template = template
    self.task_cmd_builder.validate_percentage_of_dataset = (
        validate_percentage_of_dataset
    )
    self.task_cmd_builder.validate_k_rows_of_dataset = validate_top_k_rows
    self.task_cmd_builder.use_multiprocessing = use_multiprocessing
    result = self.run_cmd()
    self.assertEqual(result, expected_result)

  @parameterized.named_parameters(
      dict(
          testcase_name="full_hf_dataset_with_multiprocessing",
          dataset_name="timdettmers/openassistant-guanaco",
          template=(
              "gs://cloud-nas-260507-tmp-20240724/openassistant-guanaco.json"
          ),
          validate_percentage_of_dataset=None,
          validate_top_k_rows=None,
          use_multiprocessing=True,
          expected_result=0,
      ),
      dict(
          testcase_name="full_gcs_dataset_with_multiprocessing",
          dataset_name="gs://cloud-nas-260507-tmp-20240724/model-evaluation/peft_train_sample.jsonl",
          template="gs://cloud-nas-260507-tmp-20240724/sample_template.json",
          validate_percentage_of_dataset=None,
          validate_top_k_rows=None,
          use_multiprocessing=True,
          expected_result=0,
      ),
      dict(
          testcase_name="half_dataset_with_multiprocessing",
          dataset_name="timdettmers/openassistant-guanaco",
          template=(
              "gs://cloud-nas-260507-tmp-20240724/openassistant-guanaco.json"
          ),
          validate_percentage_of_dataset=50,
          validate_top_k_rows=None,
          use_multiprocessing=True,
          expected_result=0,
      ),
      dict(
          testcase_name="top_100_rows_with_multiprocessing",
          dataset_name="timdettmers/openassistant-guanaco",
          template=(
              "gs://cloud-nas-260507-tmp-20240724/openassistant-guanaco.json"
          ),
          validate_percentage_of_dataset=None,
          validate_top_k_rows=100,
          use_multiprocessing=True,
          expected_result=0,
      ),
      dict(
          testcase_name="full_gcs_dataset_without_multiprocessing",
          dataset_name="gs://cloud-nas-260507-tmp-20240724/model-evaluation/peft_train_sample.jsonl",
          template="gs://cloud-nas-260507-tmp-20240724/sample_template.json",
          validate_percentage_of_dataset=None,
          validate_top_k_rows=None,
          use_multiprocessing=False,
          expected_result=0,
      ),
      dict(
          testcase_name="full_hf_dataset_without_multiprocessing",
          dataset_name="timdettmers/openassistant-guanaco",
          template=(
              "gs://cloud-nas-260507-tmp-20240724/openassistant-guanaco.json"
          ),
          validate_percentage_of_dataset=None,
          validate_top_k_rows=None,
          use_multiprocessing=False,
          expected_result=0,
      ),
      dict(
          testcase_name="half_dataset_without_multiprocessing",
          dataset_name="timdettmers/openassistant-guanaco",
          template=(
              "gs://cloud-nas-260507-tmp-20240724/openassistant-guanaco.json"
          ),
          validate_percentage_of_dataset=50,
          validate_top_k_rows=None,
          use_multiprocessing=False,
          expected_result=0,
      ),
      dict(
          testcase_name="top_100_rows_without_multiprocessing",
          dataset_name="timdettmers/openassistant-guanaco",
          template=(
              "gs://cloud-nas-260507-tmp-20240724/openassistant-guanaco.json"
          ),
          validate_percentage_of_dataset=None,
          validate_top_k_rows=100,
          use_multiprocessing=True,
          expected_result=0,
      ),
  )
  def test_validate_dataset_with_template_multiprocessing_option(
      self,
      dataset_name,
      template,
      validate_percentage_of_dataset,
      validate_top_k_rows,
      use_multiprocessing,
      expected_result,
  ):
    self.task_cmd_builder.dataset_name = dataset_name
    self.task_cmd_builder.train_split_name = "train"
    self.task_cmd_builder.instruct_column_in_dataset = "text"
    self.task_cmd_builder.template = template
    self.task_cmd_builder.validate_percentage_of_dataset = (
        validate_percentage_of_dataset
    )
    self.task_cmd_builder.validate_k_rows_of_dataset = validate_top_k_rows
    self.task_cmd_builder.use_multiprocessing = use_multiprocessing
    result = self.run_cmd()
    self.assertEqual(result, expected_result)


if __name__ == "__main__":
  absltest.main()
