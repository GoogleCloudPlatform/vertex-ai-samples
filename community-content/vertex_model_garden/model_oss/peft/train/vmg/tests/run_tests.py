"""Run the tests from docker command line."""

import subprocess
import sys
from typing import Sequence

from absl import app
from absl import flags


_ALLOWED_TEST_FILE_PATHS = (
    "test_instruct_lora_adapters",
    "test_instruct_lora_features",
    "test_instruct_lora_throughput",
    "test_instruct_lora_trained_model_quality",
    "test_validate_dataset_with_template",
)

_TEST_FILE_PATH = flags.DEFINE_multi_enum(
    "test_file_path",
    None,
    _ALLOWED_TEST_FILE_PATHS + ("all",),
    "The test file path.",
    required=True,
)

_IS_AUTOMATED_TEST = flags.DEFINE_bool(
    "is_automated_test",
    True,
    "Whether the test is an automated test.",
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  test_file_path = _TEST_FILE_PATH.value
  if "all" in test_file_path:
    test_file_path = _ALLOWED_TEST_FILE_PATHS

  for test_file in test_file_path:
    cmd = [
        "python3",
        f"vertex_vision_model_garden_peft/tests/{test_file}.py",
    ]
    if (
        test_file == "test_instruct_lora_throughput"
        and _IS_AUTOMATED_TEST.value
    ):
      subprocess.run(
          cmd + ["--", "-k", "peft_train_image_automated_test"],
          stdout=sys.stdout,
          stderr=sys.stdout,
          check=True,
      )
    else:
      subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stdout, check=True)


if __name__ == "__main__":
  app.run(main)
