"""Utility functions."""

import logging
import subprocess
import sys
import time


def run_cmd(cmd: list[str]) -> float:
  """Runs the command and logs the output.

  Args:
    cmd: The command to run.

  Returns:
    The time it took to run the command.
  """
  cmd_str = ' \\\n'.join(cmd)
  logging.info('launching cmd: \n%s', cmd_str)
  start_time = time.time()
  subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stdout, check=True)
  elapsed_time = round(time.time() - start_time, 2)
  logging.info('Command %s finished in %0.2f seconds.', cmd_str, elapsed_time)
  return elapsed_time
