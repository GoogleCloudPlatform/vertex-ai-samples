"""Sync local directory to GCS directory using rsync."""

import multiprocessing
import os
import subprocess
import time
from typing import Optional, Sequence, Tuple

from absl import logging

from util import constants
from util import fileutils

_GCS_COMMAND_RETRIES = 3
_RSYNC_RETRY_INTERVAL_SECS = 30


def is_gcs_or_gcsfuse_path(path: str) -> bool:
  """Returns if the path is a GCS or gcsfuse path.

  Args:
    path: The path to check.

  Returns:
    True if the path is a GCS or gcsfuse path.
  """
  return path.startswith(
      (constants.GCS_URI_PREFIX, constants.GCSFUSE_URI_PREFIX)
  )


def manage_sync_path(
    path: str, node_rank: Optional[int] = None
) -> Tuple[str, str]:
  """Returns local dir and GCS location for the given path if the given path is a GCS or gcsfuse path.

  It will also create a local directory if it does not exist. Otherwise, it
  returns the same path.

  Args:
    path: The local or GCS path to manage.
    node_rank: The node rank to be appended to the GCS path.

  Returns:
    The local and GCS paths.
  """
  local_dir = path
  gcs_dir = path
  if is_gcs_or_gcsfuse_path(path):
    local_dir = os.path.join(
        constants.LOCAL_OUTPUT_DIR,
        fileutils.force_gcs_fuse_path(path)[1:],
    )
    gcs_dir = fileutils.force_gcs_path(path)
  if not os.path.exists(local_dir):
    os.makedirs(local_dir, exist_ok=True)

  if node_rank is None:
    return local_dir, gcs_dir
  return local_dir, os.path.join(gcs_dir, f"node-{node_rank}")


def setup_gcs_rsync(
    dirs_to_sync: Sequence[Tuple[str, str]],
    mp_queue: multiprocessing.Queue,
    gcs_rsync_interval_secs: int,
) -> multiprocessing.Process:
  """Sets up the GCS rsync process.

  Args:
    dirs_to_sync: The absolute directory paths which will be synced to GCS.
    mp_queue: The multiprocessing queue to check if the training is finished.
    gcs_rsync_interval_secs: Integer, interval in seconds to run gcs rsync.

  Returns:
    The GCS rsync process.
  """
  rsync_process = multiprocessing.Process(
      target=start_gcs_rsync,
      args=(dirs_to_sync, mp_queue, gcs_rsync_interval_secs),
  )
  rsync_process.start()
  return rsync_process


def cleanup_gcs_rsync(
    rsync_process: multiprocessing.Process, mp_queue: multiprocessing.Queue
) -> None:
  """Cleans up the GCS rsync process.

  Args:
    rsync_process: The GCS rsync process.
    mp_queue: The multiprocessing queue.
  """
  mp_queue.put("finish rsync process")
  rsync_process.join()
  if rsync_process.exitcode == 0:
    logging.info("Artifacts have been uploaded to GCS.")
  else:
    logging.error(
        "GCS rsync process failed with exit code %d.", rsync_process.exitcode
    )


def _rsync_local_to_gcs(local_dir: str, gcs_dir: str) -> None:
  """Syncs the local directory to GCS.

  Args:
    local_dir: The local directory to sync.
    gcs_dir: The GCS directory to sync to.
  """
  if not os.listdir(local_dir):
    logging.info("Not rsyncing to GCS since %s is empty.", local_dir)
    return

  logging.info("Rsyncing %s <--> %s...", local_dir, gcs_dir)
  cmd = [
      "gcloud",
      "storage",
      "rsync",
      "-r",
      "--delete-unmatched-destination-objects",
  ]
  cmd.extend([local_dir, gcs_dir])

  attempt = 0
  while attempt < _GCS_COMMAND_RETRIES:
    try:
      subprocess.check_output(cmd)
      break
    except subprocess.CalledProcessError as e:
      attempt += 1
      if attempt < _GCS_COMMAND_RETRIES:
        logging.exception(
            "Attempt %d: Command failed: %s. Retrying in %d seconds...",
            attempt,
            e,
            _RSYNC_RETRY_INTERVAL_SECS,
        )
        time.sleep(_RSYNC_RETRY_INTERVAL_SECS)
      else:
        logging.exception(
            "Command failed after %d attempts: %s.", e, _GCS_COMMAND_RETRIES
        )

  logging.info("%s rsynced to %s.", local_dir, gcs_dir)


def start_gcs_rsync(
    dirs_to_sync: Sequence[Tuple[str, str]],
    mp_queue: multiprocessing.Queue,
    gcs_rsync_interval_secs: int,
) -> None:
  """Starts a rsync process to sync local directories to GCS directories.

  Args:
    dirs_to_sync: A list of tuples, where each tuple contains local directory
      which will be synced to GCS. For example: [('/tmp/local_dir_1',
      'gs://bucket/gcs_dir_1'), ('/tmp/local_dir_2', 'gs://bucket/gcs_dir_2')]
    mp_queue: The multiprocessing queue to check if the training is finished.
    gcs_rsync_interval_secs: Integer, interval in seconds to run gcs rsync.
  """
  while True:
    for local_dir, gcs_dir in dirs_to_sync:
      _rsync_local_to_gcs(local_dir, gcs_dir)
    if not mp_queue.empty():
      break
    time.sleep(gcs_rsync_interval_secs)

  # Sync up the directory one more time to avoid a race condition.
  # There can be a case when we are doing an rsync and receive a signal that
  # the training has been done. The final checkpoint will be skipped in such
  # case. So we do a final sync to make sure that the all directories
  # are synced.
  for local_dir, gcs_dir in dirs_to_sync:
    _rsync_local_to_gcs(local_dir, gcs_dir)
