"""Sync local directory to GCS directory using rsync."""

from collections.abc import Sequence
import multiprocessing
import os
import subprocess
import time

from absl import logging

_GCS_COMMAND_RETRIES = 3
_RSYNC_RETRY_INTERVAL_SECS = 30


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
    dirs_to_sync: Sequence[tuple[str, str]],
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
