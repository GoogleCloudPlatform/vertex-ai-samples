"""Fileutil lib to copy files between gcs and local."""

import filecmp
import fnmatch
import os
import pathlib
import shutil
import subprocess
import time
from typing import List, Optional, Tuple
import uuid

from absl import logging
from google.cloud import storage

from util import constants


_GCS_CLIENT = None


def _get_gcs_client() -> storage.Client:
  """Gets the default GCS client."""
  global _GCS_CLIENT
  if _GCS_CLIENT is None:
    _GCS_CLIENT = storage.Client()
  return _GCS_CLIENT


def generate_tmp_path(extension: str = '') -> str:
  """Generates a temporary file path with UUID.

  Args:
    extension: File extension, e.g. '.jpg', '.avi'. If not given, no extension
      will be appended to the filename.

  Returns:
    Generated file path.
  """
  return os.path.join(constants.LOCAL_DATA_DIR, uuid.uuid1().hex) + extension


def force_gcs_fuse_path(gcs_uri: str) -> str:
  """Converts gs:// uris to their /gcs/ equivalents. No-op for other uris."""
  if is_gcs_path(gcs_uri):
    return (
        constants.GCSFUSE_URI_PREFIX + gcs_uri[len(constants.GCS_URI_PREFIX) :]
    )
  else:
    return gcs_uri


def force_gcs_path(uri: str) -> str:
  """Converts /gcs/ uris to their gs:// equivalents. No-op for other uris."""
  if uri.startswith(constants.GCSFUSE_URI_PREFIX):
    return uri.replace(
        constants.GCSFUSE_URI_PREFIX, constants.GCS_URI_PREFIX, 1
    )
  else:
    return uri


def is_file_available(
    file_path: str, retry_interval_secs: int = 60, timeout_secs: int = 3600
) -> bool:
  """Checks and waits for a file to be available in GCS.

  Args:
    file_path: The file path to check.
    retry_interval_secs: The interval in seconds to check the file.
    timeout_secs: The timeout in seconds to wait for the file.

  Returns:
    True if the file is available, False otherwise.
  """
  start_time = time.time()
  while True:
    try:
      file_check_cmd = ['gcloud', 'storage', 'ls', file_path]
      result = subprocess.run(
          file_check_cmd, capture_output=True, text=True, check=True
      )
      if file_path in result.stdout:
        logging.info('File %s exists.', file_path)
        return True
    except subprocess.CalledProcessError as e:
      elapsed_time = time.time() - start_time
      if elapsed_time > timeout_secs:
        logging.info(
            "Timeout: File '%s' not found after %d seconds. Error: %s",
            file_path,
            elapsed_time,
            e,
        )
        return False

      logging.info(
          "File '%s' not found yet. Checking again in %d seconds. Error: %s",
          file_path,
          retry_interval_secs,
          e,
      )
      time.sleep(retry_interval_secs)


def compare_dirs(
    local_dir: str,
    gcsfuse_dir: str,
    retry_interval_secs: int = 30,
    timeout_secs: int = 3600,
) -> bool:
  """Compares two directories and returns True if they are the same.

  Args:
    local_dir: The local directory.
    gcsfuse_dir: The gcsfuse directory.
    retry_interval_secs: The interval in seconds to check the directories.
    timeout_secs: The timeout in seconds to wait for the directories.

  Returns:
    True if the directories are the same, False otherwise.
  """
  start_time = time.time()
  while True:
    if os.path.exists(local_dir) and os.path.exists(gcsfuse_dir):
      comparison = filecmp.dircmp(local_dir, gcsfuse_dir)
      if (
          not comparison.left_only
          and not comparison.right_only
          and not comparison.diff_files
      ):
        return True
    elapsed_time = time.time() - start_time
    if elapsed_time > timeout_secs:
      logging.info(
          "Timeout: Directories '%s' and '%s' do not match after %d seconds.",
          local_dir,
          gcsfuse_dir,
          elapsed_time,
      )
      return False

    logging.info(
        "Directories '%s' and '%s' do not match yet. Checking again in %d"
        ' seconds.',
        local_dir,
        gcsfuse_dir,
        retry_interval_secs,
    )
    time.sleep(retry_interval_secs)


def download_gcs_file_to_memory(gcs_uri: str) -> bytes:
  """Downloads a gcs file to in memory.

  Args:
    gcs_uri: A string of GCS uri.

  Returns:
    The content of the gcs file in byte format.
  """
  bucket = gcs_uri.split('/')[2]
  file_path = gcs_uri[len(constants.GCS_URI_PREFIX + bucket + '/') :]
  client = _get_gcs_client()
  bucket = client.bucket(bucket)
  blob = bucket.blob(file_path)
  return blob.download_as_bytes()


def download_gcs_file_to_local_dir(gcs_uri: str, local_dir: str):
  """Download a gcs file to a local dir.

  Args:
    gcs_uri: A string of file path on GCS.
    local_dir: A string of local directory.
  """
  if not is_gcs_path(gcs_uri):
    raise ValueError(
        f'{gcs_uri} is not a GCS path starting with {constants.GCS_URI_PREFIX}.'
    )
  filename = os.path.basename(gcs_uri)
  download_gcs_file_to_local(gcs_uri, os.path.join(local_dir, filename))


def download_gcs_file_to_local(gcs_uri: str, local_path: str):
  """Download a gcs file to a local path.

  Args:
    gcs_uri: A string of file path on GCS.
    local_path: A string of local file path.
  """
  if not is_gcs_path(gcs_uri):
    raise ValueError(
        f'{gcs_uri} is not a GCS path starting with {constants.GCS_URI_PREFIX}.'
    )
  client = _get_gcs_client()
  os.makedirs(os.path.dirname(local_path), exist_ok=True)
  with open(local_path, 'wb') as f:
    client.download_blob_to_file(gcs_uri, f)


def download_gcs_file_list_to_local(
    gcs_uri_list: List[str], local_dir: str
) -> List[str]:
  """Downloads a list of GCS files to a local directory.

  Args:
    gcs_uri_list: A list of GCS file paths.
    local_dir: Local directory in which the GCS files are saved.

  Returns:
    The local file paths corresponding to the input GCS file paths.

  Raises:
    ValueError: An input file path is not a GCS path.
  """
  local_paths = []
  for gcs_uri in gcs_uri_list:
    if not is_gcs_path(gcs_uri):
      raise ValueError(
          f'{gcs_uri} is not a GCS path starting with'
          f' {constants.GCS_URI_PREFIX}.'
      )
    local_path = os.path.join(local_dir, gcs_uri.replace('gs://', ''))
    download_gcs_file_to_local(gcs_uri, local_path)
    local_paths.append(local_path)
  return local_paths


def download_gcs_dir_to_local(
    gcs_dir: str,
    local_dir: str,
    skip_hf_model_bin: bool = False,
    allow_patterns: Optional[List[str]] = None,
    log: bool = True,
) -> None:
  """Downloads files in a GCS directory to a local directory.

  For example:
    download_gcs_dir_to_local(gs://bucket/foo, /tmp/bar)
    gs://bucket/foo/a -> /tmp/bar/a
    gs://bucket/foo/b/c -> /tmp/bar/b/c

  Args:
    gcs_dir: A string of directory path on GCS.
    local_dir: A string of local directory path.
    skip_hf_model_bin: True to skip downloading HF model bin files.
    allow_patterns: A list of allowed patterns. If provided, only files matching
      one or more patterns are downloaded.
    log: True to log each downloaded file.
  """
  if not is_gcs_path(gcs_dir):
    raise ValueError(f'{gcs_dir} is not a GCS path starting with gs://.')
  bucket_name = gcs_dir.split('/')[2]
  prefix = (
      gcs_dir[len(constants.GCS_URI_PREFIX + bucket_name) :].strip('/') + '/'
  )
  client = _get_gcs_client()
  blobs = client.list_blobs(bucket_name, prefix=prefix)
  for blob in blobs:
    if blob.name[-1] == '/':
      continue
    file_path = blob.name[len(prefix) :].strip('/')
    local_file_path = os.path.join(local_dir, file_path)
    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
    if allow_patterns and all(
        [not fnmatch.fnmatch(file_path, p) for p in allow_patterns]
    ):
      continue
    if (
        file_path.endswith(constants.HF_MODEL_WEIGHTS_SUFFIX)
        and skip_hf_model_bin
    ):
      if log:
        logging.info('Skip downloading model bin %s', file_path)
      with open(local_file_path, 'w') as f:
        f.write(f'{constants.GCS_URI_PREFIX}{bucket_name}/{prefix}{file_path}')
    else:
      if log:
        logging.info('Downloading %s to %s', file_path, local_file_path)
      blob.download_to_filename(local_file_path)


def _get_relative_paths(base_dir: str) -> List[str]:
  """Gets relative paths of all files in a local base directory."""
  path = pathlib.Path(base_dir)
  relative_paths = []
  for local_file in path.rglob('*'):
    if os.path.isfile(local_file):
      relative_path = os.path.relpath(local_file, base_dir)
      relative_paths.append(relative_path)
  return relative_paths


def _upload_local_files_to_gcs(
    relative_paths: List[str], local_dir: str, gcs_dir: str
):
  """Uploads local files to gcs."""
  bucket_name = gcs_dir.split('/')[2]
  blob_dir = '/'.join(gcs_dir.split('/')[3:])
  client = _get_gcs_client()
  bucket = client.bucket(bucket_name)
  for relative_path in relative_paths:
    blob = bucket.blob(os.path.join(blob_dir, relative_path))
    blob.upload_from_filename(os.path.join(local_dir, relative_path))


def upload_local_dir_to_gcs(local_dir: str, gcs_dir: str):
  """Uploads local dir to gcs.

  For example:
    upload_local_dir_to_gcs(/tmp/bar, gs://bucket/foo)
    /tmp/bar/a -> gs://bucket/foo/a
    /tmp/bar/b/c -> gs://bucket/foo/b/c

  Arguments:
    local_dir: A string of local directory path.
    gcs_dir: A string of directory path on GCS.
  """
  # Relative paths of all files in local_dir.
  relative_paths = _get_relative_paths(local_dir)
  _upload_local_files_to_gcs(relative_paths, local_dir, gcs_dir)


def upload_file_to_gcs_path(
    source_path: str,
    destination_uri: str,
):
  """Uploads local files to GCS uri.

  After upload the destination_uri will contain the same data as the
  source_path.

  Args:
      source_path: Required. Path of the local data to copy to GCS.
      destination_uri: Required. GCS URI where the data should be uploaded.

  Raises:
      RuntimeError: When source_path does not exist.
      GoogleCloudError: When the upload process fails.
  """
  source_path_obj = pathlib.Path(source_path)
  if not source_path_obj.exists():
    raise RuntimeError(f'Source path does not exist: {source_path}')

  storage_client = _get_gcs_client()
  source_file_path = source_path
  destination_file_uri = destination_uri
  logging.info('Uploading "%s" to "%s"', source_file_path, destination_file_uri)
  destination_blob = storage.Blob.from_string(
      destination_file_uri, client=storage_client
  )
  destination_blob.upload_from_filename(filename=source_file_path)


def is_gcs_path(input_path: str) -> bool:
  """Checks if the input path is a Google Cloud Storage (GCS) path.

  Args:
      input_path: The input path to be checked.

  Returns:
      True if the input path is a GCS path, False otherwise.
  """
  return input_path is not None and input_path.startswith(
      constants.GCS_URI_PREFIX
  )


def release_text_assets(
    output_bucket: str, local_text_file_name: str, remote_text_file_name: str
) -> None:
  """Releases text assets.

  Args:
    output_bucket: gcs output bucket.
    local_text_file_name: Local text file name.
    remote_text_file_name: Remote text file name.

  Returns:
    None
  """
  remote_file_path = '{}/{}'.format(output_bucket, remote_text_file_name)
  logging.info('Uploading "%s" to "%s"', local_text_file_name, remote_file_path)
  upload_file_to_gcs_path(local_text_file_name, remote_file_path)
  os.remove(local_text_file_name)


def upload_video_from_local_to_gcs(
    output_bucket: str,
    local_video_file_name: str,
    remote_video_file_name: str,
    temp_local_video_file_name: str,
) -> None:
  """Uploads video from local to gcs buckent and releases video assets.

  Args:
    output_bucket: GCS bucket address.
    local_video_file_name: Local video file name.
    remote_video_file_name: Remote video file name.
    temp_local_video_file_name: Temporary local video file name.

  Returns:
    None
  """
  upload_file_to_gcs_path(
      temp_local_video_file_name,
      '{}/{}'.format(output_bucket, remote_video_file_name),
  )
  shutil.rmtree(local_video_file_name, ignore_errors=True)
  shutil.rmtree(temp_local_video_file_name, ignore_errors=True)


def download_video_from_gcs_to_local(video_file_path: str) -> Tuple[str, str]:
  """Downloads video from gcs to local folders.

  Args:
    video_file_path: Path to the video file.

  Returns:
    Local and remote video file paths.
  """
  _, local_video_file_name = os.path.split(video_file_path)
  file_extension = os.path.splitext(video_file_path)[1]
  remote_video_file_name = local_video_file_name.replace(
      file_extension, '_overlay.mp4'
  )
  local_file_path = generate_tmp_path(os.path.splitext(video_file_path)[1])
  logging.info('Downloading %s to %s...', video_file_path, local_file_path)
  download_gcs_file_to_local(video_file_path, local_file_path)
  return local_file_path, remote_video_file_name


def get_output_video_file(video_output_file_path: str) -> str:
  """Gets the output video file name for writing video.

  Args:
    video_output_file_path: Path to the video output file.

  Returns:
    str: Local video output file path.
  """
  file_extension = os.path.splitext(video_output_file_path)[1]
  out_local_video_file_name = video_output_file_path.replace(
      file_extension, '_overlay' + file_extension
  )
  return out_local_video_file_name


def delete_local_file(local_file_path: str) -> None:
  """Deletes a local file."""
  if os.path.exists(local_file_path):
    os.remove(local_file_path)


def delete_local_dir(local_dir: str) -> None:
  """Deletes a local directory recursively."""
  if os.path.exists(local_dir):
    shutil.rmtree(local_dir)
