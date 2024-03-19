"""Fileutil lib to copy files between gcs and local."""

import glob
import os
import pathlib
import shutil
from typing import Tuple
import uuid

from absl import logging
from google.cloud import storage

from util import constants


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
  client = storage.Client()
  os.makedirs(os.path.dirname(local_path), exist_ok=True)
  with open(local_path, 'wb') as f:
    client.download_blob_to_file(gcs_uri, f)


def download_gcs_dir_to_local(
    gcs_dir: str, local_dir: str, skip_hf_model_bin: bool = False
):
  """Downloads files in a GCS directory to a local directory.

  For example:
    download_gcs_dir_to_local(gs://bucket/foo, /tmp/bar)
    gs://bucket/foo/a -> /tmp/bar/a
    gs://bucket/foo/b/c -> /tmp/bar/b/c

  Arguments:
    gcs_dir: A string of directory path on GCS.
    local_dir: A string of local directory path.
    skip_hf_model_bin: True to skip downloading HF model bin files.
  """
  if not is_gcs_path(gcs_dir):
    raise ValueError(f'{gcs_dir} is not a GCS path starting with gs://.')
  bucket_name = gcs_dir.split('/')[2]
  prefix = gcs_dir[len(constants.GCS_URI_PREFIX + bucket_name) :].strip('/')
  client = storage.Client()
  blobs = client.list_blobs(bucket_name, prefix=prefix)
  for blob in blobs:
    if blob.name[-1] == '/':
      continue
    file_path = blob.name[len(prefix) :].strip('/')
    local_file_path = os.path.join(local_dir, file_path)
    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
    if (
        file_path.endswith(constants.HF_MODEL_WEIGHTS_SUFFIX)
        and skip_hf_model_bin
    ):
      logging.info('Skip downloading model bin %s', file_path)
      with open(local_file_path, 'w') as f:
        f.write(f'{constants.GCS_URI_PREFIX}{bucket_name}/{prefix}/{file_path}')
    else:
      logging.info('Downloading %s to %s', file_path, local_file_path)
      blob.download_to_filename(local_file_path)


def upload_local_dir_to_gcs(local_dir: str, gcs_dir: str):
  """Uploads local dir to gcs.

  For example:
    upload_local_dir_to_gcs(/tmp/bar, gs://bucket/foo)
    gs://bucket/foo/a -> /tmp/bar/a
    gs://bucket/foo/b/c -> /tmp/bar/b/c

  Arguments:
    local_dir: A string of local directory path.
    gcs_dir: A string of directory path on GCS.
  """
  bucket_name = gcs_dir.split('/')[2]
  blob_dir = '/'.join(gcs_dir.split('/')[3:])
  client = storage.Client()
  bucket = client.bucket(bucket_name)
  for local_file in glob.glob(local_dir + '/**'):
    if os.path.isfile(local_file):
      logging.info(
          'Uploading %s to %s',
          local_file,
          os.path.join(constants.GCS_URI_PREFIX, bucket_name, blob_dir),
      )
      blob = bucket.blob(os.path.join(blob_dir, os.path.basename(local_file)))
      blob.upload_from_filename(local_file)


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

  storage_client = storage.Client()
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
  return input_path.startswith(constants.GCS_URI_PREFIX)


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
  if file_extension:
    remote_video_file_name = local_video_file_name.replace(
        file_extension, '_overlay.mp4'
    )
  else:
    remote_video_file_name = local_video_file_name + '_overlay.mp4'
  local_file_path = generate_tmp_path(file_extension)
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
  if file_extension:
    out_local_video_file_name = video_output_file_path.replace(
        file_extension, '_overlay' + file_extension
    )
  else:
    out_local_video_file_name = video_output_file_path + '_overlay'
  return out_local_video_file_name
