"""Fileutil lib to copy files between gcs and local."""

import glob
import os

from absl import logging
from google.cloud import storage

from google3.cloud.ml.applications.vision.model_garden.model_oss.util import constants


def download_gcs_file_to_local(gcs_uri: str, local_path: str):
  """Download a gcs file to a local path.

  Args:
    gcs_uri: A string of file path on GCS.
    local_path: A string of local file path.
  """
  if not gcs_uri.startswith(constants.GCS_URI_PREFIX):
    raise ValueError(
        f'{gcs_uri} is not a GCS path starting with {constants.GCS_URI_PREFIX}.'
    )
  client = storage.Client()
  os.makedirs(os.path.dirname(local_path), exist_ok=True)
  with open(local_path, 'wb') as f:
    client.download_blob_to_file(gcs_uri, f)


def download_gcs_dir_to_local(gcs_dir: str, local_dir: str):
  """Downloads files in a GCS directory to a local directory.

  For example:
    download_gcs_dir_to_local(gs://bucket/foo, /tmp/bar)
    gs://bucket/foo/a -> /tmp/bar/a
    gs://bucket/foo/b/c -> /tmp/bar/b/c

  Arguments:
    gcs_dir: A string of directory path on GCS.
    local_dir: A string of local directory path.
  """
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
