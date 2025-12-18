"""Utility functions for interacting with Google Cloud Platform."""

import datetime
import logging
import os
import subprocess
import uuid

from google.cloud import aiplatform
import requests


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_project_id() -> str:
  """Read cloud project id from metadata service."""
  project_request = requests.get(
      "http://metadata.google.internal/computeMetadata/v1/project/project-id",
      headers={"Metadata-Flavor": "Google"},
  )
  return project_request.text


def get_region() -> str:
  """Read region from metadata service."""
  region_request = requests.get(
      "http://metadata.google.internal/computeMetadata/v1/instance/region",
      headers={"Metadata-Flavor": "Google"},
  )
  return region_request.text.split("/")[-1]


# Get the default cloud project id and region
PROJECT_ID = get_project_id()
REGION = get_region()


def init_aiplatform(project: str = None, location: str = None) -> None:
  """Initialize the Vertex AI SDK.

  Args:
      project: The Google Cloud project ID.
      location: The Google Cloud location.
  """
  project = PROJECT_ID if project is None else project
  location = REGION if location is None else location
  aiplatform.init(project=project, location=location)
  subprocess.call([
      "gcloud",
      "services",
      "enable",
      "aiplatform.googleapis.com",
      "compute.googleapis.com",
  ])


def run_command(command: list[str]) -> str:
  """Runs a shell command and returns the output.

  Args:
      command: The shell command to run as a list.

  Returns:
      The output of the command.
  """
  try:
    result = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout
  except subprocess.CalledProcessError as e:
    logger.error("Error: %s", e.stderr)
    raise e


def enable_apis() -> None:
  """Enable the Vertex AI API and Compute Engine API."""
  logger.info("Enabling Vertex AI API and Compute Engine API.")
  run_command([
      "gcloud",
      "services",
      "enable",
      "aiplatform.googleapis.com",
      "compute.googleapis.com",
  ])


def setup_buckets(bucket_uri: str, model_bucket_name: str) -> tuple[str, str]:
  """Set up Cloud Storage buckets for storing experiment artifacts.

  Args:
      bucket_uri: The bucket URI provided by the user.
      model_bucket_name: The name of the model bucket.

  Returns:
      A tuple containing the bucket name and model bucket path.
  """
  if not bucket_uri.strip():
    # Generate a default bucket URI if none provided
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    bucket_uri = f"gs://{PROJECT_ID}-tmp-{now}-{str(uuid.uuid4())[:4]}"
    logger.info("No bucket URI provided. Using default bucket: %s", bucket_uri)
  else:
    if not bucket_uri.startswith("gs://"):
      raise ValueError("Bucket URI must start with 'gs://'.")
    # Remove any trailing slashes
    bucket_uri = bucket_uri.rstrip("/")

  bucket_name = "/".join(bucket_uri.split("/")[:3])

  # Check if bucket exists
  try:
    run_command(["gcloud", "storage", "ls", "--buckets", bucket_uri])
    logger.info("Bucket %s already exists.", bucket_uri)
  except subprocess.CalledProcessError:
    logger.info("Creating bucket %s.", bucket_uri)
    # Create the bucket in the same region as the project
    run_command(["gcloud", "storage", "buckets", "create", "--location", REGION, bucket_uri])

  # Construct the model bucket path
  model_bucket = os.path.join(bucket_uri, model_bucket_name)

  # Check if the model bucket exists (as a folder within the main bucket)
  try:
    run_command(["gcloud", "storage", "ls", model_bucket])
    logger.info("Model bucket %s already exists.", model_bucket)
  except subprocess.CalledProcessError:
    logger.info("Creating model bucket %s.", model_bucket)
    # Create the model bucket folder
    run_command(["gcloud", "storage", "cp", "/dev/null", model_bucket + "/"])

  return bucket_name, model_bucket


def get_service_account() -> str:
  """Get the default service account."""
  shell_output = run_command(["gcloud", "projects", "describe", PROJECT_ID])
  project_number_line = next(
      (line for line in shell_output.splitlines() if "projectNumber" in line),
      None,
  )
  if project_number_line:
    project_number = project_number_line.split(":")[1].strip().replace("'", "")
    service_account = f"{project_number}-compute@developer.gserviceaccount.com"
    logger.info("Using default Service Account: %s", service_account)
    return service_account
  else:
    raise ValueError("Could not find project number in gcloud output.")


def get_project_number() -> str:
  """Get the default project number."""
  shell_output = run_command(["gcloud", "projects", "describe", PROJECT_ID])
  project_number_line = next(
      (line for line in shell_output.splitlines() if "projectNumber" in line),
      None,
  )
  if project_number_line:
    project_number = project_number_line.split(":")[1].strip().replace("'", "")
    logger.info("Using default Project Number: %s", project_number)
    return project_number
  else:
    raise ValueError("Could not find project number in gcloud output.")


def provision_permissions(service_account: str, bucket_name: str) -> None:
  """Provision permissions to the service account with the GCS bucket."""
  if bucket_name:
    run_command([
        "gcloud",
        "storage",
        "buckets",
        "add-iam-policy-binding",
        bucket_name,
        f"--member=serviceAccount:{service_account}",
        "--role=roles/storage.admin",
    ])


def set_gcloud_project() -> None:
  """Set gcloud config project."""
  run_command(["gcloud", "config", "set", "project", PROJECT_ID])


def initialize(
    bucket_uri: str, model_bucket_name: str, create_bucket: bool
) -> tuple[str, str]:
  """Initialize the environment.

  Args:
      bucket_uri: The bucket URI provided by the user.
      model_bucket_name: The name of the model bucket.
      create_bucket: Whether to create the bucket or not.

  Returns:
      A tuple containing the model bucket path and service account.
  """
  enable_apis()
  bucket_name = None
  if create_bucket:
    bucket_name, model_bucket = setup_buckets(bucket_uri, model_bucket_name)
  else:
    model_bucket = None
  service_account = get_service_account()
  provision_permissions(service_account, bucket_name)
  set_gcloud_project()
  return model_bucket, service_account


def clean_resources_ui(
    project_id: str,
    region: str,
    endpoint_name: str,
    delete_bucket: bool,
    bucket_name: str = None,
) -> str:
  """UI function for cleaning a specific Vertex AI endpoint and its model."""
  if delete_bucket and not bucket_name:
    raise ValueError("Bucket name is required when 'Delete Bucket' is checked.")

  try:
    delete_endpoint_and_model(project_id, region, endpoint_name)
    bucket_status_message = ""
    if delete_bucket:
      bucket_status_message = delete_gcs_bucket(bucket_name)
    if endpoint_name:
      return (
          f"Endpoint {endpoint_name} and associated model deleted successfully!"
          f" {bucket_status_message}"
      )
    else:
      return (
          "There are currently no endpoints available for deletion."
          f" {bucket_status_message}"
      )
  except Exception as e:  # pylint: disable=broad-exception-caught
    return f"Error cleaning up resources: {e}"


def delete_endpoint_and_model(
    project_id: str, region: str, endpoint_name: str
) -> None:
  """Deletes a specific Vertex AI endpoint and its associated model."""
  if endpoint_name:
    endpoint_id = endpoint_name.split(" - ")[0]
    endpoint_resource_name = (
        f"projects/{project_id}/locations/{region}/endpoints/{endpoint_id}"
    )
    endpoint = aiplatform.Endpoint(
        endpoint_resource_name, project=project_id, location=region
    )
    deployed_models = endpoint.list_models()
    for deployed_model in deployed_models:
      endpoint.undeploy(deployed_model_id=deployed_model.id)
      model = aiplatform.Model(deployed_model.model)
      model.delete()
    endpoint.delete()


def delete_gcs_bucket(bucket_name: str) -> str:
  """Deletes a GCS bucket using gcloud storage."""
  try:
    run_command(["gcloud", "storage", "rm", "--recursive", bucket_name])
    logger.info("Bucket %s deleted using gcloud storage.", bucket_name)
    return f"Bucket {bucket_name} deleted successfully!"
  except subprocess.CalledProcessError as e:
    logger.error(
        "Error deleting bucket %s using gcloud storage: %s", bucket_name, str(e)
    )
    return f"Bucket {bucket_name} could not be found or deleted. "
