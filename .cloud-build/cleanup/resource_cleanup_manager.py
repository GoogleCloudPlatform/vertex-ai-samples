'''
READ FIRST BEFORE MAKING CHANGES
-  Create a convention for resources created from vertex-ai-samples GH. We already have one IIRC
- Only delete those objects as part of our clean-up script.
- Don't run any tests on python-docs-samples-tests project, especially ones that affect resources created outside of our purview
- Add --dry-run option to the clean-up script. This option will just output the list of resources the script will delete instead of actually deleting the resources.
- Have a larger conversation in DEE before touching any resources that were not created as part of vertex-ai-samples
'''
import os
import abc
from typing import Any, Type

from google.cloud import aiplatform
from google.cloud.aiplatform import base
from google.cloud.aiplatform_v1beta1 import (FeatureOnlineStoreAdminServiceClient,
                                             FeatureOnlineStore)
from google.cloud import storage
from proto.datetime_helpers import DatetimeWithNanoseconds

PROJECT_ID = "python-docs-samples-tests"
REGION = "us-central1"
API_ENDPOINT = f"{REGION}-aiplatform.googleapis.com"

# If a resource was updated within this number of seconds, do not delete.
RESOURCE_UPDATE_BUFFER_IN_SECONDS = 60 * 60 * 8


class ResourceCleanupManager(abc.ABC):
    @property
    @abc.abstractmethod
    def type_name(str) -> str:
        pass

    @abc.abstractmethod
    def list(self) -> Any:
        pass

    @abc.abstractmethod
    def resource_name(self, resource: Any) -> str:
        pass

    @abc.abstractmethod
    def delete(self, resource: Any):
        pass

    @abc.abstractmethod
    def get_seconds_since_modification(self, resource: Any) -> float:
        pass

    def is_deletable(self, resource: Any) -> bool:
        time_difference = self.get_seconds_since_modification(resource)

        if self.resource_name(resource).startswith("perm"):
            print(f"Skipping '{resource}' due to name starting with 'perm'.")
            return False

        # Check that it wasn't created too recently, to prevent race conditions
        if time_difference <= RESOURCE_UPDATE_BUFFER_IN_SECONDS:
            print(
                f"Skipping '{resource}' due to update_time being '{time_difference}', which is less than '{RESOURCE_UPDATE_BUFFER_IN_SECONDS}'."
            )
            return False

        return True


class VertexAIResourceCleanupManager(ResourceCleanupManager):
    @property
    @abc.abstractmethod
    def vertex_ai_resource(self) -> Type[base.VertexAiResourceNounWithFutureManager]:
        pass

    @property
    def type_name(self) -> str:
        return self.vertex_ai_resource._resource_noun

    def list(self) -> Any:
        return self.vertex_ai_resource.list()

    def resource_name(
        self, resource: Type[base.VertexAiResourceNounWithFutureManager]
    ) -> str:
        return resource.display_name

    def delete(self, resource):
        resource.delete()

    def get_seconds_since_modification(self, resource: Any) -> float:
        update_time = resource.update_time
        current_time = DatetimeWithNanoseconds.now(tz=update_time.tzinfo)
        return (current_time - update_time).total_seconds()


class DatasetResourceCleanupManager(VertexAIResourceCleanupManager):
    vertex_ai_resource = aiplatform.datasets._Dataset
    dataset_types = [
        aiplatform.ImageDataset,
        aiplatform.TabularDataset,
        aiplatform.TextDataset,
        aiplatform.TimeSeriesDataset,
        aiplatform.VideoDataset,
    ]

    def list(self) -> Any:
        return [
            dataset
            for dataset_type in self.dataset_types
            for dataset in dataset_type.list()
        ]


class EndpointResourceCleanupManager(VertexAIResourceCleanupManager):
    vertex_ai_resource = aiplatform.Endpoint

    def delete(self, resource):
        for deployed_model_id in [
            models.id for models in resource._gca_resource.deployed_models
        ]:
            resource._undeploy(deployed_model_id=deployed_model_id)
        resource.delete(force=True)


class ModelResourceCleanupManager(VertexAIResourceCleanupManager):
    vertex_ai_resource = aiplatform.Model


class MatchingEngineIndexResourceCleanupManager(VertexAIResourceCleanupManager):
    vertex_ai_resource = aiplatform.MatchingEngineIndex


class MatchingEngineIndexEndpointResourceCleanupManager(VertexAIResourceCleanupManager):
    vertex_ai_resource = aiplatform.MatchingEngineIndexEndpoint

    def delete(self, resource):
        resource.undeploy_all()
        resource.delete(force=True)

class FeatureStoreLegacyCleanupManager(VertexAIResourceCleanupManager):
    # TODO: only deleting legacy
    #    not deleting ingestions jobs
    #       ingest_from_xxx methods do not return a job ID, there is no list command, aka no python way to delete
    #    not deleting batch serving jobs
    #       batch_serve_to_xxx methods do not return a job ID, there is no list command, aka no python way to delete
    vertex_ai_resource = aiplatform.Featurestore

    def resource_name(self, resource: Any) -> str:
        return resource.name

    def delete(self, resource):
        resource.delete(force=True)


class FeatureStoreCleanupManager(VertexAIResourceCleanupManager):
    # for FS 2.0
    # TODO: use _v1beta1, and gapic clients
    #    delete features, feature groups, feature views, feature online stores
    vertex_ai_resource = FeatureOnlineStore

    admin_client = FeatureOnlineStoreAdminServiceClient(
        client_options={"api_endpoint": API_ENDPOINT}
    )

    def resource_name(self, resource: Any) -> str:
        return resource.name

    def type_name(self) -> str:
        return "FeatureOnlineStore"

    def list(self) -> Any:
        try:
            return self.admin_client.list_feature_online_stores(parent=f"projects/{PROJECT_ID}/locations/{REGION}")
        except Exception as e:
            print(e)
            return []

    def delete(self, resource):
        try:
            self.admin_client.delete_feature_online_store(name=resource.name, force=True)
        except Exception as e:
            print(e)


class PipelineJobCleanupManager(VertexAIResourceCleanupManager):
    vertex_ai_resource = aiplatform.PipelineJob

class TrainingJobCleanupManager(VertexAIResourceCleanupManager):
    vertex_ai_resource = aiplatform.training_jobs._CustomTrainingJob

    job_types = [
            aiplatform.AutoMLImageTrainingJob,
            aiplatform.AutoMLTextTrainingJob,
            aiplatform.AutoMLTabularTrainingJob,
            aiplatform.AutoMLVideoTrainingJob,
            aiplatform.AutoMLForecastingTrainingJob,
            aiplatform.CustomJob,
            aiplatform.CustomTrainingJob,
            aiplatform.CustomContainerTrainingJob,
            aiplatform.CustomPythonPackageTrainingJob
    ]

    def list(self) -> Any:
        return [
            job
            for job_type in self.job_types
            for job in job_type.list()
        ]

class HyperparameterTuningCleanupManager(VertexAIResourceCleanupManager):
    vertex_ai_resource = aiplatform.HyperparameterTuningJob


class BatchPredictionJobCleanupManager(VertexAIResourceCleanupManager):
    vertex_ai_resource = aiplatform.BatchPredictionJob

class ExperimentCleanupManager(VertexAIResourceCleanupManager):
    vertex_ai_resource = aiplatform.Experiment

    @property
    def type_name(self) -> str:
        return "Experiment"

    def resource_name(self, resource: Any) -> str:
        return resource.name

    def get_seconds_since_modification(self, resource: Any) -> float:
        update_time = resource._metadata_context.update_time
        current_time = DatetimeWithNanoseconds.now()
        return float(current_time.timestamp() - update_time.timestamp())

class BucketCleanupManager(ResourceCleanupManager):
    vertex_ai_resource = storage.bucket.Bucket

    def list(self) -> Any:
        storage_client = storage.Client()
        return list(storage_client.list_buckets())

    def delete(self, resource):
        try:
            resource.delete(force=True)
        except Exception as e:
            print(e)

    @property
    def type_name(self) -> str:
        return "Bucket"

    def get_seconds_since_modification(self, resource: Any) -> float:
        # Bucket has no last_update property, only time created
        created_time = resource.time_created
        current_time = DatetimeWithNanoseconds.now()
        return float(current_time.timestamp() - created_time.timestamp())

    def resource_name(self, resource: Any) -> str:
        return resource.name

    def is_deletable(self, resource: Any) -> bool:
        time_difference = self.get_seconds_since_modification(resource)

        if not self.resource_name(resource).startswith('your-bucket-name'):
            print(f"Skipping '{resource}' not a Vertex AI notebook bucket")
            return False

        # Check that it wasn't created too recently, to prevent race conditions
        if time_difference <= RESOURCE_UPDATE_BUFFER_IN_SECONDS:
            print(
                f"Skipping '{resource}' due to update_time being '{time_difference}', which is less than '{RESOURCE_UPDATE_BUFFER_IN_SECONDS}'."
            )
            return False
        return True

class ArtifactRegistryCleanupManager(ResourceCleanupManager):
    vertex_ai_resource = "Artifact Registry"

    def list(self) -> Any:
        import subprocess

        result = subprocess.run(["gcloud artifacts repositories list --location=us-central1"], 
                                shell=True, capture_output=True, text=True)

        ret = []
        lines = result.stdout.split('\n')[2:]
        for line in lines:
            repo = line.split(' ')[0]
            if repo.startswith("my-docker-repo"):
                ret.append(repo)

        return ret

    def delete(self, resource):
        os.system(f"! gcloud artifacts repositories delete {resource} --location=us-central1")

    @property
    def type_name(self) -> str:
        return "ArtifactRepository"

    def resource_name(self, resource: Any) -> str:
        return resource

    # delete repository regardless of age
    def get_seconds_since_modification(self, resource: Any) -> float:
        return RESOURCE_UPDATE_BUFFER_IN_SECONDS + 1
    
    def is_deleteable(self, resource: Any) -> bool:
        return True

