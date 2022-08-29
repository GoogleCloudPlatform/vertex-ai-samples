import abc
from typing import Any, Type

from google.cloud import aiplatform
from google.cloud.aiplatform import base
from proto.datetime_helpers import DatetimeWithNanoseconds

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

    def get_seconds_since_modification(self, resource: Any) -> bool:
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
        # TODO: Remove this once https://github.com/googleapis/python-aiplatform/issues/1441 is fixed
        resource._sync_gca_resource()
        for deployed_model_id in [
            models.id for models in resource._gca_resource.deployed_models
        ]:
            resource._undeploy(deployed_model_id=deployed_model_id)

        resource.delete(force=True)


class ModelResourceCleanupManager(VertexAIResourceCleanupManager):
    vertex_ai_resource = aiplatform.Model
