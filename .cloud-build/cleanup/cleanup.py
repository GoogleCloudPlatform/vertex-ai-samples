from typing import List
from ratemate import RateLimit
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dry_run",
                    type=bool,
                    default=False)
args = parser.parse_args()


from resource_cleanup_manager import (
    DatasetResourceCleanupManager,
    ModelResourceCleanupManager,
    EndpointResourceCleanupManager,
    ResourceCleanupManager,
    MatchingEngineIndexEndpointResourceCleanupManager,
    MatchingEngineIndexResourceCleanupManager,
    FeatureStoreLegacyCleanupManager,
    FeatureStoreCleanupManager,
    PipelineJobCleanupManager,
    TrainingJobCleanupManager,
    HyperparameterTuningCleanupManager,
    BatchPredictionJobCleanupManager,
    ExperimentCleanupManager,
    BucketCleanupManager,
    ArtifactRegistryCleanupManager
)

rate_limit = RateLimit(max_count=25, per=60, greedy=False)


def run_cleanup_managers(managers: List[ResourceCleanupManager], is_dry_run: bool):
    for manager in managers:
        type_name = manager.type_name

        print(f"Fetching {type_name}'s...")
        resources = manager.list()
        try:
            print(f"Found {len(resources)} {type_name}'s")
        except Exception as e:
            print(f"{type_name} {e}")
        for resource in resources:
            try:
                if not manager.is_deletable(resource):
                    continue
                if is_dry_run:
                    resource_name = manager.resource_name(resource)
                    print(f"Will delete '{type_name}': {resource_name}")
                else:
                    rate_limit.wait()  # wait before deleting
                    manager.delete(resource)
            except Exception as exception:
                print(exception)

        print("")


if args.dry_run:
    print("Starting cleanup in dry run mode...")

# List of all cleanup managers
managers: List[ResourceCleanupManager] = [
    DatasetResourceCleanupManager(),
    EndpointResourceCleanupManager(),
    ModelResourceCleanupManager(),  # ModelResourceCleanupManager must follow EndpointResourceCleanupManager due to deployed models blocking model deletion.
    MatchingEngineIndexEndpointResourceCleanupManager(),
    MatchingEngineIndexResourceCleanupManager(),
    FeatureStoreLegacyCleanupManager(),
    FeatureStoreCleanupManager(),
    PipelineJobCleanupManager(),
    TrainingJobCleanupManager(),
    HyperparameterTuningCleanupManager(),
    BatchPredictionJobCleanupManager(),
    ExperimentCleanupManager(), # Experiment missing _resource_noun
    BucketCleanupManager(),
    ArtifactRegistryCleanupManager()
]

run_cleanup_managers(managers=managers, is_dry_run=args.dry_run)
