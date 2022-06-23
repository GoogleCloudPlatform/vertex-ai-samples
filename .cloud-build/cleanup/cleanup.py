from typing import List
from ratemate import RateLimit
from resource_cleanup_manager import (
    DatasetResourceCleanupManager,
    ModelResourceCleanupManager,
    EndpointResourceCleanupManager,
    ResourceCleanupManager,
)

rate_limit = RateLimit(max_count=25, per=60, greedy=False)


def run_cleanup_managers(managers: List[ResourceCleanupManager], is_dry_run: bool):
    for manager in managers:
        type_name = manager.type_name

        print(f"Fetching {type_name}'s...")
        resources = manager.list()
        print(f"Found {len(resources)} {type_name}'s")
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


is_dry_run = False

if is_dry_run:
    print("Starting cleanup in dry run mode...")

# List of all cleanup managers
managers = [
    DatasetResourceCleanupManager(),
    EndpointResourceCleanupManager(),
    ModelResourceCleanupManager(),  # ModelResourceCleanupManager must follow EndpointResourceCleanupManager due to deployed models blocking model deletion.
]

run_cleanup_managers(managers=managers, is_dry_run=is_dry_run)
