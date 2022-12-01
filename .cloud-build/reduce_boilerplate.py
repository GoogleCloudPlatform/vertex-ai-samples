import abc
from typing import Dict, List, Optional

# Based on https://app.reviewnb.com/GoogleCloudPlatform/vertex-ai-samples/pull/1217/

TEMPLATE_CELLS: Dict[str, List[str]] = {}

class Fix(abc.ABC):
    def match(self, cell_source: str, next_cells: List[str]) -> bool:
        pass

    def replace(self, cell_source: str) -> Optional[List[str]]:
        pass

class SetupRemoval(Fix):
    def match(self, cell_source: str, next_cells: List[str]) -> bool:
        return cell_source.contains("Set up your local development environment")

    def replace(self, cell_source: str) -> Optional[List[str]]:
        return None

class RestartKernalReplacement(Fix):
    def match(self, cell_source: str, next_cells: List[str]) -> bool:
        return cell_source.contains("Restart the kernel") and 

    def replace(self, cell_source: str) -> Optional[List[str]]:
        return TEMPLATE_CELLS["RestartKernelForColab"]

# Find all notebooks


def process_notebook(path: str):
    for source_
    if cell_source
    
notebooks = []

for notebook in notebooks:
