from nbconvert.preprocessors import Preprocessor
from typing import Dict
import UpdateNotebookVariables


class RemoveNoExecuteCells(Preprocessor):
    def preprocess(self, notebook, resources=None):
        executable_cells = []
        for cell in notebook.cells:
            if cell.metadata.get("tags"):
                if "no_execute" in cell.metadata.get("tags"):
                    continue
            executable_cells.append(cell)
        notebook.cells = executable_cells
        return notebook, resources


class UpdateVariablesPreprocessor(Preprocessor):
    def __init__(self, replacement_map: Dict):
        self._replacement_map = replacement_map

    @staticmethod
    def update_variables(content: str, replacement_map: Dict[str, str]):
        # replace variables inside .ipynb files
        # looking for this format inside notebooks:
        # VARIABLE_NAME = '[description]'

        for variable_name, variable_value in replacement_map.items():
            content = UpdateNotebookVariables.get_updated_value(
                content=content,
                variable_name=variable_name,
                variable_value=variable_value,
            )

        return content

    def preprocess(self, notebook, resources=None):
        executable_cells = []
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.source = self.update_variables(
                    content=cell.source,
                    replacement_map=self._replacement_map,
                )

            executable_cells.append(cell)
        notebook.cells = executable_cells
        return notebook, resources