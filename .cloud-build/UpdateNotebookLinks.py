
from nbconvert.preprocessors import Preprocessor

import argparse
import nbformat
import os
import re


class UpdateNotebookLinks(Preprocessor):

  def __init__(self, repo_name):
    self._repo_name = repo_name
    self._regex = {
        "colab": "https://colab\\.research\\.google\\.com/github/GoogleCloudPlatform/.*\\.ipynb",
        "github": "https://github\\.com/GoogleCloudPlatform/.*\\.ipynb"
    }
    self._prefix = {
        "colab": "https://colab.research.google.com/github/GoogleCloudPlatform/",
        "github": "https://github.com/GoogleCloudPlatform/"
    }

  def generate_link(self, notebook_file_path, type):
    return self._prefix[type] + self._repo_name + "/blob/master" + notebook_file_path

  def search_link(self, text, type):
    return re.search(self._regex[type], text)

  def replace_link(self, text, notebook_file_path, type):
    return re.sub(self._regex[type], self.generate_link(notebook_file_path, type), text)

  def update_text(self, original_text, notebook_file_path):
    updated_text = self.replace_link(original_text, notebook_file_path, 'colab')
    updated_text = self.replace_link(updated_text, notebook_file_path, 'github')
    return updated_text

  def preprocess(self, notebook, notebook_file_path):
    updated = False
    for cell in notebook.cells:
      if cell.cell_type == 'markdown' and (
          self.search_link(cell.source, 'colab') or self.search_link(cell.source, 'github')):
        cell.source = self.update_text(cell.source, notebook_file_path)
        updated = True
        break
    return notebook, updated

def main(args):

  notebook_file_paths = list()
  for root, dirs, files in os.walk(args.repo_path):
    for file in files:
      if file.endswith(".ipynb"):
        notebook_file_paths.append(os.path.join(root.replace(args.repo_path, ''), file))

  for notebook_file_path in notebook_file_paths:
    with open(args.repo_path + notebook_file_path) as f:
      nb = nbformat.read(f, as_version=4)

    update_notebook_links_preprocessor = UpdateNotebookLinks(repo_name=args.repo_name)
    nb, updated = update_notebook_links_preprocessor.preprocess(nb, notebook_file_path)

    if updated:
      with open(args.repo_path + notebook_file_path, mode="w", encoding="utf-8") as f:
        nbformat.write(nb, f)

  return

parser = argparse.ArgumentParser(description="Run changed notebooks.")

parser.add_argument(
    "--repo-path",
    type=str,
    default=os.getcwd(),
    help="The local path containing the folders of notebooks that should be updated.",
)
parser.add_argument(
    "--repo-name",
    type=str,
    default='vertex-ai-samples',
    help="The repository name used to generate the notebook links.",
)

args = parser.parse_args()

if __name__ == '__main__':
  main(args)