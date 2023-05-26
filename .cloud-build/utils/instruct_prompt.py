"""
"""

import argparse
import json
import os
import sys
import urllib.request
import csv
from enum import Enum
from abc import ABC, abstractmethod
from typing import List


parser = argparse.ArgumentParser()
parser.add_argument('--notebook-dir', 
                    dest='notebook_dir',
                    default=None, 
                    type=str, 
                    help='Notebook directory')
parser.add_argument('--notebook', 
                    dest='notebook',
                    default=None, 
                    type=str, 
                    help='Notebook to extract prompts')
parser.add_argument('--notebook-file', 
                    dest='notebook_file',
                    default=None, 
                    type=str, 
                    help='File with list of notebooks to extract prompts')
parser.add_argument('--debug',
                    dest='debug',
                    default=False,
                    action='store_true', 
                    help='Debugging mode')
args = parser.parse_args()



class Notebook(object):
    '''
    Class for navigating through a notebook
    '''
    def __init__(self, path):
        """
        Initializer
            path: The path to the notebook
        """
        self._path = path
        
        with open(self._path, 'r') as f:
            try:
                self._content = json.load(f)
            except:
                print("Corrupted notebook:", path)
                return

        self._cells = self._content['cells']
        self._cell_index = 0
        self._ncells = len(self._cells)
        
    def get(self) -> list:
        '''
        Get the next cell in the notebook
        
        Returns the current cell
        '''
        cell = self._cells[self._cell_index]
        self._cell_index += 1
        return cell

    
    def peek(self) -> list:
        '''
        Peek at the next cell in the notebook
        
        Returns the current cell
        '''
        cell = self._cells[self._cell_index]
        return cell
    

    def pop(self, n_cells=1):
        '''
        Advance the specified number of cells
        
            n_cells: The number of cells to advance
        '''
        self._cell_index += n_cells
        
    def eon(self) -> bool:
        '''
        End of notebook cells
        '''
        if self._cell_index >= self._ncells:
            return True
        return False
    

    @property
    def path(self):
        '''
        Getter: return the filename path for the notebook
        '''
        return self._path


def parse_dir(directory: str) -> int:
    """
        Recursively walk the specified directory, reviewing each notebook (.ipynb) encountered.
        
            directory: The directory path.
            
        Return TBD
    """
    exit_code = 0

    entries = os.scandir(directory)
    for entry in entries:
        if entry.is_dir():
            if entry.name[0] == '.':
                continue
            if entry.name == 'src' or entry.name == 'images' or entry.name == 'sample_data':
                continue
            exit_code += parse_dir(entry.path)
        elif entry.name.endswith('.ipynb'):
            tag = directory.split('/')[-1]
            if tag == 'automl':
                tag = 'AutoML'
            elif tag == 'bigquery_ml':
                tag = 'BigQuery ML'
            elif tag == 'custom':
                tag = 'Vertex AI Training'
            elif tag == 'experiments':
                tag = 'Vertex AI Experiments'
            elif tag == 'explainable_ai':
                tag = 'Vertex Explainable AI'
            elif tag == 'feature_store':
                tag = 'Vertex AI Feature Store'
            elif tag == 'matching_engine':
                tag = 'Vertex AI Matching Engine'
            elif tag == 'migration':
                tag = 'CAIP to Vertex AI migration'
            elif tag == 'ml_metadata':
                tag = 'Vertex ML Metadata'
            elif tag == 'model_evaluation':
                tag = 'Vertex AI Model Evaluation'
            elif tag == 'model_monitoring':
                tag = 'Vertex AI Model Monitoring'
            elif tag == 'model_registry':
                tag = 'Vertex AI Model Registry'
            elif tag == 'pipelines':
                tag = 'Vertex AI Pipelines'
            elif tag == 'prediction':
                tag = 'Vertex AI Prediction'
            elif tag == 'pytorch':
                tag = 'Vertex AI Training'
            elif tag == 'reduction_server':
                tag = 'Vertex AI Reduction Server'
            elif tag == 'sdk':
                tag = 'Vertex AI SDK'
            elif tag == 'structured_data':
                tag = 'AutoML / BQML'
            elif tag == 'tabnet':
                tag = 'Vertex AI TabNet'
            elif tag == 'tabular_workflows':
                tag = 'AutoML Tabular Workflows'
            elif tag == 'tensorboard':
                tag = 'Vertex AI TensorBoard'
            elif tag == 'training':
                tag = 'Vertex AI Training'
            elif tag == 'vizier':
                tag = 'Vertex AI Vizier'
                
            # special case
            if 'workbench' in directory:
                tag = 'Vertex AI Workbench'
                
            exit_code += parse_notebook(entry.path, tags=[tag], rules=rules)
            
    return exit_code

def parse_notebook(path: str,
                   tags: List,
                   rules: List) -> int:
    """
        Review the specified notebook for conforming to the notebook template
        and notebook authoring requirements.
        
            path: The path to the notebook.
            tags: The associated tags
            rules: The cell rules to apply
            
        Returns TBA
    """
    notebook = Notebook(path)
    
    # Parse thru the boiler plate
    for rule in rules:
        if rule.validate(notebook):
            if args.debug:
                print("DEBUG: RULE FOUND", rule)

    
    # Automatic Prompt Generation
    if objective.desc != '':       
        add_primary_prompt(path, 
                  links.title, 
                  objective.desc,
                  objective.steps,
        )
        
    # Parse the tutorial
    while not notebook.eon():
        text = []
        code = []
        
        cell = notebook.peek()
        if 'Clean' in cell['source'][0]:
            break
            
        while cell['cell_type'] == 'markdown':
            notebook.pop()
            text += cell['source']
            if notebook.eon():
                break
            cell = notebook.peek()
        
        while cell['cell_type'] == 'code':
            notebook.pop()
            code += cell['source']
            if notebook.eon():
                break
            cell = notebook.peek()
            
        if args.debug:
            print("\nDEBUG: HINT")
            print("Text Hint:", text)
            print("Code Hint:", code)
        else:
            if len(text):
                question = text[0].strip('#').strip()
                response = text[1:]
                write_prompt(question, 
                             response,
                             code
                            )
        
    return 0


class NotebookRule(ABC):
    """
    Abstract class for defining notebook conformance rules
    """
    @abstractmethod
    def validate(self, notebook: Notebook) -> bool:
        '''
        Applies cell specific rules to validate whether the cell 
        does or does not conform to the rules.
        
        Returns whether the cell passed the validation rules
        '''
        pass


class CopyrightRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool:
        """
        Parse the copyright cell
        """
        cell = notebook.peek()
        if 'Copyright' in cell['source'][0]:
            notebook.pop()
            return True
        return False


class NoticesRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool:
        """
        Parse the (optional) notices cell
        """
        cell = notebook.peek()
        if cell['source'][0].startswith('This notebook'):
            notebook.pop()
            return True
        return False


class LinksRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool: 
        """
        Parse the links in the links cell
        """
        self.title = ''
        
        cell = notebook.peek()
        if not cell['source'][0].startswith('# '):
            return False
        else:
            self.title = cell['source'][0][2:].strip()
           
        if len(cell['source']) == 1:
            notebook.pop()
            cell = notebook.peek()

                
        source = ''
        ret = False
        
        for ix in range(len(cell['source'])):
        
            line = cell['source'][ix]
            source += line
            if '<a href="https://github.com' in line:
                ret = True
                    
            if '<a href="https://colab.research.google.com/' in line:
                ret = True

            if '<a href="https://console.cloud.google.com/vertex-ai/workbench/' in line:
                ret = True
               
        
        if ret:
            notebook.pop()
        return ret


class TableRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool:
        """
        Parse the (optional) table of contents cell
        """
        cell = notebook.peek()
        if cell['source'][0].startswith('## Table of contents'):
            notebook.pop()
            return True
        return False


class TestEnvRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool:
        """
        Parse the (optional) test in which environment cell
        """
        cell = notebook.peek()
        if cell['source'][0].startswith('**_NOTE_**: This notebook has been tested'):
            notebook.pop()
            return True
        return False


class OverviewRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool: 
        """
        Parse the overview cell
        """
        
        cell = notebook.peek()
        if not cell['source'][0].startswith("## Overview"):
            return False
        
        last_line = cell['source'][-1]
        if last_line.startswith('Learn more about ['):
            for more in last_line.split('[')[1:]:
                tag = more.split(']')[0]
              
        notebook.pop()
        return True


class ObjectiveRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool: 
        """
        Parse the objective cell.
            Find the description, uses and steps.
        """
        
        self.desc = ''
        self.uses = ''
        self.steps = ''

        cell = notebook.peek()
        if not cell['source'][0].startswith("### Objective"):
            return False

        in_desc = True
        in_uses = False
        in_steps = False
    
        for line in cell['source'][1:]:
            # TOC anchor
            if line.startswith('<a name='):
                continue
                
            if line.startswith('This tutorial uses'):
                in_desc = False
                in_steps = False
                in_uses = True
                self.uses += line
                continue
            elif line.startswith('The steps performed'):
                in_desc = False
                in_uses = False
                in_steps = True
                self.steps += line
                continue

            if in_desc:
                if len(self.desc) > 0 and line.strip() == '':
                    in_desc = False
                    continue
                self.desc += line
            elif in_uses:
                sline = line.strip()
                if len(sline) == 0:
                    self.uses += '\n'
                else:
                    ch = sline[0]
                    if ch in ['-', '*', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                        self.uses += line
            elif in_steps:
                sline = line.strip()
                if len(sline) == 0:
                    self.steps += '\n'
                else:
                    ch = sline[0]
                    if ch in ['-', '*', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                        # check for italic font setting
                        if ch == '*' and sline[1] != ' ':
                            in_steps = False
                        # special case
                        elif sline.startswith('* Prediction Service'):
                            in_steps = False
                        else:
                            self.steps += line
                    elif ch == '#':
                        in_steps = False

            
        if self.desc == '':
            pass
        else:
            self.desc = self.desc.lstrip()
            
            bracket = False
            paren = False
            sentences = ""
            for _ in range(len(self.desc)):
                if self.desc[_] == '[':
                    bracket = True
                    continue
                elif self.desc[_] == ']':
                    bracket = False
                    continue
                elif self.desc[_] == '(':
                    paren = True
                elif self.desc[_] == ')':
                    paren = False
                    continue
                    
                if not paren:
                    sentences += self.desc[_]
            sentences = sentences.split('.')
            if len(sentences) > 1:
                self.desc = sentences[0] + '.\n'
            if self.desc.startswith('In this tutorial, you learn') or self.desc.startswith('In this notebook, you learn'):
                self.desc = self.desc[22].upper() + self.desc[23:]

            
        notebook.pop()
        return True


class RecommendationsRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool: 
        """
        Parse the (optional) recommendations cell
        """
        # (optional) Recommendation
        cell = notebook.peek()
        if cell['source'][0].startswith("### Recommendations"):
            notebook.pop()
            return True
        return False


class DatasetRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool: 
        """
        Parse the dataset cell
        """
        cell = notebook.peek()
        if cell['source'][0].startswith("### Dataset") or \
           cell['source'][0].startswith("### Model") or \
           cell['source'][0].startswith("### Embedding"):
            notebook.pop()
            return True
        return False


class CostsRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool: 
        """
        Parse the costs cell
        """
        
        cell = notebook.peek()
        if cell['source'][0].startswith("### Costs"):
            notebook.pop()
            return True
        return False


# OLD
class OldSetupLocalRule(NotebookRule):
    def helper(self, notebook: Notebook, cell: List) -> bool:
        
        if not cell['source'][0].startswith('### Set up your local development environment'):
            return True
        notebook.pop()
        
        cell = notebook.peek()
        if not cell['source'][0].startswith('**Otherwise**, make sure your environment meets'):
            return True
        notebook.pop()
        return True
        
    def validate(self, notebook: Notebook) -> bool:
        """
        Parse the (optional) setup local environment cell
        """
        cell = notebook.peek()
        if not cell['source'][0].startswith('## Before you begin'):
            self.helper(notebook, cell)
            return False
           
        notebook.pop()

        cell = notebook.peek()
            
        return self.helper(notebook, cell)


class HelpersRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool:
        """
        Parse the (optional) helpers text/code cell
        """
        cell = notebook.peek()
        if 'helper' in cell['source'][0]:
            notebook.pop(2)  # text and cell
            return True
        return False


class InstallationRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool:
        """
        Parse the installation cells
        """
        
        cell = notebook.peek()
        
        if 'Install' not in cell['source'][0]:
            return False
        notebook.pop()

            
        cell = notebook.peek()
        if cell['cell_type'] != 'code':
            return False
        else:
            notebook.pop()
            if cell['source'][0].startswith('! mkdir'):
                cell = notebook.get()
            if 'requirements.txt' in cell['source'][0]:
                cell = notebook.get()
            
            cell = notebook.peek()
            if 'Install' in cell['source'][0]:  # second install cell
                notebook.pop(2)
            
            cell = notebook.peek()
            if 'Install' in cell['source'][0]:  # third install cell
                notebook.pop(2)

        return True

# OLD
class OldRestartRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool:
        """
        Parse the restart cells
        """
        ret = True

        while True:
            cont = False
            cell = notebook.peek()
            for line in cell['source']:
                if 'pip' in line:
                    cont = True
                    break
            if not cont:
                break
            notebook.pop()

        cell = notebook.peek()
        if not cell['source'][0].startswith("### Restart the kernel"):
            return False
        else:
            notebook.pop()
            cell = notebook.peek()  # code cell
            if cell['cell_type'] != 'code':
                return False
           
        notebook.pop()
        return True

class RestartRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool:
        """
        Parse the restart cells
        """
        cell = notebook.peek()
        if not cell['source'][0].startswith("### Colab only: Uncomment the following cell to restart the kernel") and \
           not cell['source'][0].startswith("### Colab Only: Uncomment the following cell to restart the kernel"):
            return False
        else:
            notebook.pop()
            cell = notebook.peek()  # code cell
            if cell['cell_type'] != 'code':
                return False
           
        notebook.pop()
        return True


class VersionsRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool:
        """
        Parse the (optional) package versions code/text cell
        """
        cell = notebook.peek()
        if cell['source'][0].startswith('#### Check package versions') or \
           cell['source'][0].startswith('### Check package versions') or \
           cell['source'][0].startswith('### Check the package versions') or \
           cell['source'][0].startswith('### Check installed package versions') or \
           cell['source'][0].startswith('Check the versions of the packages') :
            notebook.pop(2)  # text and code
            return True
        
        return False

# OLD 
class OldBeforeBeginRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool:
        """
        Parse the before you begin cell
        """
        
        cell = notebook.peek()
        if not cell['source'][0].startswith("## Before you begin"):
            return False
        else:
            notebook.pop()
            # is two cells instead of one
            if len(cell['source']) < 2:
                cell = notebook.peek()
                if cell['source'][0].startswith("### Set up your Google Cloud project"):
                    notebook.pop()
        return True

class BeforeBeginRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool:
        """
        Parse the before you begin cell
        """
        
        cell = notebook.peek()
        if not cell['source'][0].startswith("## Before you begin") and \
           not cell['source'][0].startswith("### Before you begin") and \
           not cell['source'][0].startswith("### Set up your Google Cloud project"):
            return False
        else:
            notebook.pop()

            cell = notebook.peek() 
            if cell['cell_type'] == 'markdown':  # broken into 2 text cells
                notebook.pop()
                cell = notebook.peek()
                
            if cell['cell_type'] != 'code':
                return False
            notebook.pop()
            cell = notebook.peek()
            
            # old
            if cell['cell_type'] == 'code':
                notebook.pop()
                cell = notebook.peek()
                if cell['cell_type'] == 'code':
                    notebook.pop()
                    cell = notebook.peek()
           
        return True

class ProjectNumberRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool:
        """
        Parse the project number cell
        """
        
        cell = notebook.peek()
        if not cell['source'][0].startswith("#### Get your project number"):
            return False
        else:
            notebook.pop()

            cell = notebook.peek()
                
            if cell['cell_type'] != 'code':
                return False
           
        notebook.pop()
        return True


class EnableAPIsRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool:
        """
        Parse the (optional) enable apis code/text cell
        """
        cell = notebook.peek()
        if cell['source'][0].startswith("### Enable APIs"):
            notebook.pop(2)  # text and code
            return True
        return False


class NoteServiceAccountRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool:
        """
        Parse the (optional) notes about service account text cell
        """
        cell = notebook.peek()
        if cell['source'][0].startswith("## Notes about service account"):
            notebook.pop()
            return True
        return False

# OLD
class OldSetupProjectRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool:
        """
        Parse the set project cells
        """
        ret = True
        
        cell = notebook.peek()
        if not cell['source'][0].startswith('#### Set your project ID'):
            return False
        else: 
            notebook.pop()
            cell = notebook.peek()
            if cell['cell_type'] != 'code':
                return False
            elif not cell['source'][0].startswith('PROJECT_ID = "[your-project-id]"'):
                return False
            notebook.pop()

            cell = notebook.peek()
            if cell['cell_type'] != 'code' or 'or PROJECT_ID == "[your-project-id]":' not in cell['source'][0]:
                return False
            notebook.pop()

            cell = notebook.peek()
            if cell['cell_type'] != 'code' or '! gcloud config set project' not in cell['source'][0]:
                return False
            notebook.pop()
            
        return True


class RegionRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool:
        """
        Parse the region code/text cell
        """
        cell = notebook.peek()
        if not cell['source'][0].startswith("#### Region"):
            return False
        else:
            notebook.pop()

            cell = notebook.peek()  # code cell
            if cell['cell_type'] != 'code':
                return False
            notebook.pop()
            
            cell = notebook.peek()  # 2nd code cell
            if cell['cell_type'] == 'code':
                notebook.pop()
           
        return True


class EmailRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool:
        """
        Parse the email code/text cell
        """
        cell = notebook.peek()
        if not cell['source'][0].startswith("#### Email") and \
           not cell['source'][0].startswith("#### User Email"):
            return False
        else:
            notebook.pop()

            cell = notebook.peek()  # code cell
            if cell['cell_type'] != 'code':
                return False

        notebook.pop()

        cell = notebook.peek()
        if cell['cell_type'] == 'code':
            notebook.pop()
            
        return True


class UUIDRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool:
        """
        Parse the UUID/Timestamp code/text cell
        """
        cell = notebook.peek()
        if not cell['source'][0].startswith("#### UUID") and \
           not cell['source'][0].startswith("#### Timestamp"):
            return False
        else:
            notebook.pop()

            cell = notebook.peek()  # code cell
            if cell['cell_type'] != 'code':
                return False
           
        notebook.pop()
        return True


class AuthenticateRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool:
        """
        Parse the authenticate cells
        """
        
        cell = notebook.peek()
        if not cell['source'][0].startswith("### Authenticate your Google Cloud account"):
            return False
        else:
            notebook.pop()
            
            while True:
                cell = notebook.get()
                if cell['source'][0].startswith('**If you are using'):  # old 
                    notebook.pop()
                    break
                    
                if cell['source'][0].startswith('**1. Vertex AI Workbench**'):
                    continue
                elif cell['source'][0].startswith('**2. Local JupyterLab instance, uncomment and run:**'):
                    continue
                elif cell['source'][0].startswith('# ! gcloud auth login'):
                    continue
                elif cell['source'][0].startswith('**3. Colab'):
                    continue
                elif cell['source'][0].startswith('# from google.colab import auth') or \
                     cell['source'][0].startswith('IS_COLAB = False'):
                    continue
                elif cell['source'][0].startswith('**4.'):
                    break
                elif cell['cell_type'] == 'code':  # old
                    break
                else:
                    notebook._cell_index -= 1
                    break
                    
        return True  


class NotesServiceRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool:
        """
        Parse the notes about service account cells
        """
        cell = notebook.peek()
        if not cell['source'][0].startswith('### Notes about service account and permission'):
            return False
        
        notebook.pop()
        return True


class CreateBucketRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool:
        """
        Parse the create bucket cells
        """
        cell = notebook.peek()
        
        if not cell['source'][0].startswith("### Create a Cloud Storage bucket"):
            return False
        else:
            notebook.pop()

            cell = notebook.peek()  # code cell
            if cell['cell_type'] != 'code':
                return False
            notebook.pop()
            
            cell = notebook.peek()
            if cell['cell_type'] == 'code':
                notebook.pop()
                cell = notebook.peek()
        
            if not cell['source'][0].startswith("**Only if your bucket"):
                return False
            notebook.pop()

            cell = notebook.peek()  # code cell
            if cell['cell_type'] != 'code':
                return False
            notebook.pop()
            
            cell = notebook.peek()  # optional
            if cell['source'][0].startswith('Finally, validate access to your Cloud Storage'):
                notebook.pop(2)
            return True


class ServiceAccountRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool:
        """
        Parse the service account cells
        """
        cell = notebook.peek()
        
        if not cell['source'][0].startswith("#### Service Account"):
            return False
        else:
            notebook.pop()

            cell = notebook.peek()  # code cell
            if cell['cell_type'] != 'code':
                return False
            notebook.pop()

            cell = notebook.peek()  # code cell
            if cell['cell_type'] == 'code':
                notebook.pop()
                
            return True


class SetServiceAccountRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool:
        """
        Parse the set service account text/code cells
        """
        cell = notebook.peek()
        
        if not cell['source'][0].startswith("#### Set service account") and \
           not cell['source'][0].startswith("### Service Account"):
            return False
        else:
            notebook.pop()

            cell = notebook.peek()  # code cell
            if cell['cell_type'] != 'code':
                return False
            notebook.pop()
            
            cell = notebook.peek()  # code cell
            if cell['cell_type'] == 'code':
                notebook.pop()
            
            return True    


class SetServiceAccounAccesstRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool:
        """
        Parse the set service account acces text/code cells
        """
        cell = notebook.peek()
        
        if not cell['source'][0].startswith("#### Set service account access"):
            return False
        else:
            notebook.pop()

            cell = notebook.peek()  # code cell
            if cell['cell_type'] != 'code':
                return False
            notebook.pop()
            
            return True


class ImportLibsRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool:
        """
        Parse the set import libraries text/code cells
        """
        cell = notebook.peek()
        
        if not cell['source'][0].startswith("#### Import") and \
           not cell['source'][0].startswith("### Import") and \
           not cell['source'][0].startswith("## Import") and \
           not cell['source'][0].startswith('### Set up variables'):
            return False
        else:
            notebook.pop()

            cell = notebook.peek()  # code cell
            if cell['source'][0].startswith('#### Import'):
                notebook.pop()
                cell = notebook.peek()
                
            if cell['cell_type'] != 'code':
                return False
            notebook.pop()
            
            return True


def write_prompt(question: str,
                 response: str,
                 code) -> None:
    print("prompt:", question)
    print("response:")
    if isinstance(response, list):
        for line in response:
            print(line)
    else:
        print(response)
    if isinstance(code, list):
        for line in code:
            print(line)
    else:
        print(code)
    print('')


def add_primary_prompt(path: str,
                title: str,
                desc: str,
                steps: str) -> None:
    '''
    '''
    if args.debug:
        print("PRIMARY PROMPT")
        print(f"DEBUG: {path}")
        print(f"DEBUG: {title}")
        print(f"DEBUG: {desc}")
        print(f"DEBUG: {steps}")
        
    else:
        write_prompt(title,
                     desc,
                     steps)


# Instantiate the rules
copyright = CopyrightRule()
notices = NoticesRule()
links = LinksRule()
testenv = TestEnvRule()
table = TableRule()
overview = OverviewRule()
objective = ObjectiveRule()
recommendations = RecommendationsRule()
dataset = DatasetRule()
costs = CostsRule()
old_setuplocal = OldSetupLocalRule()
helpers = HelpersRule()
installation = InstallationRule()
old_restart = OldRestartRule()
restart = RestartRule()
versions = VersionsRule()
beforebegin = BeforeBeginRule()
projectno = ProjectNumberRule()
enableapis = EnableAPIsRule()
notessa = NoteServiceAccountRule()
old_setupproject = OldSetupProjectRule()
region = RegionRule()
email = EmailRule()
uuid = UUIDRule()
authenticate = AuthenticateRule()
notesservice = NotesServiceRule()
createbucket = CreateBucketRule()
serviceaccount = ServiceAccountRule()
setserviceaccount = SetServiceAccountRule()
setsaa = SetServiceAccounAccesstRule()
importlibs = ImportLibsRule()

 # Cell Validation
rules = [ copyright, notices, links, testenv, table, overview, objective,
          recommendations, dataset, costs, old_setuplocal, helpers,
          installation, old_restart, restart, versions, beforebegin, 
          projectno, notessa, enableapis,
          old_setupproject, region, email, uuid, email, authenticate, notesservice,
          createbucket, serviceaccount, setserviceaccount, setsaa, importlibs
]


if args.notebook_dir:
    if not os.path.isdir(args.notebook_dir):
        print(f"Error: not a directory: {args.notebook_dir}", file=sys.stderr)
        exit(1)
    exit_code = parse_dir(args.notebook_dir)
elif args.notebook:
    if not os.path.isfile(args.notebook):
        print(f"Error: not a notebook: {args.notebook}", file=sys.stderr)
        exit(1)
    exit_code = parse_notebook(args.notebook, tags=[], rules=rules)
elif args.notebook_file:
    if not os.path.isfile(args.notebook_file):
        print("Error: file does not exist", args.notebook_file)
    else:
        exit_code = 0
        with open(args.notebook_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            heading = True
            for row in reader:
                if heading:
                    heading = False
                else:
                    tags = row[0].split(',')
                    notebook = row[1]
                    exit_code += parse_notebook(notebook, tags=tags, rules=rules)
else:
    print("Error: must specify a directory or notebook", file=sys.stderr)
    exit(1)

exit(exit_code)
