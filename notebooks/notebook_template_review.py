"""
    AutoReview: Script to automatically review Vertex AI notebooks for conformance to notebook template requirements:
    
    python3 notebook_template_review.py [options]
        # options for selecting notebooks
        --notebook: review the specified notebook
        --notebook-dir: recursively traverse the directory and review each notebook enocuntered
        --notebook-file: A CSV file with list of notebooks to review.
        
        # options for error handling
        --errors: Report detected errors.
        --errors-codes: A list of error codes to report errors. Otherwise, all errors are reported.
        --errors-csv: Report errors in CSV format
        
        # index generatation
        --repo: Generate index in markdown format
        --web: Generate index in HTML format
        --title: Add title to index
        --desc: Add description to index
        --steps: Add steps to index
        --uses: Add "resources" used to index
        
    Format of CSV file for notebooks to review:
    
        tags,notebook-path,backlink
        
        tags: Double quoted ist of tags: e.g., "AutoML, Tabular Data"
        notebook-path: path of notebook, relative to https://github.com/GoogleCloudPlatform/vertex-ai-samples/notebooks
        backlink: webdoc page with more details, relative to https://cloud.google.com/
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
parser.add_argument('--notebook-dir', dest='notebook_dir',
                    default=None, type=str, help='Notebook directory')
parser.add_argument('--notebook', dest='notebook',
                    default=None, type=str, help='Notebook to review')
parser.add_argument('--notebook-file', dest='notebook_file',
                    default=None, type=str, help='File with list of notebooks to review')
parser.add_argument('--errors', dest='errors', action='store_true', 
                    default=False, help='Report errors')
parser.add_argument('--errors-csv', dest='errors_csv', action='store_true', 
                    default=False, help='Report errors as CSV')
parser.add_argument('--errors-codes', dest='errors_codes',
                    default=None, type=str, help='Report only specified errors')
parser.add_argument('--title', dest='title', action='store_true',
                    default=False, help='Output description')
parser.add_argument('--desc', dest='desc', action='store_true', 
                    default=False, help='Output description')
parser.add_argument('--uses', dest='uses', action='store_true', 
                    default=False, help='Output uses (resources)')
parser.add_argument('--steps', dest='steps', action='store_true', 
                    default=False, help='Ouput steps')
parser.add_argument('--web', dest='web', action='store_true', 
                    default=False, help='Output format in HTML')
parser.add_argument('--repo', dest='repo', action='store_true', 
                    default=False, help='Output format in Markdown')
parser.add_argument('--fix', dest='fix', action='store_true', 
                    default=False, help='Fix the notebook non-conformance errors')
parser.add_argument('--fix-codes', dest='fix_codes',
                    default=None, type=str, help='Fix only specified errors')
args = parser.parse_args()

if args.errors_codes:
    args.errors_codes = args.errors_codes.split(',')
    args.errors = True

if args.errors_csv:
    args.errors = True

if args.fix_codes:
    args.fix_codes = args.fix_codes.split(',')
    args.fix = True


class ErrorCode(Enum):
    # Copyright cell
    #   Google copyright cell required
    ERROR_COPYRIGHT = 0,

    # Links cell
    #   H1 heading required
    #   git, colab and workbench link required
    #   links must be valid links
    ERROR_TITLE_HEADING = 1,
    ERROR_HEADING_CASE = 2,
    ERROR_HEADING_CAP = 3,
    ERROR_LINK_GIT_MISSING = 4,
    ERROR_LINK_COLAB_MISSING = 5,
    ERROR_LINK_WORKBENCH_MISSING = 6,
    ERROR_LINK_GIT_BAD = 7,
    ERROR_LINK_COLAB_BAD = 8,
    ERROR_LINK_WORKBENCH_BAD = 9,

    # Overview cells
    #   Overview cell required
    #   Objective cell required
    #   Dataset cell required
    #   Costs cell required
    #     Check for required Vertex and optional BQ and Dataflow
    ERROR_OVERVIEW_NOTFOUND = 10,
    ERROR_OBJECTIVE_NOTFOUND = 11,
    ERROR_OBJECTIVE_MISSING_DESC = 12,
    ERROR_OBJECTIVE_MISSING_USES = 13,
    ERROR_OBJECTIVE_MISSING_STEPS = 14,
    ERROR_DATASET_NOTFOUND = 15,
    ERROR_COSTS_NOTFOUND = 16,
    ERROR_COSTS_MISSING = 17,

    # Installation cell
    #   Installation cell required
    #   Wrong heading for installation cell
    #   Installation code cell not found
    #   pip3 required
    #   option -q required
    #   option {USER_FLAG} required
    #   installation code cell not match template
    #   all packages must be installed as a single pip3
    ERROR_INSTALLATION_NOTFOUND = 18,
    ERROR_INSTALLATION_HEADING = 19,
    ERROR_INSTALLATION_CODE_NOTFOUND = 20,
    ERROR_INSTALLATION_PIP3 = 21,
    ERROR_INSTALLATION_QUIET = 22,
    ERROR_INSTALLATION_USER_FLAG = 23,
    ERROR_INSTALLATION_CODE_TEMPLATE = 24,
    ERROR_INSTALLATION_SINGLE_PIP3 = 25,

    # Restart kernel cell
    #    Restart code cell required
    #    Restart code cell not found
    ERROR_RESTART_NOTFOUND = 23,
    ERROR_RESTART_CODE_NOTFOUND = 24,

    # Before you begin cell
    #    Before you begin cell required
    #    Before you begin cell incomplete
    ERROR_BEFOREBEGIN_NOTFOUND = 25,
    ERROR_BEFOREBEGIN_INCOMPLETE = 26,

    # Set Project ID
    #    Set project ID cell required
    #    Set project ID code cell not found
    #    Set project ID not match template
    ERROR_PROJECTID_NOTFOUND = 27,
    ERROR_PROJECTID_CODE_NOTFOUND = 28,
    ERROR_PROJECTID_TEMPLATE = 29,

    # Technical Writer Rules
    ERROR_TWRULE_TODO = 51,
    ERROR_TWRULE_FIRSTPERSON = 52,
    ERROR_TWRULE_FUTURETENSE = 53,
    ERROR_TWRULE_BRANDING = 54,

    ERROR_EMPTY_CALL = 101

class FixCode(Enum):
    FIX_BAD_LINK = 0,
    FIX_PLACEHOLDER = 1


# globals
last_tag = ''


def parse_dir(directory: str) -> int:
    """
        Recursively walk the specified directory, reviewing each notebook (.ipynb) encountered.
        
            directory: The directory path.
            
        Returns the numbern of errors
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
                
            exit_code += parse_notebook(entry.path, tags=[tag], linkback=None, rules=rules)
            
    return exit_code


def parse_notebook(path: str,
                   tags: List,
                   linkback: str,
                   rules: List) -> int:
    """
        Review the specified notebook for conforming to the notebook template
        and notebook authoring requirements.
        
            path: The path to the notebook.
            tags: The associated tags
            linkback: A link back to the web docs
            rules: The cell rules to apply
            
        Returns the number of errors
    """
    notebook = Notebook(path)
    
    for rule in rules:
        rule.validate(notebook)

    
    # Automatic Index Generation
    if objective.desc != '':
        if overview.linkbacks:
            linkbacks = overview.linkbacks
        else:
            linkbacks = [linkback]
        if overview.tags:
            tags = overview.tags
                
        add_index(path, 
                  tags, 
                  linkbacks,
                  title.title, 
                  objective.desc, 
                  objective.uses, 
                  objective.steps, 
                  links.git_link, 
                  links.colab_link, 
                  links.workbench_link
        )
        
    if args.fix:
        notebook.writeback()
        
    return notebook.num_errors

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
        self._num_errors = 0
        
        # cross cell information
        self._costs = []

        
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
    

    @property
    def path(self):
        '''
        Getter: return the filename path for the notebook
        '''
        return self._path
    

    @property
    def num_errors(self):
        '''
        Getter: return the number of errors
        '''
        return self._num_errors

        
    def report_error(self,
                     code: ErrorCode,
                     errmsg: str):
        """
        Report an error.
            If args.errors_codes set, then only report these errors. Otherwise, all errors.

        code: The error code number.
        errmsg: The error message
        """

        if args.errors:
            code = code.value[0]
            if args.errors_codes:
                if str(code) not in args.errors_codes:
                    return

            if args.errors_csv:
                print(self._path, ',', code)
            else:
                print(f"{self._path}: ERROR ({code}): {errmsg}", file=sys.stderr)
                self._num_errors += 1
                
            return False
        return True


    def report_fix(self,
                   code: FixCode,
                   fixmsg: str):
        """
        Report an automatic fix
        
            code: The fox code number.
            fixmsg: The autofix message
        Returns:
            Whether code is to be fixed
        """
        if args.fix:
            code = code.value[0]
            if args.fix_codes:
                if str(code) not in args.fix_codes:
                    return False
                
            print(f"{self._path}: FIXED ({code}): {fixmsg}", file=sys.stderr)
            return True
        return False
        
                
    def writeback(self):
        """
        Write back the updated (autofixed) notebook 
        """
        with open(self._path, 'w') as f:
            json.dump(self._content, f)



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
        cell = notebook.get()
        if not 'Copyright' in cell['source'][0]:
            return notebook.report_error(ErrorCode.ERROR_COPYRIGHT, "missing copyright cell")
        return True


class NoticesRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool:
        """
        Parse the (optional) notices cell
        """
        cell = notebook.peek()
        if cell['source'][0].startswith('This notebook'):
            notebook.pop()
        return True


class TitleRule(NotebookRule): 
    def validate(self, notebook: Notebook) -> bool: 
        """
        Parse the title in the links cell
        """
        ret = True
        self.title = ''
        
        cell = notebook.peek()
        if not cell['source'][0].startswith('# '):
            ret = notebook.report_error(ErrorCode.ERROR_TITLE_HEADING, "title cell must start with H1 heading")
        else:
            self.title = cell['source'][0][2:].strip()
            SentenceCaseTWRule().validate(notebook, [self.title])

            # H1 title only
            if len(cell['source']) == 1:
                notebook.pop()
                
        return ret


class LinksRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool: 
        """
        Parse the links in the links cell
        """

        self.git_link = None
        self.colab_link = None
        self.workbench_link = None
        source = ''
        ret = True
        
        cell = notebook.get()
        for ix in range(len(cell['source'])):
        
            line = cell['source'][ix]
            source += line
            if '<a href="https://github.com' in line:
                self.git_link = line.strip()[9:-2].replace('" target="_blank', '').replace('" target=\'_blank', '')
                
                derived_link = os.path.join('https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/', notebook.path)
                if self.git_link != derived_link:
                    if notebook.report_fix(FixCode.FIX_BAD_LINK, f"fixed GitHub link: {derived_link}"):
                        fix_link = f"<a href=\"{derived_link}\" target='_blank'>\n"
                        cell['source'][ix] = fix_link
                    else:
                        ret = notebook.report_error(ErrorCode.ERROR_LINK_GIT_BAD, f"bad GitHub link: {self.git_link}")
                    
            if '<a href="https://colab.research.google.com/' in line:
                self.colab_link = 'https://colab.research.google.com/github/' + line.strip()[50:-2].replace('" target="_blank', '').replace('" target=\'_blank', '')
 
                derived_link = os.path.join('https://colab.research.google.com/github/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks', notebook.path)
                if self.colab_link != derived_link:
                    if notebook.report_fix(FixCode.FIX_BAD_LINK, f"fixed Colab link: {derived_link}"):
                        fix_link = f"<a href=\"{derived_link}\" target='_blank'>\n"
                        cell['source'][ix] = fix_link
                    else:
                        ret = notebook.report_error(ErrorCode.ERROR_LINK_COLAB_BAD, f"bad Colab link: {self.colab_link}")

            if '<a href="https://console.cloud.google.com/vertex-ai/workbench/' in line:
                self.workbench_link = line.strip()[9:-2].replace('" target="_blank', '').replace('" target=\'_blank', '')

                derived_link = os.path.join('https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/main/notebooks/', notebook.path)
                if self.workbench_link != derived_link:
                    if notebook.report_fix(FixCode.FIX_BAD_LINK, f"fixed Workbench link: {derived_link}"):
                        fix_link = f"<a href=\"{derived_link}\" target='_blank'>\n"
                        cell['source'][ix] = fix_link
                    else:
                        ret = notebook.report_error(ErrorCode.ERROR_LINK_WORKBENCH_BAD, f"bad Workbench link: {self.workbench_link}")


        if 'View on GitHub' not in source or not self.git_link:
            ret = notebook.report_error(ErrorCode.ERROR_LINK_GIT_MISSING, 'Missing link for GitHub')
        if 'Run in Colab' not in source or not self.colab_link:
            ret = notebook.report_error(ErrorCode.ERROR_LINK_COLAB_MISSING, 'Missing link for Colab')    
        if 'Open in Vertex AI Workbench' not in source or not self.workbench_link:
            ret = notebook.report_error(ErrorCode.ERROR_LINK_WORKBENCH_MISSING, 'Missing link for Workbench')
        
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


class TestEnvRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool:
        """
        Parse the (optional) test in which environment cell
        """
        cell = notebook.peek()
        if cell['source'][0].startswith('**_NOTE_**: This notebook has been tested'):
            notebook.pop()
        return True


class OverviewRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool: 
        """
        Parse the overview cell
        """
        self.linkbacks = []
        self.tags = []
        
        cell = notebook.get()
        if not cell['source'][0].startswith("## Overview"):
            return notebook.report_error(ErrorCode.ERROR_OVERVIEW_NOTFOUND, "Overview section not found")
        
        last_line = cell['source'][-1]
        if last_line.startswith('Learn more about ['):
            for more in last_line.split('[')[1:]:
                tag = more.split(']')[0]
                linkback = more.split('(')[1].split(')')[0]
                self.tags.append(tag)
                self.linkbacks.append(linkback)
                
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
        self.costs = []
        ret = True

        cell = notebook.get()
        if not cell['source'][0].startswith("### Objective"):
            ret = notebook.report_error(ErrorCode.ERROR_OBJECTIVE_NOTFOUND, "Objective section not found")
            notebook.costs = []
            return ret

        in_desc = True
        in_uses = False
        in_steps = False
    
        for line in cell['source'][1:]:
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
                        else:
                            self.steps += line
                    elif ch == '#':
                        in_steps = False
            
        if self.desc == '':
            ret = notebook.report_error(ErrorCode.ERROR_OBJECTIVE_MISSING_DESC, "Objective section missing desc")
        else:
            self.desc = self.desc.lstrip()
            sentences = self.desc.split('.')
            if len(sentences) > 1:
                self.desc = sentences[0] + '.\n'
            if self.desc.startswith('In this tutorial, you learn') or self.desc.startswith('In this notebook, you learn'):
                self.desc = self.desc[22].upper() + self.desc[23:]

        if self.uses == '':
            ret = notebook.report_error(ErrorCode.ERROR_OBJECTIVE_MISSING_USES, "Objective section missing uses services list")
        else:
            if 'BigQuery' in self.uses:
                self.costs.append('BQ')
            if 'Vertex' in self.uses:
                self.costs.append('Vertex')
            if 'Dataflow' in self.uses:
                self.costs.append('Dataflow')

        if self.steps == '':
            ret = notebook.report_error(ErrorCode.ERROR_OBJECTIVE_MISSING_STEPS, "Objective section missing steps list")
            
        notebook.costs = self.costs
        ret = True


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


class DatasetRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool: 
        """
        Parse the dataset cell
        """
        cell = notebook.get()
        if not cell['source'][0].startswith("### Dataset") and not cell['source'][0].startswith("### Model") and not cell['source'][0].startswith("### Embedding"):
            return notebook.report_error(ErrorCode.ERROR_DATASET_NOTFOUND, "Dataset/Model section not found")
        return True


class CostsRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool: 
        """
        Parse the costs cell
        """
        ret = True
        
        cell = notebook.get()
        if not cell['source'][0].startswith("### Costs"):
            ret = notebook.report_error(ErrorCode.ERROR_COSTS_NOTFOUND, "Costs section not found")
        else:
            text = ''
            for line in cell['source']:
                text += line
            if 'BQ' in notebook.costs and 'BigQuery' not in text:
                ret = notebook.report_error(ErrorCode.ERROR_COSTS_MISSING, 'Costs section missing reference to BiqQuery')
            if 'Vertex' in notebook.costs and 'Vertex' not in text:
                ret = notebook.report_error(ErrorCode.ERROR_COSTS_MISSING, 'Costs section missing reference to Vertex')
            if 'Dataflow' in notebook.costs and 'Dataflow' not in text:    
                ret = notebook.report_error(ErrorCode.ERROR_COSTS_MISSING, 'Costs section missing reference to Dataflow')
        return ret



class SetupLocalRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool:
        """
        Parse the (optional) setup local environment cell
        """
        cell = notebook.peek()
        if cell['source'][0].startswith('## Before you begin'):
            notebook.pop()

        cell = notebook.peek()
        if not cell['source'][0].startswith('### Set up your local development environment'):
            return True
        notebook.pop()
        
        cell = notebook.peek()
        if cell['source'][0].startswith('**Otherwise**, make sure your environment meets'):
            notebook.pop()
            
        return True


class HelpersRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool:
        """
        Parse the (optional) helpers text/code cell
        """
        cell = notebook.peek()
        if 'helper' in cell['source'][0]:
            notebook.pop(2)  # text and cell
        return True


class InstallationRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool:
        """
        Parse the installation cells
        """
        ret = True
        
        cell = notebook.get()
        
        if 'Install' not in cell['source'][0]:
            return notebook.report_error(ErrorCode.ERROR_INSTALLATION_NOTFOUND, "Installation section not found")

        if not cell['source'][0].startswith("## Install"):
            ret = notebook.report_error(ErrorCode.ERROR_INSTALLATION_HEADING, "Installation section needs to be H2 heading")
       
            
        cell = notebook.get()
        if cell['cell_type'] != 'code':
            ret = notebook.report_error(ErrorCode.ERROR_INSTALLATION_NOTFOUND, "Installation section not found")
        else:
            if cell['source'][0].startswith('! mkdir'):
                cell = notebook.get()
            if 'requirements.txt' in cell['source'][0]:
                cell = notebook.get()

            text = ''
            for line in cell['source']:
                text += line
                if 'pip ' in line:
                    if 'pip3' not in line:
                        notebook.report_error(ErrorCode.ERROR_INSTALLATION_PIP3, "Installation code section: use pip3")
                    if line.endswith('\\\n'):
                        continue
                    if '-q' not in line and '--quiet' not in line :
                        notebook.report_error(ErrorCode.ERROR_INSTALLATION_QUIET, "Installation code section: use -q with pip3")
                    if 'USER_FLAG' not in line and 'sh(' not in line:
                        notebook.report_error(ErrorCode.ERROR_INSTALLATION_USER_FLAG, "Installation code section: use {USER_FLAG} with pip3")
            if 'required_packages <' in text:
                pass  # R kernel
            elif 'if IS_WORKBENCH_NOTEBOOK:' not in text:
                ret = notebook.report_error(ErrorCode.ERROR_INSTALLATION_CODE_TEMPLATE, "Installation code section out of date (see template)")
        return ret


class RestartRule(NotebookRule):
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
                    ret = notebook.report_error(ErrorCode.ERROR_INSTALLATION_SINGLE_PIP3, f"All pip installations must be in a single code cell: {line}")
                    cont = True
                    break
            if not cont:
                break
            notebook.pop()

        cell = notebook.peek()
        if not cell['source'][0].startswith("### Restart the kernel"):
            ret = notebook.report_error(ErrorCode.ERROR_RESTART_NOTFOUND, "Restart the kernel section not found")
        else:
            notebook.pop()
            cell = notebook.get()  # code cell
            if cell['cell_type'] != 'code':
                ret = notebook.report_error(ErrorCode.ERROR_RESTART_CODE_NOTFOUND, "Restart the kernel code section not found")
                
        return ret


class VersionsRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool:
        """
        Parse the (optional) package versions code/text cell
        """
        cell = notebook.peek()
        if cell['source'][0].startswith('#### Check package versions'):
            notebook.pop(2)  # text and code
        return True


class BeforeBeginRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool:
        """
        Parse the before you begin cell
        """
        ret = True
        
        cell = notebook.get()
        if not cell['source'][0].startswith("## Before you begin"):
            ret = notebook.report_error(ErrorCode.ERROR_BEFOREBEGIN_NOTFOUND, "Before you begin section not found")
        else:
            # is two cells instead of one
            if len(cell['source']) < 2:
                cell = notebook.get()
                if not cell['source'][0].startswith("### Set up your Google Cloud project"):
                    ret = notebook.report_error(ErrorCode.ERROR_BEFOREBEGIN_INCOMPLETE, "Before you begin section incomplete")
        return ret


class EnableAPIsRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool:
        """
        Parse the (optional) enable apis code/text cell
        """
        cell = notebook.peek()
        if cell['source'][0].startswith("### Enable APIs"):
            notebook.pop(2)  # text and code
        return True


class SetupProjectRule(NotebookRule):
    def validate(self, notebook: Notebook) -> bool:
        """
        Parse the set project cells
        """
        ret = True
        
        cell = notebook.get()
        if not cell['source'][0].startswith('#### Set your project ID'):
            ret = notebook.report_error(ErrorCode.ERROR_PROJECTID_NOTFOUND, "Set project ID section not found")
        else: 
            cell = notebook.get()
            if cell['cell_type'] != 'code':
                ret = notebook.report_error(ErrorCode.ERROR_PROJECTID_CODE_NOTFOUND, "Set project ID code section not found")
            elif not cell['source'][0].startswith('PROJECT_ID = "[your-project-id]"'):
                ret = notebook.report_error(ErrorCode.ERROR_PROJECTID_TEMPLATE, "Set project ID not match template")

            cell = notebook.get()
            if cell['cell_type'] != 'code' or 'or PROJECT_ID == "[your-project-id]":' not in cell['source'][0]:
                ret = notebook.report_error(ErrorCode.ERROR_PROJECTID_TEMPLATE, "Set project ID not match template")  

            cell = notebook.get()
            if cell['cell_type'] != 'code' or '! gcloud config set project' not in cell['source'][0]:
                ret = notebook.report_error(ErrorCode.ERROR_PROJECTID_TEMPLATE, "Set project ID not match template")
                
        return ret


class TextRule(ABC):
    """
    Abstract class for defining text writing conformance rules
    """
    @abstractmethod
    def validate(self, notebook: Notebook, text: List[str]) -> bool:
        '''
        Applies text writing specific rules to validate whether the text 
        does or does not conform to the rules.
        
        Returns whether the test passed the validation rules
        '''
        return False


class BrandingRule(TextRule):
    def validate(self, notebook: Notebook, text: List[str]) -> bool:
        """
            Check the text for branding issues
                1. Product branding names
                2. No future tense
                3. No 1st person

        """
        ret = True
        branding = {
                'Vertex SDK': 'Vertex AI SDK',
                'Vertex Training': 'Vertex AI Training',
                'Vertex Prediction': 'Vertex AI Prediction',
                'Vertex Batch Prediction': 'Vertex AI Batch Prediction',
                'Vertex XAI': 'Vertex Explainable AI',
                'Vertex Explainability': 'Vertex Explainable AI',
                'Vertex AI Explainability': 'Vertex Explainable AI',
                'Vertex Pipelines': 'Vertex AI Pipelines',
                'Vertex Experiments': 'Vertex AI Experiments',
                'Vertex TensorBoard': 'Vertex AI TensorBoard',
                'Vertex Hyperparameter Tuning': 'Vertex AI Hyperparameter Tuning',
                'Vertex Metadata': 'Vertex ML Metadata',
                'Vertex AI Metadata': 'Vertex ML Metadata',
                'Vertex AI ML Metadata': 'Vertex ML Metadata',
                'Vertex Vizier': 'Vertex AI Vizier',
                'Vertex Feature Store': 'Vertex AI Feature Store',
                'Vertex Forecasting': 'Vertex AI Forecasting',
                'Vertex Matching Engine': 'Vertex AI Matching Engine',
                'Vertex TabNet': 'Vertex AI TabNet',
                'Tabnet': 'TabNet',
                'Vertex Two Towers': 'Vertex AI Two-Towers',
                'Vertex Two-Towers': 'Vertex AI Two-Towers',
                'Vertex Dataset': 'Vertex AI Dataset',
                'Vertex Model': 'Vertex AI Model',
                'Vertex Endpoint': 'Vertex AI Endpoint',
                'Vertex Private Endpoint': 'Vertex AI Private Endpoint',
                'Automl': 'AutoML',
                'AutoML Tables': 'AutoML Tabular',
                'AutoML Vision': 'AutoML Image',
                'AutoML Language': 'AutoML Text',
                'Tensorflow': 'TensorFlow',
                'Tensorboard': 'TensorBoard',
                'Google Cloud Notebooks': 'Vertex AI Workbench Notebooks',
                'BQ ': 'BigQuery',
                'BQ.': 'BigQuery',
                'Bigquery': 'BigQuery',
                'BQML': 'BigQuery ML',
                'GCS ': 'Cloud Storage',
                'GCS.': 'Cloud Storage',
                'Google Cloud Storage': 'Cloud Storage',
                'Pytorch': 'PyTorch',
                'Sklearn': 'scikit-learn',
                'sklearn': 'scikit-learn'
        }

        for line in text:
            for mistake, brand in branding.items():
                if mistake in line:
                    ret = notebook.report_error(ErrorCode.ERROR_TWRULE_BRANDING, f"Branding {mistake} -> {brand}: {line}")

        return ret


class SentenceCaseTWRule(TextRule):
    def validate(self,
                 notebook,
                 text: List[str]) -> bool:
        """
        Check that headings are in sentence case

        path: used only for reporting an error
        text: the heading to check
        """
        ret = True

        ACRONYMS = ['E2E', 'Vertex', 'AutoML', 'ML', 'AI', 'GCP', 'API', 'R', 'CMEK', 
                    'TF', 'TFX', 'TFDV', 'SDK', 'VM', 'CPR', 'NVIDIA', 'ID', 'DASK', 
                    'ARIMA_PLUS', 'KFP', 'I/O', 'GPU', 'Google', 'TensorFlow', 'PyTorch'
                    ]

        # Check the first line
        words = text[0].replace('#', '').split(' ')
        if not words[0][0].isupper():
            ret = notebook.report_error(ErrorCode.ERROR_HEADING_CAP, f"heading must start with capitalized word: {words[0]}")

        for word in words[1:]:
            word = word.replace(':', '').replace('(', '').replace(')', '')
            if word in ACRONYMS:
                continue
            if word.isupper():
                ret = notebook.report_error(ErrorCode.ERROR_HEADING_CASE, f"heading is not sentence case: {word}")
                
        return ret



class TextTWRule(TextRule):
    def validate(self, notebook: Notebook, text: List[str]) -> bool:
        """
        Check for conformance to the following techwriter rules
                1. No future tense
                2. No 1st person

        """
        ret = True
    
        for line in text:
            # HTML code
            if '<a ' in line:
                continue

            if 'TODO' in line or 'WIP' in line:
                ret = notebook.report_error(ErrorCode.ERROR_TWRULE_TODO, f'TODO in cell: {line}')
            if 'we ' in line.lower() or "let's" in line.lower() in line.lower():
                ret = notebook.report_error(ErrorCode.ERROR_TWRULE_FIRSTPERSON, f'Do not use first person (e.g., we), replace with 2nd person (you): {line}')
            if 'will' in line.lower() or 'would' in line.lower():
                ret = notebook.report_error(ErrorCode.ERROR_TWRULE_FUTURETENSE, f'Do not use future tense (e.g., will), replace with present tense: {line}')

                    
        return ret


def add_index(path: str, 
              tags: List, 
              linkbacks: List,
              title : str, 
              desc: str, 
              uses: str, 
              steps: str, 
              git_link: str, 
              colab_link: str, 
              workbench_link: str
             ):
    """
    Add a discoverability index for this notebook
    
        path: The path to the notebook
        tags: The tags (if any) for the notebook
        title: The H1 title for the notebook
        desc: The notebook description
        uses: The resources/services used by the notebook
        steps: The steps specified by the notebook
        git_link: The link to the notebook in the git repo
        colab_link: Link to launch notebook in Colab
        workbench_link: Link to launch notebook in Workbench
        linkbacks: The linkbacks per tag
    """
    global last_tag
    
    if not args.web and not args.repo:
        return
    
    title = title.split(':')[-1].strip()
    title = title[0].upper() + title[1:]
    if args.web:
        title = title.replace('`', '')
        
        print('    <tr>')
        print('        <td>')
        for tag in tags:
            print(f'            {tag.strip()}<br/>\n')
        print('        </td>')
        print('        <td>')
        print(f'            <b>{title}</b><br/>\n')
        if args.desc:
            desc = desc.replace('`', '')
            print('<br/>')
            print(f'            {desc}<br/>\n')
            
        if args.steps:
            print('<br/>' + steps.replace('\n', '<br/>').replace('-', '&nbsp;&nbsp;-').replace('*', '&nbsp;&nbsp;-') +  '<br/>')
            
        if linkbacks:
            num = len(tags)
            for _ in range(num):
                if linkbacks[_].startswith("vertex-ai"):
                    print(f'<br/>            Learn more about <a href="https://cloud.google.com/{linkbacks[_]}">{tags[_]}</a>\n')
                else:
                    print(f'<br/>            Learn more about <a href="{linkbacks[_]}">{tags[_]}</a>\n')

        print('        </td>')
        print('        <td>')
        if colab_link:
            print(f'            <a href="{colab_link}" target="_blank">Colab</a><br/>\n')
        if git_link:
            print(f'            <a href="{git_link}" target="_blank">GitHub</a><br/>\n')
        if workbench_link:
            print(f'            <a href="{workbench_link}" target="_blank">Vertex AI Workbench</a><br/>\n')
        print('        </td>')
        print('    </tr>\n')
    elif args.repo:
        if tags != last_tag and tag != '':
            last_tag = tags
            flat_list = ''
            for item in tags:
                flat_list += item.replace("'", '') + ' '
            print(f"\n### {flat_list}\n")
        print(f"\n[{title}]({git_link})\n")
    
        print("```")
        if args.desc:
            print(desc)

        if args.uses:
            print(uses)

        if args.steps:
            print(steps.rstrip() + '\n')
        print("```\n")


# Instantiate the rules
copyright = CopyrightRule()
notices = NoticesRule()
title = TitleRule()
links = LinksRule()
testenv = TestEnvRule()
table = TableRule()
overview = OverviewRule()
objective = ObjectiveRule()
recommendations = RecommendationsRule()
dataset = DatasetRule()
costs = CostsRule()
setuplocal = SetupLocalRule()
helpers = HelpersRule()
installation = InstallationRule()
restart = RestartRule()
versions = VersionsRule()
beforebegin = BeforeBeginRule()
enableapis = EnableAPIsRule()
setupproject = SetupProjectRule()

 # Cell Validation
rules = [ copyright, notices, title, links, testenv, table, overview, objective,
          recommendations, dataset, costs, setuplocal, helpers,
          installation, restart, versions, beforebegin, enableapis,
          setupproject
]

if args.web:
    print('<style>')
    print('table, th, td {')
    print('  border: 1px solid black;')
    print('  padding-left:10px')
    print('}')
    print('</style>')
    print('<table>')
    print('    <th width="180px">Services</th>')
    print('    <th>Description</th>')
    print('    <th width="80px">Open in</th>')

if args.notebook_dir:
    if not os.path.isdir(args.notebook_dir):
        print(f"Error: not a directory: {args.notebook_dir}", file=sys.stderr)
        exit(1)
    exit_code = parse_dir(args.notebook_dir)
elif args.notebook:
    if not os.path.isfile(args.notebook):
        print(f"Error: not a notebook: {args.notebook}", file=sys.stderr)
        exit(1)
    exit_code = parse_notebook(args.notebook, tags=[], linkback=None, rules=rules)
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
                    try:
                        linkback = row[2]
                    except:
                        linkback = None
                    exit_code += parse_notebook(notebook, tags=tags, linkback=linkback, rules=rules)
else:
    print("Error: must specify a directory or notebook", file=sys.stderr)
    exit(1)

if args.web:
    print('</table>\n')

exit(exit_code)
