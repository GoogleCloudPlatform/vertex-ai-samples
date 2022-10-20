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
args = parser.parse_args()

if args.errors_codes:
    args.errors_codes = args.errors_codes.split(',')
    args.errors = True

if args.errors_csv:
    args.errors = True


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


# globals
num_errors = 0
last_tag = ''


def parse_dir(directory: str) -> None:
    """
        Recursively walk the specified directory, reviewing each notebook (.ipynb) encountered.
        
        directory: The directory path.
    """
    entries = os.scandir(directory)
    for entry in entries:
        if entry.is_dir():
            if entry.name[0] == '.':
                continue
            if entry.name == 'src' or entry.name == 'images' or entry.name == 'sample_data':
                continue
            print("\n##", entry.name, "\n")
            parse_dir(entry.path)
        elif entry.name.endswith('.ipynb'):
            parse_notebook(entry.path, tag=directory.split('/')[-1], linkback=None)


def parse_notebook(path: str,
                   tag: str,
                   linkback: str) -> None:
    """
        Review the specified notebook.
        
            path: The path to the notebook.
            tag: The associated tag
            linkback: A link back to the web docs
    """
    with open(path, 'r') as f:
        try:
            content = json.load(f)
        except:
            print("Corrupted notebook:", path)
            return
        
        cells = content['cells']
        
        NotebookRule.init(path, content['cells'])
        
        CopyrightRule().validate()
            
        NoticesRule().validate()
        
        TitleRule().validate()

        LinksRule().validate()
        
        OverviewRule().validate()
       
        ObjectiveRule().validate()
            
        RecommendationsRule().validate()

        DatasetRule().validate()

        CostsRule().validate()

        SetupLocalRule().validate()

        HelpersRule().validate()

        InstallationRule().validate()

        RestartRule().validate()
        
        VersionsRule().validate()
        
        BeforeBeginRule().validate()
        
        EnableAPIsRule().validate()
        
        SetupProjectRule().validate()
        if NotebookRule.desc != '':
            add_index(path, 
                      tag, 
                      linkback,
                      NotebookRule.title, 
                      NotebookRule.desc, 
                      NotebookRule.uses, 
                      NotebookRule.steps, 
                      NotebookRule.git_link, 
                      NotebookRule.colab_link, 
                      NotebookRule.workbench_link
            )


class NotebookRule(ABC):
    """
    Abstract class for defining notebook conformance rules
    """
    path = None     # The relative path to the notebook
    cell_index = 0  # The index of the last cell that was parsed (reviewed).
    title = ''      # The H1 title for the notebook
    git_link = None  # The GitHub link
    colab_link = None  # The Colab link
    workbench_link = None  # The Workbench link
    desc = ''  # The description in the objective section
    uses = ''  # The resources used in the objective section
    steps = ''  # The steps in the objective section
    costs = []  # The paid services used as derived from the resources used subsection
    
    @staticmethod
    def init(path: str,
             cells: list) -> None:
        """
        Initialize the static variables in the abstract class, specific to a notebook instance
        
        path: The relative path to the notebook
        cells: The content cells (JSON) for the notebook
        """
        NotebookRule.cell_index = 0
        NotebookRule.path = path
        NotebookRule.cells = cells
        NotebookRule.title = ''
        NotebookRule.git_link = None
        NotebookRule.colab_link = None
        NotebookRule.workbench_link = None
        NotebookRule.desc = ''
        NotebookRule.uses = ''
        NotebookRule.steps = ''
        NotebookRule.costs = ''
    
    @abstractmethod
    def validate(self, cells: list) -> None:
        pass
    
    def get_cell(self) -> list:
        """
            Get the next notebook cell.

            Returns:

            cell: content of the next cell
        """
        while self.empty_cell():
            NotebookRule.cell_index += 1

        cell = NotebookRule.cells[NotebookRule.cell_index]
        if cell['cell_type'] == 'markdown':
            TextTWRule().validate()

        NotebookRule.cell_index += 1
        return cell
    
    def empty_cell(self) -> bool:
        """
            Check for empty cells

            Returns:

            bool: whether cell is empty or not
        """
        if len(NotebookRule.cells[NotebookRule.cell_index]['source']) == 0:
            self.report_error(ErrorCode.ERROR_EMPTY_CELL, f'empty cell: cell #{NotebookRule.cell_index}')
            return True
        else:
            return False
        
    def report_error(self,
                     code: ErrorCode,
                     errmsg: str) -> None:
        """
        Report an error.
            If args.errors_codes set, then only report these errors. Otherwise, all errors.

        code: The error code number.
        errmsg: The error message
        """

        global num_errors

        if args.errors:
            code = code.value[0]
            if args.errors_codes:
                if str(code) not in args.errors_codes:
                    return

            if args.errors_csv:
                print(NotebookRule.path, ',', code)
            else:
                print(f"{NotebookRule.path}: ERROR ({code}): {errmsg}", file=sys.stderr)
                num_errors += 1


class CopyrightRule(NotebookRule):
    def validate(self) -> None:
        """
        Parse the copyright cell
        """
        cell = self.get_cell()
        if not 'Copyright' in cell['source'][0]:
            self.report_error(ErrorCode.ERROR_COPYRIGHT, "missing copyright cell")


class NoticesRule(NotebookRule):
    def validate(self) -> None:
        """
        Parse the (optional) notices cell
        """
        cell = self.get_cell()
        if not cell['source'][0].startswith('This notebook'):
             NotebookRule.cell_index -= 1


class TitleRule(NotebookRule):
    def validate(self) -> None: 
        """
        Parse the title in the links cell
        """
        cell = self.get_cell()
        if not cell['source'][0].startswith('# '):
            self.report_error(ErrorCode.ERROR_TITLE_HEADING, "title cell must start with H1 heading")
            NotebookRule.title = ''
        else:
            NotebookRule.title = cell['source'][0][2:].strip()
            SentenceCaseTWRule().validate(NotebookRule.title)

            # H1 title only
            if len(cell['source']) == 1:
                cell = self.get_cell()


class LinksRule(NotebookRule):
    def validate(self) -> None: 
        """
        Parse the links in the links cell
        """

        cell = NotebookRule.cells[NotebookRule.cell_index - 1]
        source = ''
        git_link = None
        colab_link = None
        workbench_link = None
        for line in cell['source']:
            source += line
            if '<a href="https://github.com' in line:
                git_link = line.strip()[9:-2].replace('" target="_blank', '')
                try:
                    code = urllib.request.urlopen(git_link).getcode()
                except Exception as e:
                    # if new notebook
                    derived_link = os.path.join('https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/', NotebookRule.path)
                    if git_link != derived_link:
                        self.report_error(ErrorCode.ERROR_LINK_GIT_BAD, f"bad GitHub link: {git_link}")

            if '<a href="https://colab.research.google.com/' in line:
                colab_link = 'https://colab.research.google.com/github/' + line.strip()[50:-2].replace('" target="_blank', '')
                try:
                    code = urllib.request.urlopen(colab_link).getcode()
                except Exception as e:
                    # if new notebook
                    derived_link = os.path.join('https://colab.research.google.com/github/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks', NotebookRule.path)
                    if colab_link != derived_link:
                        self.report_error(ErrorCode.ERROR_LINK_COLAB_BAD, f"bad Colab link: {colab_link}")


            if '<a href="https://console.cloud.google.com/vertex-ai/workbench/' in line:
                workbench_link = line.strip()[91:-2].replace('" target="_blank', '')
                try:
                    code = urllib.request.urlopen(workbench_link).getcode()
                except Exception as e:
                    derived_link = os.path.join('https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/main/notebooks/', NotebookRule.path)
                    if colab_link != workbench_link:
                        self.report_error(ErrorCode.ERROR_LINK_WORKBENCH_BAD, f"bad Workbench link: {workbench_link}")

        if 'View on GitHub' not in source or not git_link:
            self.report_error(ErrorCode.ERROR_LINK_GIT_MISSING, 'Missing link for GitHub')
        if 'Run in Colab' not in source or not colab_link:
            self.report_error(ErrorCode.ERROR_LINK_COLAB_MISSING, 'Missing link for Colab')    
        if 'Open in Vertex AI Workbench' not in source or not workbench_link:
            self.report_error(ErrorCode.ERROR_LINK_WORKBENCH_MISSING, 'Missing link for Workbench')
        
        NotebookRule.git_link = git_link
        NotebookRule.colab_link = colab_link
        NotebookRule.workbench_link = workbench_link


class OverviewRule(NotebookRule):
    def validate(self) -> None: 
        """
        Parse the overview cell
        """
        cell = self.get_cell()
        if not cell['source'][0].startswith("## Overview"):
            self.report_error(ErrorCode.ERROR_OVERVIEW_NOTFOUND, "Overview section not found")


class ObjectiveRule(NotebookRule):
    def validate(self) -> None: 
        """
        Parse the objective cell.
            Find the description, uses and steps.
        """
        desc = ''
        uses = ''
        steps = ''
        costs = []

        cell = self.get_cell()
        if not cell['source'][0].startswith("### Objective"):
            self.report_error(ErrorCode.ERROR_OBJECTIVE_NOTFOUND, "Objective section not found")
            return

        in_desc = True
        in_uses = False
        in_steps = False
    
        for line in cell['source'][1:]:
            if line.startswith('This tutorial uses'):
                in_desc = False
                in_steps = False
                in_uses = True
                uses += line
                continue
            elif line.startswith('The steps performed'):
                in_desc = False
                in_uses = False
                in_steps = True
                steps += line
                continue

            if in_desc:
                if len(desc) > 0 and line.strip() == '':
                    in_desc = False
                    continue
                desc += line
            elif in_uses:
                sline = line.strip()
                if len(sline) == 0:
                    uses += '\n'
                else:
                    ch = sline[0]
                    if ch in ['-', '*', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                        uses += line
            elif in_steps:
                sline = line.strip()
                if len(sline) == 0:
                    steps += '\n'
                else:
                    ch = sline[0]
                    if ch in ['-', '*', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                        steps += line
            
        if desc == '':
            self.report_error(ErrorCode.ERROR_OBJECTIVE_MISSING_DESC, "Objective section missing desc")
        else:
            desc = desc.lstrip()
            sentences = desc.split('.')
            if len(sentences) > 1:
                desc = sentences[0] + '.\n'
            if desc.startswith('In this tutorial, you learn') or desc.startswith('In this notebook, you learn'):
                desc = desc[22].upper() + desc[23:]

        if uses == '':
            self.report_error(ErrorCode.ERROR_OBJECTIVE_MISSING_USES, "Objective section missing uses services list")
        else:
            if 'BigQuery' in uses:
                costs.append('BQ')
            if 'Vertex' in uses:
                costs.append('Vertex')
            if 'Dataflow' in uses:
                costs.append('Dataflow')

        if steps == '':
            self.report_error(ErrorCode.ERROR_OBJECTIVE_MISSING_STEPS, "Objective section missing steps list")
            
        NotebookRule.desc = desc
        NotebookRule.uses = uses
        NotebookRule.steps = steps
        NotebookRule.costs = costs


class RecommendationsRule(NotebookRule):
    def validate(self) -> None: 
        """
        Parse the (optional) recommendations cell
        """
        # (optional) Recommendation
        cell = self.get_cell()
        if not cell['source'][0].startswith("### Recommendations"):
            NotebookRule.cell_index -= 1


class DatasetRule(NotebookRule):
    def validate(self) -> None: 
        """
        Parse the dataset cell
        """
        # Dataset
        cell = self.get_cell()
        if not cell['source'][0].startswith("### Dataset") and not cell['source'][0].startswith("### Model") and not cell['source'][0].startswith("### Embedding"):
            self.report_error(ErrorCode.ERROR_DATASET_NOTFOUND, "Dataset/Model section not found")


class CostsRule(NotebookRule):
    def validate(self) -> None: 
        """
        Parse the costs cell
        """
        # Costs
        cell = self.get_cell()
        if not cell['source'][0].startswith("### Costs"):
            self.report_error(ErrorCode.ERROR_COSTS_NOTFOUND, "Costs section not found")
        else:
            text = ''
            for line in cell['source']:
                text += line
            if 'BQ' in NotebookRule.costs and 'BigQuery' not in text:
                self.report_error(ErrorCode.ERROR_COSTS_MISSING, 'Costs section missing reference to BiqQuery')
            if 'Vertex' in NotebookRule.costs and 'Vertex' not in text:
                self.report_error(ErrorCode.ERROR_COSTS_MISSING, 'Costs section missing reference to Vertex')
            if 'Dataflow' in NotebookRule.costs and 'Dataflow' not in text:    
                self.report_error(ErrorCode.ERROR_COSTS_MISSING, 'Costs section missing reference to Dataflow')


class SetupLocalRule(NotebookRule):
    def validate(self) -> None:
        """
        Parse the (optional) setup local environment cell
        """
        cell = self.get_cell()
        if not cell['source'][0].startswith('### Set up your local development environment'):
            NotebookRule.cell_index -= 1
            return
        
        cell = self.get_cell()
        if not cell['source'][0].startswith('**Otherwise**, make sure your environment meets'):
            NotebookRule.cell_index -= 1


class HelpersRule(NotebookRule):
    def validate(self) -> None:
        """
        Parse the (optional) helpers text/code cell
        """
        cell = self.get_cell()
        if 'helper' in cell['source'][0]:
            NotebookRule.cell_index += 1  # text and code
        else:
            NotebookRule.cell_index -= 1


class InstallationRule(NotebookRule):
    def validate(self) -> None:
        """
        Parse the installation cells
        """
        cell = self.get_cell()
        if not cell['source'][0].startswith("## Install"):
            if cell['source'][0].startswith("### Install"):
                self.report_error(ErrorCode.ERROR_INSTALLATION_HEADING, "Installation section needs to be H2 heading")
            else:
                self.report_error(ErrorCode.ERROR_INSTALLATION_NOTFOUND, "Installation section not found")
        else:
            cell = self.get_cell()
            if cell['cell_type'] != 'code':
                self.report_error(ErrorCode.ERROR_INSTALLATION_NOTFOUND, "Installation section not found")
            else:
                if cell['source'][0].startswith('! mkdir'):
                    cell = self.get_cell()
                if 'requirements.txt' in cell['source'][0]:
                    cell = self.get_cell()

                text = ''
                for line in cell['source']:
                    text += line
                    if 'pip ' in line:
                        if 'pip3' not in line:
                            self.report_error(ErrorCode.ERROR_INSTALLATION_PIP3, "Installation code section: use pip3")
                        if line.endswith('\\\n'):
                            continue
                        if '-q' not in line and '--quiet' not in line :
                            self.report_error(ErrorCode.ERROR_INSTALLATION_QUIET, "Installation code section: use -q with pip3")
                        if 'USER_FLAG' not in line and 'sh(' not in line:
                            self.report_error(ErrorCode.ERROR_INSTALLATION_USER_FLAG, "Installation code section: use {USER_FLAG} with pip3")
                if 'if IS_WORKBENCH_NOTEBOOK:' not in text:
                    self.report_error(ErrorCode.ERROR_INSTALLATION_CODE_TEMPLATE, "Installation code section out of date (see template)")


class RestartRule(NotebookRule):
    def validate(self) -> None:
        """
        Parse the restart cells
        """
        # Restart kernel
        cell_index = NotebookRule.cell_index
        while True:
            cont = False
            cell = self.get_cell()
            for line in cell['source']:
                if 'pip' in line:
                    self.report_error(ErrorCode.ERROR_INSTALLATION_SINGLE_PIP3, f"All pip installations must be in a single code cell: {line}")
                    cont = True
                    break
            if not cont:
                break

        if not cell['source'][0].startswith("### Restart the kernel"):
            self.report_error(ErrorCode.ERROR_RESTART_NOTFOUND, "Restart the kernel section not found")
        else:
            cell = self.get_cell()  # code cell
            if cell['cell_type'] != 'code':
                self.report_error(ErrorCode.ERROR_RESTART_CODE_NOTFOUND, "Restart the kernel code section not found")


class VersionsRule(NotebookRule):
    def validate(self) -> None:
        """
        Parse the (optional) package versions code/text cell
        """
        cell = self.get_cell()
        if cell['source'][0].startswith('#### Check package versions'):
            NotebookRule.cell_index += 1
        else:
            NotebookRule.cell_index -= 1


class BeforeBeginRule(NotebookRule):
    def validate(self) -> None:
        """
        Parse the before you begin cell
        """
        cell = self.get_cell()
        if not cell['source'][0].startswith("## Before you begin"):
            self.report_error(ErrorCode.ERROR_BEFOREBEGIN_NOTFOUND, "Before you begin section not found")
        else:
            # maybe one or two cells
            if len(cell['source']) < 2:
                cell = self.get_cell()
                if not cell['source'][0].startswith("### Set up your Google Cloud project"):
                    self.report_error(ErrorCode.ERROR_BEFOREBEGIN_INCOMPLETE, "Before you begin section incomplete")


class EnableAPIsRule(NotebookRule):
    def validate(self) -> None:
        """
        Parse the (optional) enable apis code/text cell
        """
        cell = self.get_cell()
        if cell['source'][0].startswith("### Enable APIs"):
            NotebookRule.cell_index += 1
        else:
            NotebookRule.cell_index -= 1


class SetupProjectRule(NotebookRule):
    def validate(self) -> None:
        """
        Parse the set project cells
        """
        cell = self.get_cell()
        if not cell['source'][0].startswith('#### Set your project ID'):
            self.report_error(ErrorCode.ERROR_PROJECTID_NOTFOUND, "Set project ID section not found")
        else: 
            cell = self.get_cell()
            if cell['cell_type'] != 'code':
                self.report_error(ErrorCode.ERROR_PROJECTID_CODE_NOTFOUND, "Set project ID code section not found")
            elif not cell['source'][0].startswith('PROJECT_ID = "[your-project-id]"'):
                self.report_error(ErrorCode.ERROR_PROJECTID_TEMPLATE, "Set project ID not match template")

            cell = self.get_cell()
            if cell['cell_type'] != 'code' or 'or PROJECT_ID == "[your-project-id]":' not in cell['source'][0]:
                self.report_error(ErrorCode.ERROR_PROJECTID_TEMPLATE, "Set project ID not match template")  

            cell = self.get_cell()
            if cell['cell_type'] != 'code' or '! gcloud config set project' not in cell['source'][0]:
                self.report_error(ErrorCode.ERROR_PROJECTID_TEMPLATE, "Set project ID not match template")


class TextTWRule(NotebookRule):
    def validate(self) -> None:
        """
            Check text cells for technical writing requirements
                1. Product branding names
                2. No future tense
                3. No 1st person

            path: used only for reporting an error
            cell: The text cell to review.
        """
    
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
        cell = NotebookRule.cells[NotebookRule.cell_index]
        
        for line in cell['source']:
            # HTML code
            if '<a ' in line:
                continue

            if 'TODO' in line or 'WIP' in line:
                self.report_error(ErrorCode.ERROR_TWRULE_TODO, f'TODO in cell: {line}')
            if 'we ' in line.lower() or "let's" in line.lower() in line.lower():
                self.report_error(ErrorCode.ERROR_TWRULE_FIRSTPERSON, f'Do not use first person (e.g., we), replace with 2nd person (you): {line}')
            if 'will' in line.lower() or 'would' in line.lower():
                self.report_error(ErrorCode.ERROR_TWRULE_FUTURETENSE, f'Do not use future tense (e.g., will), replace with present tense: {line}')

            for mistake, brand in branding.items():
                if mistake in line:
                    self.report_error(ErrorCode.ERROR_TWRULE_BRANDING, f"Branding {mistake} -> {brand}: {line}")

class SentenceCaseTWRule(NotebookRule):
    def validate(self,
                 heading: str) -> None:
        """
        Check that headings are in sentence case

        path: used only for reporting an error
        heading: the heading to check
        """

        ACRONYMS = ['E2E', 'Vertex', 'AutoML', 'ML', 'AI', 'GCP', 'API', 'R', 'CMEK', 
                    'TF', 'TFX', 'TFDV', 'SDK', 'VM', 'CPR', 'NVIDIA', 'ID', 'DASK', 
                    'ARIMA_PLUS', 'KFP', 'I/O', 'GPU', 'Google', 'TensorFlow', 'PyTorch'
                    ]

        words = heading.split(' ')
        if not words[0][0].isupper():
            self.report_error(ErrorCode.ERROR_HEADING_CAP, f"heading must start with capitalized word: {words[0]}")

        for word in words[1:]:
            word = word.replace(':', '').replace('(', '').replace(')', '')
            if word in ACRONYMS:
                continue
            if word.isupper():
                self.report_error(ErrorCode.ERROR_HEADING_CASE, f"heading is not sentence case: {word}")


def add_index(path: str, 
              tag: str, 
              linkback: str,
              title : str, 
              desc: str, 
              uses: str, 
              steps: str, 
              git_link: str, 
              colab_link: str, 
              workbench_link: str
             ) -> None:
    """
    Add a discoverability index for this notebook
    
        path: The path to the notebook
        tag: The tag (if any) for the notebook
        title: The H1 title for the notebook
        desc:
        uses:
        steps:
        git_link:
        colab_link:
        workbench_link:
        linkback:
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
        tags = tag.split(',')
        for tag in tags:
            print(f'            {tag.strip()}<br/>\n')
        print('        </td>')
        print('        <td>')
        print(f'            {title}<br/>\n')
        if args.desc:
            desc = desc.replace('`', '')
            print(f'            {desc}<br/>\n')
        if linkback:
            text = ''
            for tag in tags:
                text += tag.strip() + ' '
                
            print(f'            Learn more about <a src="https://cloud.google.com/{linkback}">{text}</a><br/>\n')
        print('        </td>')
        print('        <td>')
        if colab_link:
            print(f'            <a src="{colab_link}">Colab</a><br/>\n')
        if git_link:
            print(f'            <a src="{git_link}">GitHub</a><br/>\n')
        if workbench_link:
            print(f'            <a src="{workbench_link}">Vertex AI Workbench</a><br/>\n')
        print('        </td>')
        print('    </tr>\n')
    elif args.repo:
        tags = tag.split(',')
        if tags != last_tag and tag != '':
            last_tag = tags
            flat_list = ''
            for item in tags:
                flat_list += item.replace("'", '') + ' '
            print(f"\n### {flat_list}\n")
        print(f"\n[{title}]({git_link})\n")
    
        if args.desc:
            print(desc)

        if args.uses:
            print(uses)

        if args.steps:
            print(steps)

if args.web:
    print('<table>')
    print('    <th>Vertex AI Feature</th>')
    print('    <th>Description</th>')
    print('    <th>Open in</th>')

if args.notebook_dir:
    if not os.path.isdir(args.notebook_dir):
        print("Error: not a directory:", args.notebook_dir)
        exit(1)
    parse_dir(args.notebook_dir)
elif args.notebook:
    if not os.path.isfile(args.notebook):
        print("Error: not a notebook:", args.notebook)
        exit(1)
    parse_notebook(args.notebook, tag='', linkback=None)
elif args.notebook_file:
    if not os.path.isfile(args.notebook_file):
        print("Error: file does not exist", args.notebook_file)
    else:
        with open(args.notebook_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            heading = True
            for row in reader:
                if heading:
                    heading = False
                else:
                    tag = row[0]
                    notebook = row[1]
                    try:
                        linkback = row[2]
                    except:
                        linkback = None
                    parse_notebook(notebook, tag=tag, linkback=linkback)
else:
    print("Error: must specify a directory or notebook")
    exit(1)

if args.web:
    print('</table>\n')

exit(num_errors)
