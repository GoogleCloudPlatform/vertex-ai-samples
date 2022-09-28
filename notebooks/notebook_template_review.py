
import argparse
import json
import os
import urllib.request
import csv

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
parser.add_argument('--desc', dest='desc',
                    default=False, type=bool, help='Output description')
parser.add_argument('--uses', dest='uses',
                    default=False, type=bool, help='Output uses (resources)')
parser.add_argument('--steps', dest='steps',
                    default=False, type=bool, help='Ouput steps')
args = parser.parse_args()

if args.errors_codes:
    args.errors_codes = args.errors_codes.split(',')
    args.errors = True

if args.errors_csv:
    args.errors = True

def parse_dir(directory):
    entries = os.scandir(directory)
    for entry in entries:
        if entry.is_dir():
            if entry.name[0] == '.':
                continue
            if entry.name == 'src' or entry.name == 'images':
                continue
            print("\n##", entry.name, "\n")
            parse_dir(entry.path)
        elif entry.name.endswith('.ipynb'):
            parse_notebook(entry.path)

def parse_notebook(path):
    with open(path, 'r') as f:
        try:
            content = json.load(f)
        except:
            print("Corrupted notebook:", path)
            return
        
        cells = content['cells']
        
        # cell 1 is copyright
        nth = 0
        cell, nth = get_cell(path, cells, nth)
        if not 'Copyright' in cell['source'][0]:
            report_error(path, 0, "missing copyright cell")
            
        # check for notices
        cell, nth = get_cell(path, cells, nth)
        if cell['source'][0].startswith('This notebook'):
            cell, nth = get_cell(path, cells, nth)
            
        # cell 2 is title and links
        if not cell['source'][0].startswith('# '):
            report_error(path, 1, "title cell must start with H1 heading")
            title = ''
        else:
            title = cell['source'][0][2:].strip()
            check_sentence_case(path, title)
            
            # H1 title only
            if len(cell['source']) == 1:
                cell, nth = get_cell(path, cells, nth)
           
        # check links.
        source = ''
        for line in cell['source']:
            source += line
            if '<a href="https://github.com' in line:
                link = line.strip()[9:-2].replace('" target="_blank', '')
                try:
                    code = urllib.request.urlopen(link).getcode()
                except Exception as e:
                    report_error(path, 7, f"bad GitHub link: {link}")
            if '<a href="https://colab.research.google.com/' in line:
                link = 'https://github.com/' + line.strip()[50:-2].replace('" target="_blank', '')
                try:
                    code = urllib.request.urlopen(link).getcode()
                except Exception as e:
                    report_error(path, 8, f"bad Colab link: {link}")
            if '<a href="https://console.cloud.google.com/vertex-ai/workbench/' in line:
                link = line.strip()[91:-2].replace('" target="_blank', '')
                try:
                    code = urllib.request.urlopen(link).getcode()
                except Exception as e:
                    report_error(path, 9, f"bad Workbench link: {link}")

        if 'View on GitHub' not in source:
            report_error(path, 4, 'Missing link for GitHub')
        if 'Open in Vertex AI Workbench' not in source:
            report_error(path, 5, 'Missing link for Workbench')
        if 'master' in source:
            report_error(path, 6, 'Outdated branch (master) used in link')
            
        # Overview
        cell, nth = get_cell(path, cells, nth)
        if not cell['source'][0].startswith("## Overview"):
            report_error(path, 11, "Overview section not found")
            
        # Objective
        cell, nth = get_cell(path, cells, nth)
        if not cell['source'][0].startswith("### Objective"):
            report_error(path, 12, "Objective section not found")
            costs = []
        else:
            desc, uses, steps, costs = parse_objective(path, cell)
            add_index(path, title, desc, uses, steps)
            
        # (optional) Recommendation
        cell, nth = get_cell(path, cells, nth)
        if cell['source'][0].startswith("### Recommendations"):
            cell, nth = get_cell(path, cells, nth)
            
        # Dataset
        if not cell['source'][0].startswith("### Dataset") and not cell['source'][0].startswith("### Model") and not cell['source'][0].startswith("### Embedding"):
            report_error(path, 13, "Dataset/Model section not found")
            
        # Costs
        cell, nth = get_cell(path, cells, nth)
        if not cell['source'][0].startswith("### Costs"):
            report_error(path, 14, "Costs section not found")
        else:
            text = ''
            for line in cell['source']:
                text += line
            if 'BQ' in costs and 'BigQuery' not in text:
                report_error(path, 20, 'Costs section missing reference to BiqQuery')
            if 'Vertex' in costs and 'Vertex' not in text:
                report_error(path, 20, 'Costs section missing reference to Vertex')
            if 'Dataflow' in costs and 'Dataflow' not in text:    
                report_error(path, 20, 'Costs section missing reference to Dataflow')
                
        # (optional) Setup local environment
        cell, nth = get_cell(path, cells, nth)
        if cell['source'][0].startswith('### Set up your local development environment'):
            cell, nth = get_cell(path, cells, nth)
            if cell['source'][0].startswith('**Otherwise**, make sure your environment meets'):
                cell, nth = get_cell(path, cells, nth)
                
        # (optional) Helper functions
        if 'helper' in cell['source'][0]:
            cell, nth = get_cell(path, cells, nth)
            cell, nth = get_cell(path, cells, nth)
                
        # Installation
        if not cell['source'][0].startswith("## Install"):
            if cell['source'][0].startswith("### Install"):
                report_error(path, 27, "Installation section needs to be H2 heading")
            else:
                report_error(path, 21, "Installation section not found")
        else:
            cell, nth = get_cell(path, cells, nth)
            if cell['cell_type'] != 'code':
                report_error(path, 22, "Installation code section not found")
            else:
                if cell['source'][0].startswith('! mkdir'):
                    cell, nth = get_cell(path, cells, nth)
                if 'requirements.txt' in cell['source'][0]:
                    cell, nth = get_cell(path, cells, nth)
                    
                text = ''
                for line in cell['source']:
                    text += line
                    if 'pip ' in line:
                        if 'pip3' not in line:
                            report_error(path, 23, "Installation code section: use pip3")
                        if line.endswith('\\\n'):
                            continue
                        if '-q' not in line:
                            report_error(path, 23, "Installation code section: use -q with pip3")
                        if 'USER_FLAG' not in line and 'sh(' not in line:
                            report_error(path, 23, "Installation code section: use {USER_FLAG} with pip3")
                if 'if IS_WORKBENCH_NOTEBOOK:' not in text:
                    report_error(path, 24, "Installation code section out of date (see template)")
            
        # Restart kernel
        while True:
            cont = False
            cell, nth = get_cell(path, cells, nth)
            for line in cell['source']:
                if 'pip' in line:
                    report_error(path, 25, f"All pip installations must be in a single code cell: {line}")
                    cont = True
                    break
            if not cont:
                break
           
        if not cell['source'][0].startswith("### Restart the kernel"):
            report_error(path, 26, "Restart the kernel section not found")
        else:
            cell, nth = get_cell(path, cells, nth) # code cell
            if cell['cell_type'] != 'code':
                report_error(path, 28, "Restart the kernel code section not found")
                
        # (optional) Check package versions
        cell, nth = get_cell(path, cells, nth)
        if cell['source'][0].startswith('#### Check package versions'):
            cell, nth = get_cell(path, cells, nth) # code cell
            cell, nth = get_cell(path, cells, nth) # next text cell
            
        # Before you begin
        if not cell['source'][0].startswith("## Before you begin"):
            report_error(path, 29, "Before you begin section not found")
        else:
            # maybe one or two cells
            if len(cell['source']) < 2:
                cell, nth = get_cell(path, cells, nth)
                if not cell['source'][0].startswith("### Set up your Google Cloud project"):
                    report_error(path, 30, "Before you begin section incomplete")
              
        # (optional) enable APIs
        cell, nth = get_cell(path, cells, nth)
        if cell['source'][0].startswith("### Enable APIs"):
            cell, nth = get_cell(path, cells, nth) # code cell
            cell, nth = get_cell(path, cells, nth) # next text cell
            
        # Set project ID
        if not cell['source'][0].startswith('#### Set your project ID'):
            report_error(path, 31, "Set project ID section not found")
        else: 
            cell, nth = get_cell(path, cells, nth)
            if cell['cell_type'] != 'code':
                report_error(path, 32, "Set project ID code section not found")
            elif not cell['source'][0].startswith('PROJECT_ID = "[your-project-id]"'):
                report_error(path, 33, f"Set project ID not match template: {line}")
            
            cell, nth = get_cell(path, cells, nth)
            if cell['cell_type'] != 'code' or 'or PROJECT_ID == "[your-project-id]":' not in cell['source'][0]:
                report_error(path, 33, f"Set project ID not match template: {line}")  
            
            cell, nth = get_cell(path, cells, nth)
            if cell['cell_type'] != 'code' or '! gcloud config set project' not in cell['source'][0]:
                report_error(path, 33, f"Set project ID not match template: {line}")   
            
        '''
        # Region
        cell, nth = get_cell(path, cells, nth)
        if cell['source'][0].startswith("### Region"): 
            report_error(path, 34, "Region section not found")
        '''


def get_cell(path, cells, nth):
    while empty_cell(path, cells, nth):
        nth += 1
        
    cell = cells[nth]
    if cell['cell_type'] == 'markdown':
        check_text_cell(path, cell)
    return cell, nth + 1


def empty_cell(path, cells, nth):
    if len(cells[nth]['source']) == 0:
        report_error(path, 10, f'empty cell: cell #{nth}')
        return True
    else:
        return False

def check_text_cell(path, cell):
    
    branding = {
        'Vertex SDK': 'Vertex AI SDK',
        'Vertex Training': 'Vertex AI Training',
        'Vertex Prediction': 'Vertex AI Prediction',
        'Vertex Batch Prediction': 'Vertex AI Batch Prediction',
        'Vertex XAI': 'Vertex Explainable AI',
        'Vertex Experiments': 'Vertex AI Experiments',
        'Vertex TensorBoard': 'Vertex AI TensorBoard',
        'Vertex Pipelines': 'Vertex AI Pipelines',
        'Vertex Hyperparameter Tuning': 'Vertex AI Hyperparameter Tuning',
        'Vertex Metadata': 'Vertex ML Metadata',
        'Vertex AI Metadata': 'Vertex ML Metadata',
        'Vertex Vizier': 'Vertex AI Vizier',
        'Vertex Dataset': 'Vertex AI Dataset',
        'Vertex Model': 'Vertex AI Model',
        'Vertex Endpoint': 'Vertex AI Endpoint',
        'Vertex Private Endpoint': 'Vertex AI Private Endpoint',
        'Tensorflow': 'TensorFlow',
        'Tensorboard': 'TensorBoard',
        'Google Cloud Notebooks': 'Vertex AI Workbench Notebooks',
        'Bigquery': 'BigQuery',
        'Pytorch': 'PyTorch',
        'Sklearn': 'scikit-learn'
    }
    
    for line in cell['source']:
        if 'TODO' in line:
            report_error(path, 14, f'TODO in cell: {line}')
        if 'we ' in line.lower() or "let's" in line.lower() in line.lower():
            report_error(path, 15, f'Do not use first person (e.g., we), replace with 2nd person (you): {line}')
        if 'will' in line.lower() or 'would' in line.lower():
            report_error(path, 16, f'Do not use future tense (e.g., will), replace with present tense: {line}')
            
        for mistake, brand in branding.items():
            if mistake in line:
                report_error(path, 27, f"Branding {brand}: {line}")


def check_sentence_case(path, heading):
    words = heading.split(' ')
    if not words[0][0].isupper():
        report_error(path, 2, f"heading must start with capitalized word: {words[0]}")
        
    for word in words[1:]:
        word = word.replace(':', '').replace('(', '').replace(')', '')
        if word in ['E2E', 'Vertex', 'AutoML', 'ML', 'AI', 'GCP', 'API', 'R', 'CMEK', 'TFX', 'TFDV', 'SDK',
                    'VM', 'CPR', 'NVIDIA', 'ID', 'DASK', 'ARIMA_PLUS', 'KFP', 'I/O']:
            continue
        if word.isupper():
            report_error(path, 3, f"heading is not sentence case: {word}")


def report_error(notebook, code, msg):
    if args.errors:
        if args.errors_codes:
            if str(code) not in args.errors_codes:
                return
            
        if args.errors_csv:
            print(notebook, ',', code)
        else:
            print(f"{notebook}: ERROR ({code}): {msg}")

def parse_objective(path, cell):
    desc = ''
    in_desc = True
    uses = ''
    in_uses = False
    steps = ''
    in_steps = False
    costs = []
    
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
        report_error(path, 17, "Objective section missing desc")
        
    if uses == '':
        report_error(path, 18, "Objective section missing uses services list")
    else:
        if 'BigQuery' in uses:
            costs.append('BQ')
        if 'Vertex' in uses:
            costs.append('Vertex')
        if 'Dataflow' in uses:
            costs.append('Dataflow')
            
    if steps == '':
        report_error(path, 19, "Objective section missing steps list")
            
    return desc, uses, steps, costs

def add_index(path, title, desc, uses, steps):
    if not args.desc and not args.uses and not args.steps:
        return
    
    title = title.split(':')[-1].strip()
    title = title[0].upper() + title[1:]
    
    print(f"\n[{title}]({path})\n")
    
    if args.desc:
        print(desc)
        
    if args.uses:
        print(uses)
        
    if args.steps:
        print(steps)


if args.notebook_dir:
    if not os.path.isdir(args.notebook_dir):
        print("Error: not a directory:", args.notebook_dir)
        exit(1)
    parse_dir(args.notebook_dir)
elif args.notebook:
    if not os.path.isfile(args.notebook):
        print("Error: not a notebook:", args.notebook)
        exit(1)
    parse_notebook(args.notebook)
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
                    parse_notebook(notebook)

        


else:
    print("Error: must specify a directory or notebook")
    exit(1)
