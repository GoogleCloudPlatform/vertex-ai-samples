
import argparse
import json
import os
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('--notebook-dir', dest='notebook_dir',
                    required=True, type=str, help='Notebook directory')
parser.add_argument('--errors', dest='errors',
                    default=False, type=bool, help='Report errors')
parser.add_argument('--errors-csv', dest='errors_csv',
                    default=False, type=bool, help='Report errors as CSV')
parser.add_argument('--desc', dest='desc',
                    default=False, type=bool, help='Output description')
parser.add_argument('--uses', dest='uses',
                    default=False, type=bool, help='Output uses (resources)')
parser.add_argument('--steps', dest='steps',
                    default=False, type=bool, help='Ouput steps')
args = parser.parse_args()

if not os.path.isdir(args.notebook_dir):
    print("Error: not a directory:", args.notebook_dir)
    exit(1)
    
def parse_dir(directory):
    entries = os.scandir(directory)
    for entry in entries:
        if entry.is_dir():
            if entry.name[0] == '.':
                continue
            if entry.name == 'src' or entry.name == 'images':
                continue
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
        if not cell['source'][0].startswith('# Copyright'):
            report_error(path, 0, "missing copyright cell")
            
        # check for notices
        cell, nth = get_cell(path, cells, nth)
        if cell['source'][0].startswith('This notebook'):
            cell, nth = get_cell(path, cells, nth)
            
        # cell 2 is title and links
        if not cell['source'][0].startswith('# '):
            report_error(path, 1, "title cell must start with H1 heading")
        else:
            title = cell['source'][0][2:].strip()
            check_sentence_case(path, title)
           
        # check links.
        source = ''
        for line in cell['source']:
            source += line
            if '<a href="https://github.com' in line:
                link = line.strip()[9:-2]
                try:
                    code = urllib.request.urlopen(link).getcode()
                except Exception as e:
                    report_error(path, 7, f"bad GitHub link: {link}")
            if '<a href="https://colab.research.google.com/' in line:
                link = 'https://github.com/' + line.strip()[50:-2]
                try:
                    code = urllib.request.urlopen(link).getcode()
                except Exception as e:
                    report_error(path, 8, f"bad Colab link: {link}")
            if '<a href="https://console.cloud.google.com/vertex-ai/workbench/' in line:
                link = line.strip()[91:-2]
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
            
        # (optional) Recommendation
        cell, nth = get_cell(path, cells, nth)
        if cell['source'][0].startswith("### Recommendations"):
            cell, nth = get_cell(path, cells, nth)
            
        # Dataset
        if not cell['source'][0].startswith("### Dataset") and not cell['source'][0].startswith("### Model"):
            report_error(path, 13, "Dataset/Model section not found")

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
    for line in cell['source']:
        if 'TODO' in line:
            report_error(path, 14, f'TODO in cell: {line}')
        if 'we ' in line.lower() or "let's" in line.lower() in line.lower():
            report_error(path, 15, f'Do not use first person (e.g., we), replace with 2nd person (you): {line}')
        if 'will' in line.lower() or 'would' in line.lower():
            report_error(path, 16, f'Do not use future tense (e.g., will), replace with present tense: {line}')


def check_sentence_case(path, heading):
    words = heading.split(' ')
    if not words[0][0].isupper():
        report_error(path, 2, f"heading must start with capitalized word: {words[0]}")
        
    for word in words[1:]:
        word = word.replace(':', '').replace('(', '').replace(')', '')
        if word in ['E2E', 'Vertex', 'AutoML', 'ML', 'AI', 'GCP', 'API', 'R', 'CMEK', 'TFX', 'TFDV', 'SDK',
                    'VM', 'CPR', 'NVIDIA']:
            continue
        if word.isupper():
            report_error(path, 3, f"heading is not sentence case: {word}")
            
            
def report_error(notebook, code, msg):
    if args.errors:
        if args.errors_csv:
            print(notebook, ',', code)
        else:
            print(f"{notebook}: ERROR ({code}): {msg}")
            
        
parse_dir(args.notebook_dir)
