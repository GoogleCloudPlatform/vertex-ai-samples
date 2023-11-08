'''
Viewer for the weekly regression testing of the official notebooks

Cloud Storage location: gs://cloud-build-notebooks-presubmit/build_results/
'''
import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument('--file', dest='file',
                    default='build.json', type=str, help='build results file')
args = parser.parse_args()

with open(args.file, 'r') as f:
    results = json.load(f)

for item in results.items():
    if item[1]['passed']:
        print(f"{item[0]},PASSED")
    else:
        print(f"{item[0]},FAILED")
