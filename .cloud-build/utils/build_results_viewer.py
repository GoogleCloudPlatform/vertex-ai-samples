'''
Viewer for the weekly regression testing of the official notebooks

Cloud Storage location: gs://cloud-build-notebooks-presubmit/build_results/
'''
import argparse
import json
from  util import download_file


parser = argparse.ArgumentParser()
parser.add_argument('--file', dest='file',
                    default='build.json', type=str, help='build results file')
args = parser.parse_args()

if args.file.startswith("gs://"):
    path = args.file[5:]
    bucket = path.split('/')[0]
    file   = path[len(bucket)+1:]
    download_file(bucket, file, "build.json")
    args.file = "build.json"

with open(args.file, 'r') as f:
    results = json.load(f)

for item in results.items():
    notebook = item[0][len("/notebooks/official/")-1:-6]
    if item[1]['passed']:
        passed = "PASS"
    else:
        passed = "FAIL"
      
    error = item[1]['error_type']
  
    if passed == "FAIL":
        if error == '':
            error = "undetermined"
        if 'log_url' in item[1]:
            log_url = item[1]['log_url']
        else:
            log_url = ''
    else:
        log_url = ''

    print(f"{notebook:75} {passed} {error:10} {log_url}")
