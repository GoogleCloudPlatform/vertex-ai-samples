'''
Viewer for the weekly regression testing of the official notebooks

Cloud Storage location: gs://cloud-build-notebooks-presubmit/build_results/
'''
import argparse
import json
from  util import download_file
import csv
import datetime
from google.cloud import storage

BUILD_BUCKET = "cloud-build-notebooks-presubmit"
BUILD_FOLDER = "build_results"


parser = argparse.ArgumentParser()
parser.add_argument('--file', dest='file',
                    default=None, type=str, help='build results filei (local or GCS)')
args = parser.parse_args()

investigate = {}
with open('investigate.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        investigate[row[0][:-6]] = row[1]

if not args.file:
    client = storage.Client()
    blobs = client.list_blobs(BUILD_BUCKET, prefix=BUILD_FOLDER)
    newest_time = datetime.datetime(2000, 1, 1)
    for blob in blobs:
        # individual PR
        if blob.size < 2000:
            continue
        time_created = blob.time_created.replace(tzinfo=None)
        if time_created > newest_time:
            newest_time = time_created
            args.file = f"gs://{BUILD_BUCKET}/{blob.name}"

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
        if notebook in investigate:
            passed = "INVG"
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
