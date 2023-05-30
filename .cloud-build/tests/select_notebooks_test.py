import sys
sys.path.append("..")

from execute_changed_notebooks_helper import (load_results, select_notebook)

bucket:str = "cloud-build-notebooks-presubmit"
bucket_file: str = "build_results"

accum = load_results(bucket, bucket_file)

print(accum)
