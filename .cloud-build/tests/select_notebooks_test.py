import sys
sys.path.append("..")

from execute_changed_notebooks_helper import (load_results, select_notebook)

bucket:str = "cloud-build-notebooks-presubmit"
bucket_file: str = "build_results"

accum = load_results(bucket, bucket_file)

print(accum)

n_select = 0
n_notselect = 0
for notebook in accum:
    if select_notebook(notebook, accum, 20):
        n_select += 1
    else:
        n_notselect += 1

print(f"SELECTED {n_select}, NOT SELECTED {n_notselect}")
