# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import subprocess

def makedirs(model_dir):
  if os.path.exists(model_dir) and os.path.isdir(model_dir):
    shutil.rmtree(model_dir)
  os.makedirs(model_dir)
  return

def gcs_upload(dir, local_dir, gcs_dir, gcsfuse_ready, local_mode):
  if not local_mode and dir == local_dir and not gcsfuse_ready:
    subprocess.run(['gsutil', 'cp', '-r',
                    local_dir,
                    os.path.dirname(gcs_dir)])
    print(f'{local_dir} is uploaded to {gcs_dir}')
  return
