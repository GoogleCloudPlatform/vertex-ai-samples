# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.\n",
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json

import tensorflow as tf

def get_default_strategy():
  devices = ["device:CPU:0"]
  return tf.distribute.MirroredStrategy(devices)

def get_distribution_multiworkermirrored_strategy():
  return tf.distribute.experimental.MultiWorkerMirroredStrategy()

def get_strategy(num_worker):
  if num_worker == 1:
    print("distribution_strategy: tf.distribute.MirroredStrategy")
    return get_default_strategy()
  print("distribution_strategy: tf.distribute.MultiWorkerMirroredStrategy")
  return get_distribution_multiworkermirrored_strategy()

def setup():
  tf_config_string = os.getenv('TF_CONFIG', None)
  task_type, task_id = None, None
  if tf_config_string is None:
    num_worker = 1
  else:
    tf_config = json.loads(tf_config_string)
    num_worker = len(tf_config['cluster'].get('chief', [])) + len(
        tf_config['cluster'].get('worker', []))
    task_type = tf_config['task'].get('type', 'chief')
    task_id = tf_config['task'].get('index', 0)
  return num_worker, task_type, task_id

def _is_chief(task_type, task_id):
  # Note: there are two possible `TF_CONFIG` configuration.
  #   1) In addition to `worker` tasks, a `chief` task type is use;
  #      in this case, this function should be modified to
  #      `return task_type == 'chief'`.
  #   2) Only `worker` task type is used; in this case, worker 0 is
  #      regarded as the chief. The implementation demonstrated here
  #      is for this case.
  # For the purpose of this Colab section, the `task_type is None` case
  # is added because it is effectively run with only a single worker.
  return (task_type == 'worker' and task_id == 0) or task_type is None

def _get_temp_dir(dirpath, task_id):
  base_dirpath = 'workertemp_' + str(task_id)
  temp_dir = os.path.join(dirpath, base_dirpath)
  tf.io.gfile.makedirs(temp_dir)
  return temp_dir

def write_filepath(filepath, task_type, task_id):
  dirpath = os.path.dirname(filepath)
  base = os.path.basename(filepath)
  if not _is_chief(task_type, task_id):
    dirpath = _get_temp_dir(dirpath, task_id)
  return os.path.join(dirpath, base)

def clean_up(task_type, task_id, dir):
  if not _is_chief(task_type, task_id):
    tf.io.gfile.rmtree(os.path.dirname(dir))
