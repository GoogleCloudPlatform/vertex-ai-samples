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

import tensorflow as tf

def get_default_strategy():
  devices = ["device:CPU:0"]
  return tf.distribute.MirroredStrategy(devices)

def get_distribution_mirrored_strategy(num_gpus):

  if num_gpus < 0:
    raise ValueError("`num_gpus` can not be negative.")

  if num_gpus == 0:
    return get_default_strategy()

  num_available_gpu_devices = len(tf.config.list_physical_devices('GPU'))
  print(f'Num GPUs Available: {num_available_gpu_devices}')
  assert num_gpus <= num_available_gpu_devices

  devices = ["device:GPU:%d" % i for i in range(num_gpus)]
  return tf.distribute.MirroredStrategy(devices)
