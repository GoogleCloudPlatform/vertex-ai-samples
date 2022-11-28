# Copyright 2022 Google LLC
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

r"""Main function to shard ImageNet dataset.

Example usage:
python3 -u shard_imagenet.py \
--image_list_file=/home/jupyter/data/imagenet/train_list.txt \
--output_pattern=/home/jupyter/data/imagenet/validation-%06d.tar
"""

import argparse
import os
import random
import webdataset as wds  # version: 0.2.26


# NOTE: only supports writing to local path,
# need gcsfuse mounting if want to write to gcs bucket.
def write_shards(args):
  """Shard individual data files."""
  output_dir = os.path.dirname(args.output_pattern)
  if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

  items = []
  # Image list file is a text file, each line is a pair (image_path, label).
  with open(args.image_list_file, 'r') as f:
    for line in f:
      item = line.strip().split(' ')
      items.append((item[0], int(item[1])))
  # Shuffle items to avoid any large sequences of a single class
  # in the dataset.
  random.shuffle(items)

  def _read_image(image_path):
    with open(image_path, 'rb') as f:
      return f.read()

  with wds.ShardWriter(pattern=args.output_pattern,
                       maxcount=args.max_images_per_shard,
                       maxsize=args.max_bytes_per_shard) as sink:
    for i, (image_path, target) in enumerate(items):
      key = str(i)
      image = _read_image(image_path)
      sample = {'__key__': key, 'jpg': image, 'cls': target}
      sink.write(sample)
    if len(items) != sink.total:
      raise ValueError('Items read {} != items written {}'.format(
          len(items), sink.total))


def create_args():
  """Creates arg parser."""
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--image_list_file',
      default='',
      type=str,
      help='path to image list file')
  parser.add_argument(
      '--output_pattern',
      default='',
      type=str,
      help='the pattern for output shards, like /path/to/train-%06d.tar')
  parser.add_argument(
      '--max_images_per_shard',
      default=10 * 1024,
      type=int,
      help='max number of images per shard')
  parser.add_argument(
      '--max_bytes_per_shard',
      default=300 * 1024 * 1024,
      type=int,
      help='max bytes per shard')
  args = parser.parse_args()
  return args


def main():
  args = create_args()
  write_shards(args)


if __name__ == '__main__':
  main()
