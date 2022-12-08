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
from torchvision import datasets


# NOTE: only supports writing to local path,
# need gcsfuse mounting if want to write to gcs bucket.
def write_shards(args, split='train'):
    """Shard individual data files."""
    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # items = []
    # # Image list file is a text file, each line is a pair (image_path, label).
    # with open(args.image_list_file, 'r') as f:
    #   for line in f:
    #     item = line.strip().split(' ')
    #     items.append((item[0], int(item[1])))
    # # Shuffle items to avoid any large sequences of a single class
    # # in the dataset.
    # random.shuffle(items)

    # We're using the torchvision ImageNet dataset
    # to parse the metadata; however, we will read
    # the compressed images directly from disk (to
    # avoid having to reencode them)
    ds = datasets.ImageNet(args.image_path, split=split)
    nimages = len(ds.imgs)
    print("# nimages", nimages)

    # We shuffle the indexes to make sure that we
    # don't get any large sequences of a single class
    # in the dataset.
    indexes = list(range(nimages))
    random.shuffle(indexes)

    def _read_image(image_path):
        with open(image_path, 'rb') as f:
            return f.read()

    # This is the output pattern under which we write shards.
    output_pattern = os.path.join(output_dir, f"imagenet-{split}-%06d.tar")
    with wds.ShardWriter(pattern=output_pattern,
                         maxcount=args.max_images_per_shard,
                         maxsize=args.max_bytes_per_shard) as sink:
        for i in indexes:            
            # Internal information from the ImageNet dataset
            # instance: the file name and the numerical class.
            image_path, target = ds.imgs[i]
            assert target == ds.targets[i]
            
            # Read the JPEG-compressed image file contents.
            image = _read_image(image_path)
            
            # Construct a unique key from the filename.
            key = os.path.splitext(os.path.basename(image_path))[0]
            xkey = key if args.filekey else "%07d" % i
            sample = {'__key__': xkey, 'jpg': image, 'cls': target}
            sink.write(sample)
        if len(indexes) != sink.total:
            raise ValueError(f'Items read {len(indexes)} != items written {sink.total}')


def create_args():
    """Creates arg parser."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--splits", default="train,val", 
                        help="which splits to write")
    parser.add_argument("--filekey", action="store_true", 
                        help="use file as key (default: index)")
    parser.add_argument('--image_path', default='', type=str,
                        help='path to image directory')
    parser.add_argument('--output_dir', default='', type=str,
                        help='the directory for output shards, like /path/to/shards')
    parser.add_argument('--max_images_per_shard', default=10 * 1024, type=int,
                        help='max number of images per shard')
    parser.add_argument('--max_bytes_per_shard', default=300 * 1024 * 1024, type=int,
                        help='max bytes per shard')
    args = parser.parse_args()
    return args


def main():
    args = create_args()
    splits = args.splits.split(",")
    for split in splits:
        print("# split", split)
        write_shards(args, split)


if __name__ == '__main__':
    main()
