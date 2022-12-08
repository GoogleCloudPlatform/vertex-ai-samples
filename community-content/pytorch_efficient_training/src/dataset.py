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

import math
import functools
import itertools

from torch.utils import data
import torchvision
from torchvision.transforms import transforms

import webdataset as wds

class ImageFolder(torchvision.datasets.ImageFolder):
    """Class for loading imagenet."""
    def __init__(self, image_list_file, transform=None, target_transform=None):
        self.samples = self._make_dataset(image_list_file)
        self.loader = self._loader

        self.imgs = self.samples
        self.targets = [s[1] for s in self.samples]

        self.transform = transform
        self.target_transform = target_transform

    def _make_dataset(self, image_list_file):
        items = []
        with open(image_list_file, 'r') as f:
            for line in f:
                item = line.strip().split(' ')
                items.append((item[0], int(item[1])))
        return items

    def _loader(self, image_path):
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            return img


def prepare_dataloader(mode, rank, args):
    if mode == 'train':
        # Create train dataloader.
        dataset = torchvision.datasets.ImageFolder(
            args.train_data_path,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]))
        data_path = args.train_data_path
        batch_size = args.batch_size
        shuffle = True
    else:
        # Create eval dataloader.
        dataset = torchvision.datasets.ImageFolder(
            args.val_data_path,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]))
        data_path = args.val_data_path
        batch_size = args.batch_size
        shuffle = False

    num_workers = args.workers
    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False
    else:
        sampler = None
    
    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler)
    if rank == 0:
        print(f'{mode} dataloader | samples: {len(dataloader.dataset)}, '
              f'num workers: {dataloader.num_workers}, '
              f'global batch size: {batch_size * args.ngpus_per_node}, '
              f'batches/epoch: {len(dataloader)}')
    return dataloader


def wds_split(src, rank, world_size):
    """Shards split function for webdataset."""
    # The context of caller of this function is within multiple processes
    # (by DDP world_size) and multiple workers (by workers).
    # So we totally have (world_size * num_workers) workers for processing data.
    # NOTE: Raw data should be sharded to enough shards to make sure one process
    # can handle at least one shard, otherwise the process may hang.
    worker_id = 0
    num_workers = 1
    worker_info = data.get_worker_info()
    if worker_info:
        worker_id = worker_info.id
        num_workers = worker_info.num_workers
    for s in itertools.islice(src, rank * num_workers + worker_id, None,
                              world_size * num_workers):
        yield s


def identity(x):
    return x


def prepare_wds_dataloader(mode, rank, args):
    if mode == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        data_path = args.train_data_path
        data_size = args.data_size
        batch_size_local = args.batch_size
        batch_size_global = args.batch_size * args.ngpus_per_node
        # Since webdataset disallows partial batch, we pad the last batch for train.
        batches = int(math.ceil(data_size / batch_size_global))
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        data_path = args.val_data_path
        data_size = args.data_size
        batch_size_local = args.batch_size
        batch_size_global = args.batch_size * args.ngpus_per_node
        # Since webdataset disallows partial batch, we drop the last batch for eval.
        batches = int(data_size / batch_size_global)

    dataset = wds.DataPipeline(
        wds.SimpleShardList(data_path),
        functools.partial(wds_split, rank=rank, world_size=args.ngpus_per_node),
        wds.tarfile_to_samples(),
        wds.decode('pil'),
        wds.to_tuple('jpg;png;jpeg cls'),
        wds.map_tuple(transform, identity),
        wds.batched(batch_size_local, partial=False),
      )
    num_workers = args.workers
    dataloader = wds.WebLoader(
        dataset=dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True).repeat(nbatches=batches)
    print(f'{mode} dataloader | samples: {data_size}, '
          f'num_workers: {num_workers}, '
          f'local batch size: {batch_size_local}, '
          f'global batch size: {batch_size_global}, '
          f'batches: {batches}')
    return dataloader