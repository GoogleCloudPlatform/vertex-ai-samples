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

"""Train resnet on multiple GPUs with DDP."""

import argparse
import functools
import itertools
import math
import os
import time

import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torchmetrics
from torchvision.models import resnet50
from torchvision.transforms import transforms
import webdataset as wds


def wds_split(src, rank, world_size):
  """Shards split function for webdataset."""
  # The context of caller of this function is within multiple processes
  # (by DDP world_size) and multiple workers (by dataloader_num_workers).
  # So we totally have (world_size * num_workers) workers for processing data.
  # NOTE: Raw data should be sharded to enough shards to make sure one process
  # can handle at least one shard, otherwise the process may hang.
  worker_id = 0
  num_workers = 1
  worker_info = torch.utils.data.get_worker_info()
  if worker_info:
    worker_id = worker_info.id
    num_workers = worker_info.num_workers
  for s in itertools.islice(src, rank * num_workers + worker_id, None,
                            world_size * num_workers):
    yield s


def identity(x):
  return x


def create_wds_dataloader(rank, args, mode):
  """Create webdataset dataset and dataloader."""
  if mode == 'train':
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    data_path = args.train_data_path
    data_size = args.train_data_size
    batch_size_local = args.train_batch_size
    batch_size_global = args.train_batch_size * args.gpus
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
    data_path = args.eval_data_path
    data_size = args.eval_data_size
    batch_size_local = args.eval_batch_size
    batch_size_global = args.eval_batch_size * args.gpus
    # Since webdataset disallows partial batch, we drop the last batch for eval.
    batches = int(data_size / batch_size_global)

  dataset = wds.DataPipeline(
      wds.SimpleShardList(data_path),
      functools.partial(wds_split, rank=rank, world_size=args.gpus),
      wds.tarfile_to_samples(),
      wds.decode('pil'),
      wds.to_tuple('jpg;png;jpeg cls'),
      wds.map_tuple(transform, identity),
      wds.batched(batch_size_local, partial=False),
  )
  num_workers = args.dataloader_num_workers
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


def train(model, device, dataloader, optimizer):
  model.train()
  for image, target in dataloader:
    image = image.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    pred = model(image)
    # pred.shape (N, C), target.shape (N)
    loss = nn.functional.cross_entropy(pred, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  return loss


def evaluate(model, device, dataloader, metric):
  model.eval()
  with torch.no_grad():
    for image, target in dataloader:
      image = image.to(device, non_blocking=True)
      target = target.to(device, non_blocking=True)
      pred = model(image)
      metric.update(pred, target)
  accuracy = metric.compute()
  metric.reset()
  return accuracy


def worker(gpu, args):
  """Run training and evaluation."""
  # Init process group.
  print(f'Initiating process {gpu}')
  dist.init_process_group(
      backend='nccl',
      init_method='env://',
      world_size=args.gpus,
      rank=gpu)

  # Create model.
  model = resnet50(weights=None)
  torch.cuda.set_device(gpu)
  model.to(args.device)
  model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
  model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

  # Create dataloader.
  train_dataloader = create_wds_dataloader(gpu, args, 'train')
  eval_dataloader = create_wds_dataloader(gpu, args, 'eval')

  # Optimizer.
  optimizer = torch.optim.SGD(model.parameters(), 0.1)

  # Main loop.
  metric = torchmetrics.classification.Accuracy(top_k=1).to(args.device)
  for epoch in range(1, args.epochs + 1):
    if gpu == 0:
      print(f'Running epoch {epoch}')

    start = time.time()
    train(model, args.device, train_dataloader, optimizer)
    end = time.time()
    if gpu == 0:
      print(f'Training finished in {(end - start):>0.3f} seconds')

    start = time.time()
    evaluate(model, args.device, eval_dataloader, metric)
    end = time.time()
    if gpu == 0:
      print(f'Evaluation finished in {(end - start):>0.3f} seconds')

  if gpu == 0:
    print('Done')


def create_args():
  """Create main args."""
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--gpus',
      default=4,
      type=int,
      help='number of gpus to use')
  parser.add_argument(
      '--epochs',
      default=1,
      type=int,
      help='number of total epochs to run')
  parser.add_argument(
      '--dataloader_num_workers',
      default=2,
      type=int,
      help='number of workders for dataloader')
  parser.add_argument(
      '--train_data_path',
      default='',
      type=str,
      help='path to training data')
  parser.add_argument(
      '--train_batch_size',
      default=32,
      type=int,
      help='batch size for training per gpu')
  parser.add_argument(
      '--train_data_size',
      default=50000,
      type=int,
      help='data size for training')
  parser.add_argument(
      '--eval_data_path',
      default='',
      type=str,
      help='path to evaluation data')
  parser.add_argument(
      '--eval_batch_size',
      default=32,
      type=int,
      help='batch size for evaluation per gpu')
  parser.add_argument(
      '--eval_data_size',
      default=50000,
      type=int,
      help='data size for evaluation')
  args = parser.parse_args()
  return args


def main():
  args = create_args()
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '8888'

  args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(f'Launch job on {args.gpus} GPUs with DDP')
  mp.spawn(worker, nprocs=args.gpus, args=(args,))


if __name__ == '__main__':
  main()
