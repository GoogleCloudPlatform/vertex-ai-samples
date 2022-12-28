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
import os
import time

from PIL import Image
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torchmetrics
import torchvision
from torchvision.models import resnet50


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

  # Create train dataloader.
  train_dataset = ImageFolder(
      image_list_file=args.train_data_path,
      transform=torchvision.transforms.Compose([
          torchvision.transforms.RandomResizedCrop(224),
          torchvision.transforms.RandomHorizontalFlip(),
          torchvision.transforms.ToTensor(),
          torchvision.transforms.Normalize(
              mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ]))
  train_sampler = torch.utils.data.distributed.DistributedSampler(
      train_dataset, num_replicas=args.gpus, rank=gpu)
  train_dataloader = torch.utils.data.DataLoader(
      dataset=train_dataset,
      batch_size=args.train_batch_size,
      shuffle=False,
      num_workers=args.dataloader_num_workers,
      pin_memory=True,
      sampler=train_sampler)
  if gpu == 0:
    print(f'Train dataloader | samples: {len(train_dataloader.dataset)}, '
          f'num workers: {train_dataloader.num_workers}, '
          f'global batch size: {args.train_batch_size * args.gpus}, '
          f'batches/epoch: {len(train_dataloader)}')

  # Create eval dataloader.
  eval_dataset = ImageFolder(
      image_list_file=args.eval_data_path,
      transform=torchvision.transforms.Compose([
          torchvision.transforms.Resize(256),
          torchvision.transforms.CenterCrop(224),
          torchvision.transforms.ToTensor(),
          torchvision.transforms.Normalize(
              mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ]))
  eval_sampler = torch.utils.data.distributed.DistributedSampler(
      eval_dataset, num_replicas=args.gpus, rank=gpu)
  eval_dataloader = torch.utils.data.DataLoader(
      dataset=eval_dataset,
      batch_size=args.eval_batch_size,
      shuffle=False,
      num_workers=args.dataloader_num_workers,
      pin_memory=True,
      drop_last=True,
      sampler=eval_sampler)
  if gpu == 0:
    print(f'Eval dataloader | samples: {len(eval_dataloader.dataset)}, '
          f'num workers: {eval_dataloader.num_workers}, '
          f'batch size: {args.eval_batch_size}, '
          f'batches/epoch: {len(eval_dataloader)}')

  # Optimizer.
  optimizer = torch.optim.SGD(model.parameters(), 0.1)

  # Main loop.
  metric = torchmetrics.classification.Accuracy(top_k=1).to(args.device)
  for epoch in range(1, args.epochs + 1):
    if gpu == 0:
      print(f'Running epoch {epoch}')
    train_sampler.set_epoch(epoch)

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
      '--eval_data_path',
      default='',
      type=str,
      help='path to evaluation data')
  parser.add_argument(
      '--eval_batch_size',
      default=32,
      type=int,
      help='batch size for evaluation per gpu')
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
