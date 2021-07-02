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


"""
Main program for PyTorch distributed training.
Adapted from: https://github.com/narumiruna/pytorch-distributed-example
"""

import os
import shutil
import subprocess
import argparse

import torch
import torch.nn.functional as F

from torch import distributed, nn

from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from torchvision import datasets, transforms

def parse_args():

  parser = argparse.ArgumentParser(
      description='PyTorch Image Classification Distributed Multi Replica')

  # Using environment variables for Cloud Storage directories
  # see more details in https://cloud.google.com/vertex-ai/docs/training/code-requirements
  parser.add_argument(
      '--model-dir', default=os.getenv('AIP_MODEL_DIR'), type=str,
      help='a Cloud Storage URI of a directory intended for saving model artifacts')
  parser.add_argument(
      '--tensorboard-log-dir', default=os.getenv('AIP_TENSORBOARD_LOG_DIR'), type=str,
      help='a Cloud Storage URI of a directory intended for saving TensorBoard')

  parser.add_argument(
      '--backend', type=str, default='gloo',
      help='Name of the backend to use.')
  parser.add_argument(
      '--init-method', type=str, default='env://',
      help='URL specifying how to initialize the package.')
  parser.add_argument(
      '--world-size', type=int, default=os.environ.get('WORLD_SIZE', 1),
      help='Number of processes participating in the job.')
  parser.add_argument(
      '--rank', type=int, default=os.environ.get('RANK', 0),
      help='Rank of the current process.')
  parser.add_argument(
      '--epochs', type=int, default=20)
  parser.add_argument(
      '--no-cuda', action='store_true')
  parser.add_argument(
      '-lr', '--learning-rate', type=float, default=1e-3)
  parser.add_argument(
      '--batch-size', type=int, default=128)

  args = parser.parse_args()

  return args

def distributed_is_initialized():
  if distributed.is_available():
    if distributed.is_initialized():
      return True
  return False

class Average(object):

  def __init__(self):
    self.sum = 0
    self.count = 0

  def __str__(self):
    return '{:.6f}'.format(self.average)

  @property
  def average(self):
    return self.sum / self.count

  def update(self, value, number):
    self.sum += value * number
    self.count += number

class Accuracy(object):

  def __init__(self):
    self.correct = 0
    self.count = 0

  def __str__(self):
    return '{:.2f}%'.format(self.accuracy * 100)

  @property
  def accuracy(self):
    return self.correct / self.count

  @torch.no_grad()
  def update(self, output, target):
    pred = output.argmax(dim=1)
    correct = pred.eq(target).sum().item()

    self.correct += correct
    self.count += output.size(0)


class Trainer(object):

  def __init__(self,
      model,
      optimizer,
      train_loader,
      test_loader,
      device,
      model_name
  ):
    self.model = model
    self.optimizer = optimizer
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.device = device
    self.model_name = model_name

  def save(self, model_dir):
    model_path = os.path.join(model_dir, self.model_name)
    torch.save(self.model.state_dict(), model_path)

  def fit(self, epochs, writer):

    for epoch in range(1, epochs + 1):
      train_loss, train_acc = self.train()
      test_loss, test_acc = self.evaluate()

      writer.add_scalar('Loss/train', train_loss.average, epoch)
      writer.add_scalar('Loss/test', test_loss.average, epoch)
      writer.add_scalar('Accuracy/train', train_acc.accuracy, epoch)
      writer.add_scalar('Accuracy/test', test_acc.accuracy, epoch)

      print(
          'Epoch: {}/{},'.format(epoch, epochs),
          'train loss: {}, train acc: {},'.format(train_loss, train_acc),
          'test loss: {}, test acc: {}.'.format(test_loss, test_acc),
      )

  def train(self):

    self.model.train()

    train_loss = Average()
    train_acc = Accuracy()

    for data, target in self.train_loader:
      data = data.to(self.device)
      target = target.to(self.device)

      output = self.model(data)
      loss = F.cross_entropy(output, target)

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      train_loss.update(loss.item(), data.size(0))
      train_acc.update(output, target)

    return train_loss, train_acc

  @torch.no_grad()
  def evaluate(self):
    self.model.eval()

    test_loss = Average()
    test_acc = Accuracy()

    for data, target in self.test_loader:
      data = data.to(self.device)
      target = target.to(self.device)

      output = self.model(data)
      loss = F.cross_entropy(output, target)

      test_loss.update(loss.item(), data.size(0))
      test_acc.update(output, target)

    return test_loss, test_acc


class Net(nn.Module):

  def __init__(self):
    super(Net, self).__init__()
    self.fc = nn.Linear(784, 10)

  def forward(self, x):
    return self.fc(x.view(x.size(0), -1))


class MNISTDataLoader(data.DataLoader):

  def __init__(
      self,
      data_dir,
      batch_size,
      train=True
  ):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    dataset = datasets.MNIST(data_dir, train=train, transform=transform, download=True)
    sampler = None
    if train and distributed_is_initialized():
      sampler = data.DistributedSampler(dataset)

    super(MNISTDataLoader, self).__init__(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
    )

def main():

  args = parse_args()

  local_data_dir = './tmp/data'
  local_model_dir = './tmp/model'
  local_tensorboard_log_dir = './tmp/logs'

  model_dir = args.model_dir if args.model_dir else local_model_dir
  tensorboard_log_dir = args.tensorboard_log_dir if args.tensorboard_log_dir else local_tensorboard_log_dir

  writer = SummaryWriter(local_tensorboard_log_dir)

  if args.world_size > 1:
    distributed.init_process_group(
        backend=args.backend,
        init_method=args.init_method,
        world_size=args.world_size,
        rank=args.rank,
    )

  device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
  print(f'Device: {device}')

  model = Net()

  if distributed_is_initialized():
    model.to(device)
    model = nn.parallel.DistributedDataParallel(model)
  else:
    model = nn.DataParallel(model)
    model.to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

  train_loader = MNISTDataLoader(local_data_dir, args.batch_size, train=True)
  test_loader = MNISTDataLoader(local_data_dir, args.batch_size, train=False)

  trainer = Trainer(
      model=model,
      optimizer=optimizer,
      train_loader=train_loader,
      test_loader=test_loader,
      device=device,
      model_name='mnist.pt'
  )
  trainer.fit(args.epochs, writer)

  if os.path.exists(local_model_dir) and os.path.isdir(local_model_dir):
    shutil.rmtree(local_model_dir)
  os.makedirs(local_model_dir)
  trainer.save(local_model_dir)
  print(f'Model is saved to {local_model_dir}')
  if args.model_dir:
    subprocess.run(['gsutil', 'cp', '-r', local_model_dir, os.path.dirname(args.model_dir)])
    print(f'Model is uploaded to {args.model_dir}')

  print(f'Tensorboard logs are saved to: {local_tensorboard_log_dir}')
  if args.tensorboard_log_dir:
    subprocess.run(['gsutil', 'cp', '-r', local_tensorboard_log_dir, os.path.dirname(args.tensorboard_log_dir)])
    print(f'Tensorboard logs are uploaded to {args.tensorboard_log_dir}')

  writer.close()

  return

if __name__ == '__main__':
  main()