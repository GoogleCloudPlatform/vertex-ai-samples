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

import argparse
import os
import shutil

import torch
from torch import distributed
from torch.nn.parallel import DistributedDataParallel
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from torchvision import datasets, transforms

def parse_args():

  parser = argparse.ArgumentParser()

  # Using environment variables for Cloud Storage directories
  # see more details in https://cloud.google.com/vertex-ai/docs/training/code-requirements
  parser.add_argument(
      '--model-dir', default=os.getenv('AIP_MODEL_DIR'), type=str,
      help='a Cloud Storage URI of a directory intended for saving model artifacts')
  parser.add_argument(
      '--tensorboard-log-dir', default=os.getenv('AIP_TENSORBOARD_LOG_DIR'), type=str,
      help='a Cloud Storage URI of a directory intended for saving TensorBoard')
  parser.add_argument(
      '--checkpoint-dir', default=os.getenv('AIP_CHECKPOINT_DIR'), type=str,
      help='a Cloud Storage URI of a directory intended for saving checkpoints')

  parser.add_argument(
      '--backend', type=str, default='gloo',
      help='Use the `nccl` backend for distributed GPU training.'
           'Use the `gloo` backend for distributed CPU training.')
  parser.add_argument(
      '--init-method', type=str, default='env://',
      help='URL specifying how to initialize the package.')
  parser.add_argument(
      '--world-size', type=int, default=os.environ.get('WORLD_SIZE', 1),
      help='The total number of nodes in the cluster. '
           'This variable has the same value on every node.')
  parser.add_argument(
      '--rank', type=int, default=os.environ.get('RANK', 0),
      help='A unique identifier for each node. '
           'On the master worker, this is set to 0. '
           'On each worker, it is set to a different value from 1 to WORLD_SIZE - 1.')
  parser.add_argument(
      '--epochs', type=int, default=20)
  parser.add_argument(
      '--no-cuda', action='store_true')
  parser.add_argument(
      '-lr', '--learning-rate', type=float, default=1e-3)
  parser.add_argument(
      '--batch-size', type=int, default=128)
  parser.add_argument(
      '--local-mode', action='store_true', help='use local mode when running on your local machine')

  args = parser.parse_args()

  return args

def makedirs(model_dir):
  if os.path.exists(model_dir) and os.path.isdir(model_dir):
    shutil.rmtree(model_dir)
  os.makedirs(model_dir)
  return

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

class Net(torch.nn.Module):

  def __init__(self, device):
    super(Net, self).__init__()
    self.fc = torch.nn.Linear(784, 10).to(device)

  def forward(self, x):
    return self.fc(x.view(x.size(0), -1))

class MNISTDataLoader(data.DataLoader):

  def __init__(self, root, batch_size, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    dataset = datasets.MNIST(root, train=train, transform=transform, download=True)
    sampler = None
    if train and distributed_is_initialized():
      sampler = data.DistributedSampler(dataset)

    super(MNISTDataLoader, self).__init__(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
    )

class Trainer(object):

  def __init__(self,
      model,
      optimizer,
      train_loader,
      test_loader,
      device,
      model_name,
      checkpoint_path
  ):
    self.model = model
    self.optimizer = optimizer
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.device = device
    self.model_name = model_name
    self.checkpoint_path = checkpoint_path

  def save(self, model_dir):
    model_path = os.path.join(model_dir, self.model_name)
    torch.save(self.model.state_dict(), model_path)

  def fit(self, epochs, is_chief, writer):

    for epoch in range(1, epochs + 1):

      print('Epoch: {}, Training ...'.format(epoch))
      train_loss, train_acc = self.train()

      if is_chief:
        test_loss, test_acc = self.evaluate()
        writer.add_scalar('Loss/train', train_loss.average, epoch)
        writer.add_scalar('Loss/test', test_loss.average, epoch)
        writer.add_scalar('Accuracy/train', train_acc.accuracy, epoch)
        writer.add_scalar('Accuracy/test', test_acc.accuracy, epoch)
        torch.save(self.model.state_dict(), self.checkpoint_path)

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
      loss = torch.nn.functional.cross_entropy(output, target)

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
      loss = torch.nn.functional.cross_entropy(output, target)

      test_loss.update(loss.item(), data.size(0))
      test_acc.update(output, target)

    return test_loss, test_acc

def main():

  args = parse_args()

  local_data_dir = './tmp/data'
  local_model_dir = './tmp/model'
  local_tensorboard_log_dir = './tmp/logs'
  local_checkpoint_dir = './tmp/checkpoints'

  model_dir = args.model_dir or local_model_dir
  tensorboard_log_dir = args.tensorboard_log_dir or local_tensorboard_log_dir
  checkpoint_dir = args.checkpoint_dir or local_checkpoint_dir

  gs_prefix = 'gs://'
  gcsfuse_prefix = '/gcs/'
  if model_dir and model_dir.startswith(gs_prefix):
    model_dir = model_dir.replace(gs_prefix, gcsfuse_prefix)
  if tensorboard_log_dir and tensorboard_log_dir.startswith(gs_prefix):
    tensorboard_log_dir = tensorboard_log_dir.replace(gs_prefix, gcsfuse_prefix)
  if checkpoint_dir and checkpoint_dir.startswith(gs_prefix):
    checkpoint_dir = checkpoint_dir.replace(gs_prefix, gcsfuse_prefix)

  writer = SummaryWriter(tensorboard_log_dir)

  is_chief = args.rank == 0
  if is_chief:
    makedirs(checkpoint_dir)
    print(f'Checkpoints will be saved to {checkpoint_dir}')

  checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
  print(f'checkpoint_path is {checkpoint_path}')

  if args.world_size > 1:
    print('Initializing distributed backend with {} nodes'.format(args.world_size))
    distributed.init_process_group(
          backend=args.backend,
          init_method=args.init_method,
          world_size=args.world_size,
          rank=args.rank,
      )
    print(f'[{os.getpid()}]: '
          f'world_size = {distributed.get_world_size()}, '
          f'rank = {distributed.get_rank()}, '
          f'backend={distributed.get_backend()} \n', end='')

  if torch.cuda.is_available() and not args.no_cuda:
    device = torch.device('cuda:{}'.format(args.rank))
  else:
    device = torch.device('cpu')

  model = Net(device=device)
  if distributed_is_initialized():
    model.to(device)
    model = DistributedDataParallel(model)

  if is_chief:
    # All processes should see same parameters as they all start from same
    # random parameters and gradients are synchronized in backward passes.
    # Therefore, saving it in one process is sufficient.
    torch.save(model.state_dict(), checkpoint_path)
    print(f'Initial chief checkpoint is saved to {checkpoint_path}')

  # Use a barrier() to make sure that process 1 loads the model after process
  # 0 saves it.
  if distributed_is_initialized():
    distributed.barrier()
    # configure map_location properly
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f'Initial chief checkpoint is saved to {checkpoint_path} with map_location {device}')
  else:
    model.load_state_dict(torch.load(checkpoint_path))
    print(f'Initial chief checkpoint is loaded from {checkpoint_path}')

  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

  train_loader = MNISTDataLoader(
      local_data_dir, args.batch_size, train=True)
  test_loader = MNISTDataLoader(
      local_data_dir, args.batch_size, train=False)

  trainer = Trainer(
      model=model,
      optimizer=optimizer,
      train_loader=train_loader,
      test_loader=test_loader,
      device=device,
      model_name='mnist.pt',
      checkpoint_path=checkpoint_path,
  )
  trainer.fit(args.epochs, is_chief, writer)

  if model_dir == local_model_dir:
    makedirs(model_dir)
    trainer.save(model_dir)
    print(f'Model is saved to {model_dir}')

  print(f'Tensorboard logs are saved to: {tensorboard_log_dir}')

  writer.close()

  if is_chief:
    os.remove(checkpoint_path)

  if distributed_is_initialized():
    distributed.destroy_process_group()

  return

if __name__ == '__main__':
  main()
