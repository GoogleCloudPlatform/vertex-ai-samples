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

import argparse
import copy
import os
import pathlib
import shutil
import time

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from torchvision import datasets, models, transforms

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
      '--epochs', default=25, type=int,
      help='number of training epochs')
  parser.add_argument(
      '--learning-rate', default=0.001, type=float,
      help='learning rate')
  parser.add_argument(
      '--momentum', default=0.9, type=float,
      help='momentum')
  parser.add_argument(
      '--batch-size', default=4, type=int,
      help='mini-batch size')
  parser.add_argument(
      '--num-workers', default=4, type=int,
      help='number of workers')

  parser.add_argument(
      '--local-mode', action='store_true', help='use local mode when running on your local machine')

  args = parser.parse_args()

  return args

def makedirs(model_dir):
  if os.path.exists(model_dir) and os.path.isdir(model_dir):
    shutil.rmtree(model_dir)
  os.makedirs(model_dir)
  return

def download_data(data_dir):

  dataset_url = 'https://download.pytorch.org/tutorial/hymenoptera_data.zip'
  data_dir = os.path.abspath(data_dir)
  datasets.utils.download_url(
      url=dataset_url,
      root=data_dir
  )

  from_path=os.path.join(data_dir, 'hymenoptera_data.zip')
  datasets.utils.extract_archive(
      from_path=from_path,
      to_path=data_dir,
      remove_finished=True
  )

  dataset_dir = pathlib.Path(os.path.join(data_dir, 'hymenoptera_data'))
  print(f'Data is downloaded to: {dataset_dir}')

  return dataset_dir

def load_dataset(data_dir):
  # Data augmentation and normalization for training
  # Just normalization for validation
  data_transforms = {
      'train': transforms.Compose([
          transforms.RandomResizedCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'val': transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
  }

  image_datasets = {
      x: datasets.ImageFolder(
          os.path.join(data_dir, x),
          data_transforms[x]
      )
      for x in ['train', 'val']
  }

  class_names = image_datasets['train'].classes
  print(f'Class names: {class_names}')
  print(f'Number of classes: {len(class_names)}')

  return image_datasets, class_names

def train(model, criterion, optimizer, scheduler, dataset_sizes, dataloaders, device, epochs, writer):
  since = time.time()

  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0

  for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch, epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
      if phase == 'train':
        model.train()  # Set model to training mode
      else:
        model.eval()   # Set model to evaluate mode

      running_loss = 0.0
      running_corrects = 0

      # Iterate over data.
      for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
          outputs = model(inputs)
          _, preds = torch.max(outputs, 1)
          loss = criterion(outputs, labels)

          # backward + optimize only if in training phase
          if phase == 'train':
            loss.backward()
            optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
      if phase == 'train':
        scheduler.step()

      epoch_loss = running_loss / dataset_sizes[phase]
      epoch_acc = running_corrects.double() / dataset_sizes[phase]

      if phase == 'train':
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)

      if phase == 'val':
        writer.add_scalar('Loss/test', epoch_loss, epoch)
        writer.add_scalar('Accuracy/test', epoch_acc, epoch)

      print('{} Loss: {:.4f} Acc: {:.4f}'.format(
          phase, epoch_loss, epoch_acc))

      # deep copy the model
      if phase == 'val' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())

  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(
      time_elapsed // 60, time_elapsed % 60))
  print('Best val Acc: {:4f}'.format(best_acc))

  # load best model weights
  model.load_state_dict(best_model_wts)
  return model

def load_model(class_names, device, pretrained=True):

  model_ft = models.resnet18(pretrained=pretrained)
  num_ftrs = model_ft.fc.in_features
  model_ft.fc = torch.nn.Linear(num_ftrs, len(class_names))
  model_ft = model_ft.to(device)

  return model_ft

def main():

  args = parse_args()

  local_data_dir = './tmp/data'

  local_model_dir = './tmp/model'
  local_tensorboard_log_dir = './tmp/logs'

  model_dir = args.model_dir or local_model_dir
  tensorboard_log_dir = args.tensorboard_log_dir or local_tensorboard_log_dir

  gs_prefix = 'gs://'
  gcsfuse_prefix = '/gcs/'
  if model_dir and model_dir.startswith(gs_prefix):
    model_dir = model_dir.replace(gs_prefix, gcsfuse_prefix)
  if tensorboard_log_dir and tensorboard_log_dir.startswith(gs_prefix):
    tensorboard_log_dir = tensorboard_log_dir.replace(gs_prefix, gcsfuse_prefix)

  makedirs(model_dir)
  writer = SummaryWriter(tensorboard_log_dir)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f'Device: {device}')

  data_dir = download_data(local_data_dir)
  image_datasets, class_names = load_dataset(data_dir)

  dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
  print(f'Dataset sizes: {dataset_sizes}')

  dataloaders = {
      x: torch.utils.data.DataLoader(
          image_datasets[x],
          batch_size=args.batch_size,
          shuffle=True,
          num_workers=args.num_workers
      )
      for x in ['train', 'val']
  }

  model_ft = load_model(class_names, device)

  criterion = torch.nn.CrossEntropyLoss()

  # Observe that all parameters are being optimized
  optimizer_ft = optim.SGD(
      model_ft.parameters(), lr=args.learning_rate, momentum=args.momentum)

  # Decay LR by a factor of 0.1 every 7 epochs
  exp_lr_scheduler = optim.lr_scheduler.StepLR(
      optimizer_ft, step_size=7, gamma=0.1)

  model = train(
      model=model_ft,
      criterion=criterion,
      optimizer=optimizer_ft,
      scheduler=exp_lr_scheduler,
      dataset_sizes=dataset_sizes,
      dataloaders=dataloaders,
      device=device,
      epochs=args.epochs,
      writer=writer,
  )
  model_name = 'antandbee.pth'
  model_path = os.path.join(model_dir, f'{model_name}')

  torch.save(model.state_dict(), model_path)
  print(f'Model is saved to {model_dir}')

  print(f'Tensorboard logs are saved to: {tensorboard_log_dir}')

  writer.close()

  return

if __name__ == '__main__':
  main()