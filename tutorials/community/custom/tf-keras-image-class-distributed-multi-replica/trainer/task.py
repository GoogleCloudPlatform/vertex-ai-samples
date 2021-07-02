# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import subprocess
import pathlib

import json

import tensorflow as tf

def parse_args():

  parser = argparse.ArgumentParser(
      description='TF-Keras Image Classification Distributed Multi Replica')

  # Using environment variables for Cloud Storage directories
  # see more details in https://cloud.google.com/vertex-ai/docs/training/code-requirements
  parser.add_argument(
      '--model-dir', default=os.getenv('AIP_MODEL_DIR'), type=str,
      help='a Cloud Storage URI of a directory intended for saving model artifacts')
  parser.add_argument(
      '--checkpoint-dir', default=os.getenv('AIP_CHECKPOINT_DIR'), type=str,
      help='a Cloud Storage URI of a directory intended for saving checkpoints')
  parser.add_argument(
      '--tensorboard-log-dir', default=os.getenv('AIP_TENSORBOARD_LOG_DIR'), type=str,
      help='a Cloud Storage URI of a directory intended for saving TensorBoard')

  parser.add_argument(
      '--machine-count', default=1, type=int, help='number of machine used in training')
  parser.add_argument(
      '--epochs', default=25, type=int, help='number of training epochs')
  parser.add_argument(
      '--batch-size', default=32, type=int, help='mini-batch size')
  parser.add_argument(
      '--img-height', default=180, type=int, help='image height')
  parser.add_argument(
      '--img-width', default=180, type=int, help='image width')
  parser.add_argument(
      '--seed', default=123, type=int, help='seed')
  parser.add_argument(
      '--model-version', default=1, type=int, help='model version')

  args = parser.parse_args()

  return args

def download_data():

  dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
  data_dir = tf.keras.utils.get_file(origin=dataset_url, fname='flower_photos', untar=True)
  data_dir = pathlib.Path(data_dir)

  image_count = len(list(data_dir.glob('*/*.jpg')))
  print(f'Downloaded {image_count} images')

  return data_dir

def load_dataset(data_dir, seed, img_height, img_width, batch_size, validation_split=0.2):

  train_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=validation_split,
      subset="training",
      seed=seed,
      image_size=(img_height, img_width),
      batch_size=batch_size)

  val_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=validation_split,
      subset="validation",
      seed=seed,
      image_size=(img_height, img_width),
      batch_size=batch_size)

  return train_ds, val_ds

def build_model(num_classes):

  model = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(num_classes)
  ])

  model.compile(
      optimizer='adam',
      loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])

  return model

def train(model, train_dataset, validation_dataset, epochs, tensorboard_log_dir, checkpoint_dir):

  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=tensorboard_log_dir,
      update_freq=1
  )
  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=os.path.join(checkpoint_dir, 'cp-{epoch:04d}.ckpt'),
      verbose=1,
      save_weights_only=True,
      save_freq="epoch",
      period=2
  )

  history = model.fit(
      train_dataset,
      epochs=epochs,
      validation_data=validation_dataset,
      callbacks=[tensorboard_callback, checkpoint_callback]
  )

  print('Training accuracy: {acc}, loss: {loss}'.format(
      acc=history.history['accuracy'][-1], loss=history.history['loss'][-1]))
  print('Validation accuracy: {acc}, loss: {loss}'.format(
      acc=history.history['val_accuracy'][-1], loss=history.history['val_loss'][-1]))

  return

def main():

  args = parse_args()

  local_model_dir = './tmp/model'
  local_checkpoint_dir = './tmp/checkpoints'
  local_tensorboard_log_dir = './tmp/logs'

  model_dir = args.model_dir if args.model_dir else local_model_dir
  checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else local_checkpoint_dir
  tensorboard_log_dir = args.tensorboard_log_dir if args.tensorboard_log_dir else local_tensorboard_log_dir

  print(f'Replica count: {args.machine_count}')
  strategy = None
  if args.machine_count > 1:
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    print(f'Number of replicas in sync: {strategy.num_replicas_in_sync}')

  tf_config = os.getenv('TF_CONFIG', None)
  num_workers = 1

  if tf_config:
    tf_config = json.loads(tf_config)
    num_workers = len(tf_config['cluster']['worker'])

  print(f'TF-Config: {tf_config}')
  print(f'Number of workers: {num_workers}')

  global_batch_size = args.batch_size * num_workers
  print(f'Global batch size: {global_batch_size}')

  data_dir = download_data()

  train_ds, val_ds = load_dataset(
      data_dir=data_dir,
      seed=args.seed,
      img_height=args.img_height,
      img_width=args.img_width,
      batch_size=global_batch_size
  )

  class_names = train_ds.class_names
  num_classes = len(class_names)
  print(f'Number of classes: {num_classes}')
  print(f'Class namees: {class_names}')

  if strategy:
    with strategy.scope():
      # Everything that creates variables should be under the strategy scope.
      # In general this is only model construction & `compile()`.
      model = build_model(num_classes=num_classes)
  else:
    model = build_model(num_classes=num_classes)

  train(
      model=model,
      train_dataset=train_ds,
      validation_dataset=val_ds,
      epochs=args.epochs,
      tensorboard_log_dir=tensorboard_log_dir,
      checkpoint_dir=local_checkpoint_dir
  )

  local_model_path = os.path.join(local_model_dir, str(args.model_version))
  model.save(local_model_path)
  print(f'Model version {args.model_version} is saved to {local_model_dir}')
  if args.model_dir:
    subprocess.run(['gsutil', 'cp', '-r', local_model_dir, os.path.dirname(args.model_dir)])
    print(f'Model version {args.model_version} is uploaded to {args.model_dir}')

  print(f'Checkpoints are saved to: {local_checkpoint_dir}')
  if args.checkpoint_dir:
    subprocess.run(['gsutil', 'cp', '-r', local_checkpoint_dir, os.path.dirname(args.checkpoint_dir)])
    print(f'Checkpoints are uploaded to {args.checkpoint_dir}')

  print(f'Tensorboard logs are saved to: {tensorboard_log_dir}')

  return

if __name__ == '__main__':
  main()
