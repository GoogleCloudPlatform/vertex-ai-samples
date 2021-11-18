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

import argparse
import os

import numpy as np
import tensorflow as tf

import distribution_utils

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

  parser.add_argument(
      '--local-mode', action='store_true', help='use local mode when running on your local machine')

  args = parser.parse_args()

  return args

def load_dataset(batch_size):
  (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
  # The `x` arrays are in uint8 and have values in the [0, 255] range.
  # You need to convert them to float32 with values in the [0, 1] range.
  x_train = x_train / np.float32(255)
  y_train = y_train.astype(np.int64)
  train_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(60000).batch(batch_size)
  return train_dataset

def build_model():
  model = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(28, 28)),
      tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10)
  ])
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'])
  return model

def train(
    model,
    train_dataset,
    epochs,
    tensorboard_log_dir,
    checkpoint_dir
):

  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=tensorboard_log_dir,
      update_freq=1
  )
  backup_and_restore_callback = tf.keras.callbacks.experimental.BackupAndRestore(
      backup_dir=checkpoint_dir
  )

  history = model.fit(
      train_dataset,
      epochs=epochs,
      callbacks=[
          tensorboard_callback,
          backup_and_restore_callback,
      ]
  )

  print('Training accuracy: {acc}, loss: {loss}'.format(
      acc=history.history['accuracy'][-1],
      loss=history.history['loss'][-1])
  )

  return

def main():

  args = parse_args()

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

  num_worker, task_type, task_id = distribution_utils.setup()
  print(f'task_type: {task_type}, '
        f'task_id: {task_id}, '
        f'num_worker: {num_worker} \n'
        )

  strategy = distribution_utils.get_strategy(num_worker=num_worker)

  global_batch_size = args.batch_size * num_worker
  print(f'Global batch size: {global_batch_size}')

  train_ds = load_dataset(batch_size=global_batch_size)

  if num_worker > 1:
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_ds = train_ds.with_options(options)

  train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

  with strategy.scope():
    model = build_model()
    latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_ckpt:
      model.load_weights(latest_ckpt)

  train(
      model=model,
      train_dataset=train_ds,
      epochs=args.epochs,
      tensorboard_log_dir=tensorboard_log_dir,
      checkpoint_dir=checkpoint_dir
  )

  model_path = os.path.join(model_dir, str(args.model_version))
  model_path = distribution_utils.write_filepath(model_path, task_type, task_id)
  model.save(model_path)
  print(f'Model version {args.model_version} is saved to {model_dir}')

  distribution_utils.clean_up(task_type, task_id, model_path)

  print(f'Tensorboard logs are saved to: {tensorboard_log_dir}')

  return

if __name__ == '__main__':
  main()
