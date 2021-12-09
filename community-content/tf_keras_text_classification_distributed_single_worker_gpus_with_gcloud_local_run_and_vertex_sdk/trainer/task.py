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

import tensorflow as tf

from tensorflow.keras import layers, losses
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

import distribution_utils

VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 250

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
      '--num-gpus', default=0, type=int, help='number of gpus')
  parser.add_argument(
      '--epochs', default=25, type=int, help='number of training epochs')
  parser.add_argument(
      '--batch-size', default=16, type=int, help='mini-batch size')
  parser.add_argument(
      '--model-version', default=1, type=int, help='model version')

  parser.add_argument(
      '--local-mode', action='store_true', help='use local mode when running on your local machine')

  args = parser.parse_args()

  return args

def download_data(data_dir):
  """Download data."""

  if not os.path.exists(data_dir):
    os.makedirs(data_dir)

  data_url = "https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz"
  dataset = tf.keras.utils.get_file(
      fname="stack_overflow_16k.tar.gz",
      origin=data_url,
      untar=True,
      cache_dir=data_dir,
      cache_subdir="",
  )
  dataset_dir = os.path.join(os.path.dirname(dataset))

  return dataset_dir


def load_dataset(dataset_dir, batch_size, validation_split=0.2, seed=42):

  train_dir = os.path.join(dataset_dir, 'train')
  test_dir = os.path.join(dataset_dir, 'test')

  raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
      train_dir,
      batch_size=batch_size,
      validation_split=validation_split,
      subset='training',
      seed=seed)

  raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
      train_dir,
      batch_size=batch_size,
      validation_split=validation_split,
      subset='validation',
      seed=seed)

  raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
      test_dir,
      batch_size=batch_size,
  )

  for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(10):
      print("Question: ", text_batch.numpy()[i])
      print("Label:", label_batch.numpy()[i])

  for i, label in enumerate(raw_train_ds.class_names):
    print("Label", i, "corresponds to", label)

  return raw_train_ds, raw_val_ds, raw_test_ds

def build_model(num_classes, loss, optimizer, metrics):
  # vocab_size is VOCAB_SIZE + 1 since 0 is used additionally for padding.
  model = tf.keras.Sequential([
      layers.Embedding(VOCAB_SIZE + 1, 64, mask_zero=True),
      layers.Conv1D(64, 5, padding="valid", activation="relu", strides=2),
      layers.GlobalMaxPooling1D(),
      layers.Dense(num_classes)
  ])

  model.compile(
      loss=loss,
      optimizer=optimizer,
      metrics=metrics)

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
      period=100
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

def get_string_labels(predicted_scores_batch, class_names):
  predicted_labels = tf.argmax(predicted_scores_batch, axis=1)
  predicted_labels = tf.gather(class_names, predicted_labels)
  return predicted_labels

def predict(export_model, class_names, inputs):
  predicted_scores = export_model.predict(inputs)
  predicted_labels = get_string_labels(predicted_scores, class_names)
  return predicted_labels

def main():

  args = parse_args()

  local_data_dir = './tmp/data'

  local_model_dir = './tmp/model'
  local_checkpoint_dir = './tmp/checkpoints'
  local_tensorboard_log_dir = './tmp/logs'

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

  class_names = ['csharp', 'java', 'javascript', 'python']
  class_indices = dict(zip(class_names, range(len(class_names))))
  num_classes = len(class_names)
  print(f' class names: {class_names}')
  print(f' class indices: {class_indices}')
  print(f' num classes: {num_classes}')

  strategy = distribution_utils.get_distribution_mirrored_strategy(
      num_gpus=args.num_gpus)
  print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

  global_batch_size = args.batch_size * strategy.num_replicas_in_sync
  print(f'Global batch size: {global_batch_size}')

  dataset_dir = download_data(local_data_dir)
  raw_train_ds, raw_val_ds, raw_test_ds = load_dataset(dataset_dir, global_batch_size)

  vectorize_layer = TextVectorization(
      max_tokens=VOCAB_SIZE,
      output_mode='int',
      output_sequence_length=MAX_SEQUENCE_LENGTH)

  train_text = raw_train_ds.map(lambda text, labels: text)
  vectorize_layer.adapt(train_text)
  print('The vectorize_layer is adapted')

  def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

  # Retrieve a batch (of 32 reviews and labels) from the dataset
  text_batch, label_batch = next(iter(raw_train_ds))
  first_question, first_label = text_batch[0], label_batch[0]
  print("Question", first_question)
  print("Label", first_label)
  print("Vectorized question:", vectorize_text(first_question, first_label)[0])

  train_ds = raw_train_ds.map(vectorize_text)
  val_ds = raw_val_ds.map(vectorize_text)
  test_ds = raw_test_ds.map(vectorize_text)

  AUTOTUNE = tf.data.AUTOTUNE

  def configure_dataset(dataset):
    return dataset.cache().prefetch(buffer_size=AUTOTUNE)

  train_ds = configure_dataset(train_ds)
  val_ds = configure_dataset(val_ds)
  test_ds = configure_dataset(test_ds)

  print('Build model')
  loss = losses.SparseCategoricalCrossentropy(from_logits=True),
  optimizer = 'adam'
  metrics = ['accuracy']

  with strategy.scope():
    model = build_model(
        num_classes=num_classes,
        loss=loss,
        optimizer=optimizer,
        metrics=metrics,
    )

  train(
      model=model,
      train_dataset=train_ds,
      validation_dataset=val_ds,
      epochs=args.epochs,
      tensorboard_log_dir=tensorboard_log_dir,
      checkpoint_dir=checkpoint_dir
  )

  test_loss, test_accuracy = model.evaluate(test_ds)
  print("Int model accuracy: {:2.2%}".format(test_accuracy))

  with strategy.scope():
    export_model = tf.keras.Sequential(
        [vectorize_layer, model,
         layers.Activation('softmax')])

    export_model.compile(
        loss=losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer='adam',
        metrics=['accuracy'])

  loss, accuracy = export_model.evaluate(raw_test_ds)
  print("Accuracy: {:2.2%}".format(accuracy))

  model_path = os.path.join(model_dir, str(args.model_version))
  model.save(model_path)
  print(f'Model version {args.model_version} is saved to {model_dir}')

  print(f'Tensorboard logs are saved to: {tensorboard_log_dir}')

  print(f'Checkpoints are saved to: {checkpoint_dir}')

  return

if __name__ == '__main__':
  main()