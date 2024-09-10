"""Train Keras Stable Diffusion.

Most the codes below are from
https://keras.io/examples/generative/finetune_stable_diffusion/.
"""
import os

from absl import app
from absl import flags
from absl import logging
import keras_cv
# pylint: disable=g-importing-member
from keras_cv.models.stable_diffusion.clip_tokenizer import SimpleTokenizer
from keras_cv.models.stable_diffusion.diffusion_model import DiffusionModel
from keras_cv.models.stable_diffusion.image_encoder import ImageEncoder
from keras_cv.models.stable_diffusion.noise_scheduler import NoiseScheduler
from keras_cv.models.stable_diffusion.text_encoder import TextEncoder
import numpy as np
# The docker builds could not find pandas.
# pylint: disable=import-error
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow.experimental.numpy as tnp

from util import constants
from util import fileutils

_INPUT_CSV_PATH = flags.DEFINE_string(
    'input_csv_path',
    None,
    'The input csv path.',
    required=True,
)

_USE_MP = flags.DEFINE_bool(
    'use_mp',
    True,
    'Enable mixed-precision training if the underlying GPU has tensor cores.',
)

_EPOCHS = flags.DEFINE_integer('epochs', 1, 'The number of epochs.')

_OUTPUT_MODEL_DIR = flags.DEFINE_string(
    'output_model_dir',
    None,
    'The output model dir.',
    required=True,
)

# These hyperparameters defaults come from this tutorial by Hugging Face:
# https://huggingface.co/docs/diffusers/training/text2image
_LEARNING_RATE = flags.DEFINE_float(
    'learning_rate', 1e-5, 'The learning rate parameter for AdamW optimizer.'
)

_BETA_1 = flags.DEFINE_float(
    'beta_1', 0.9, 'The beta_1 parameter for AdamW optimizer.'
)

_BETA_2 = flags.DEFINE_float(
    'beta_2', 0.999, 'The beta_2 parameter for AdamW optimizer.'
)

_WEIGHT_DECAY = flags.DEFINE_float(
    'weight_decay', 1e-2, 'The weight decay parameter for AdamW optimizer.'
)

_EPSILON = flags.DEFINE_float(
    'epsilon', 1e-08, 'The epsilon parameter for AdamW optimizer.'
)

RESOLUTION = int(os.environ.get('RESOLUTION', 512))

# The padding token and maximum prompt length are specific to the text encoder.
# If you're using a different text encoder be sure to change them accordingly.
PADDING_TOKEN = 49407
MAX_PROMPT_LENGTH = 77

AUTO = tf.data.AUTOTUNE
POS_IDS = tf.convert_to_tensor([list(range(MAX_PROMPT_LENGTH))], dtype=tf.int32)


augmenter = keras.Sequential(
    layers=[
        keras_cv.layers.CenterCrop(RESOLUTION, RESOLUTION),
        keras_cv.layers.RandomFlip(),
        tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1),
    ]
)
text_encoder = TextEncoder(MAX_PROMPT_LENGTH)


def process_image(image_path, tokenized_text):
  image = tf.io.read_file(image_path)
  image = tf.io.decode_png(image, 3)
  image = tf.image.resize(image, (RESOLUTION, RESOLUTION))
  return image, tokenized_text


def apply_augmentation(image_batch, token_batch):
  return augmenter(image_batch), token_batch


def run_text_encoder(image_batch, token_batch):
  return (
      image_batch,
      token_batch,
      text_encoder([token_batch, POS_IDS], training=False),
  )


def prepare_dict(image_batch, token_batch, encoded_text_batch):
  return {
      'images': image_batch,
      'tokens': token_batch,
      'encoded_text': encoded_text_batch,
  }


def prepare_dataset(image_paths, tokenized_texts, batch_size=1):
  dataset = tf.data.Dataset.from_tensor_slices((image_paths, tokenized_texts))
  dataset = dataset.shuffle(batch_size * 10)
  dataset = dataset.map(process_image, num_parallel_calls=AUTO).batch(
      batch_size
  )
  dataset = dataset.map(apply_augmentation, num_parallel_calls=AUTO)
  dataset = dataset.map(run_text_encoder, num_parallel_calls=AUTO)
  dataset = dataset.map(prepare_dict, num_parallel_calls=AUTO)
  return dataset.prefetch(AUTO)


def prepare_training_dataset(dataset_csv):
  """Prepares training datasets."""
  if dataset_csv.startswith(constants.GCS_URI_PREFIX):
    if not os.path.exists(constants.LOCAL_DATA_DIR):
      os.makedirs(constants.LOCAL_DATA_DIR)
    logging.info(
        'Start to download data from %s to %s.',
        os.path.dirname(dataset_csv),
        constants.LOCAL_DATA_DIR,
    )
    fileutils.download_gcs_dir_to_local(
        os.path.dirname(dataset_csv), constants.LOCAL_DATA_DIR
    )
    data_frame = pd.read_csv(
        os.path.join(constants.LOCAL_DATA_DIR, os.path.basename(dataset_csv))
    )
    data_frame['image_path'] = data_frame['image_path'].apply(
        lambda x: os.path.join(constants.LOCAL_DATA_DIR, x)
    )
  else:
    # Keeps the following codes for experiments with
    # https://keras.io/examples/generative/finetune_stable_diffusion/.
    data_path = tf.keras.utils.get_file(origin=dataset_csv, untar=True)
    data_frame = pd.read_csv(os.path.join(data_path, 'data.csv'))
    data_frame['image_path'] = data_frame['image_path'].apply(
        lambda x: os.path.join(data_path, x)
    )
  data_frame.head()

  # Load the tokenizer.
  tokenizer = SimpleTokenizer()

  #  Method to tokenize and pad the tokens.
  def process_text(caption):
    tokens = tokenizer.encode(caption)
    tokens = tokens + [PADDING_TOKEN] * (MAX_PROMPT_LENGTH - len(tokens))
    return np.array(tokens)

  # Collate the tokenized captions into an array.
  tokenized_texts = np.empty((len(data_frame), MAX_PROMPT_LENGTH))

  all_captions = list(data_frame['caption'].values)
  for i, caption in enumerate(all_captions):
    tokenized_texts[i] = process_text(caption)

  # Prepare the dataset.
  training_dataset = prepare_dataset(
      np.array(data_frame['image_path']), tokenized_texts, batch_size=4
  )

  return training_dataset


class Trainer(tf.keras.Model):
  """The trainer for Keras Stable Diffusion."""

  # Reference:
  # https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py

  def __init__(
      self,
      diffusion_model,
      vae,
      noise_scheduler,
      use_mixed_precision=False,
      max_grad_norm=1.0,
      **kwargs,
  ):
    super().__init__(**kwargs)

    self.diffusion_model = diffusion_model
    self.vae = vae
    self.noise_scheduler = noise_scheduler
    self.max_grad_norm = max_grad_norm

    self.use_mixed_precision = use_mixed_precision
    self.vae.trainable = False

  def train_step(self, inputs):
    images = inputs['images']
    encoded_text = inputs['encoded_text']
    batch_size = tf.shape(images)[0]

    with tf.GradientTape() as tape:
      # Project image into the latent space and sample from it.
      latents = self.sample_from_encoder_outputs(
          self.vae(images, training=False)
      )
      # Know more about the magic number here:
      # https://keras.io/examples/generative/fine_tune_via_textual_inversion/
      latents = latents * 0.18215

      # Sample noise that we'll add to the latents.
      noise = tf.random.normal(tf.shape(latents))

      # Sample a random timestep for each image.
      timesteps = tnp.random.randint(
          0, self.noise_scheduler.train_timesteps, (batch_size,)
      )

      # Add noise to the latents according to the noise magnitude at each
      # timestep (this is the forward diffusion process).
      noisy_latents = self.noise_scheduler.add_noise(
          tf.cast(latents, noise.dtype), noise, timesteps
      )

      # Get the target for loss depending on the prediction type
      # just the sampled noise for now.
      target = noise  # noise_schedule.predict_epsilon == True

      # Predict the noise residual and compute loss.
      # pylint: disable=unnecessary-lambda
      timestep_embedding = tf.map_fn(
          lambda t: self.get_timestep_embedding(t), timesteps, dtype=tf.float32
      )
      timestep_embedding = tf.squeeze(timestep_embedding, 1)
      model_pred = self.diffusion_model(
          [noisy_latents, timestep_embedding, encoded_text], training=True
      )
      loss = self.compiled_loss(target, model_pred)
      if self.use_mixed_precision:
        loss = self.optimizer.get_scaled_loss(loss)

    # Update parameters of the diffusion model.
    trainable_vars = self.diffusion_model.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    if self.use_mixed_precision:
      gradients = self.optimizer.get_unscaled_gradients(gradients)
    gradients = [tf.clip_by_norm(g, self.max_grad_norm) for g in gradients]
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    return {m.name: m.result() for m in self.metrics}

  def get_timestep_embedding(self, timestep, dim=320, max_period=10000):
    half = dim // 2
    log_max_preiod = tf.math.log(tf.cast(max_period, tf.float32))
    # The docker builds could not support unary `-`.
    # pylint: disable=invalid-unary-operand-type
    freqs = tf.math.exp(
        -log_max_preiod * tf.range(0, half, dtype=tf.float32) / half
    )
    args = tf.convert_to_tensor([timestep], dtype=tf.float32) * freqs
    embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], 0)
    embedding = tf.reshape(embedding, [1, -1])
    return embedding

  def sample_from_encoder_outputs(self, outputs):
    mean, logvar = tf.split(outputs, 2, axis=-1)
    logvar = tf.clip_by_value(logvar, -30.0, 20.0)
    std = tf.exp(0.5 * logvar)
    sample = tf.random.normal(tf.shape(mean), dtype=mean.dtype)
    return mean + std * sample

  def save_weights(
      self, filepath, overwrite=True, save_format=None, options=None
  ):
    # Overriding this method will allow us to use the `ModelCheckpoint`
    # callback directly with this trainer class. In this case, it will
    # only checkpoint the `diffusion_model` since that's what we're training
    # during fine-tuning.
    self.diffusion_model.save_weights(
        filepath=filepath,
        overwrite=overwrite,
        save_format=save_format,
        options=options,
    )


def main(_) -> None:
  # _INPUT_CSV_PATH and _OUTPUT_MODEL_DIR should have the format as
  # gs://<bucket_name>/<object_name>.
  if _INPUT_CSV_PATH.value:
    if not _INPUT_CSV_PATH.value.startswith(constants.GCS_URI_PREFIX):
      raise ValueError('The input csv path should be a gcs path like gs://<>')
  if _OUTPUT_MODEL_DIR.value:
    if not _OUTPUT_MODEL_DIR.value.startswith(constants.GCS_URI_PREFIX):
      raise ValueError('The output model dir should be a gcs path like gs://<>')

  if _USE_MP.value:
    keras.mixed_precision.set_global_policy('mixed_float16')

  image_encoder = ImageEncoder(RESOLUTION, RESOLUTION)
  diffusion_ft_trainer = Trainer(
      diffusion_model=DiffusionModel(RESOLUTION, RESOLUTION, MAX_PROMPT_LENGTH),
      # Remove the top layer from the encoder, which cuts off the variance and
      # only returns the mean.
      vae=tf.keras.Model(
          image_encoder.input,
          image_encoder.layers[-2].output,
      ),
      noise_scheduler=NoiseScheduler(),
      use_mixed_precision=_USE_MP.value,
  )

  optimizer = tf.keras.optimizers.experimental.AdamW(
      learning_rate=_LEARNING_RATE.value,
      weight_decay=_WEIGHT_DECAY.value,
      beta_1=_BETA_1.value,
      beta_2=_BETA_2.value,
      epsilon=_EPSILON.value,
  )
  diffusion_ft_trainer.compile(optimizer=optimizer, loss='mse')

  training_dataset = prepare_training_dataset(_INPUT_CSV_PATH.value)

  # Note: gcsfuse does not work for Keras. We saves the trained models locally
  # first, and then copy to gcs storages.
  if not os.path.exists(constants.LOCAL_MODEL_DIR):
    os.makedirs(constants.LOCAL_MODEL_DIR)
  # The default saved model is in HDF5.
  ckpt_path = os.path.join(constants.LOCAL_MODEL_DIR, 'saved_model.h5')
  ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
      ckpt_path,
      save_weights_only=True,
      monitor='loss',
      mode='min',
  )
  diffusion_ft_trainer.fit(
      training_dataset, epochs=_EPOCHS.value, callbacks=[ckpt_callback]
  )

  # Copies the files in constants.LOCAL_MODEL_DIR to output_model_dir.
  fileutils.upload_local_dir_to_gcs(
      constants.LOCAL_MODEL_DIR, _OUTPUT_MODEL_DIR.value
  )

  return


if __name__ == '__main__':
  app.run(main)