"""Returns a config for a Vision Transformer model without asking for data."""
import ml_collections
from vit_jax.configs import common
from vit_jax.configs import models


def get_config(model: str) -> ml_collections.ConfigDict:
  """Returns default parameters for finetuning ViT `model`."""
  config = common.get_config()

  get_model_config = getattr(models, f'get_{model}_config')
  config.model = get_model_config()

  # These values are often overridden on the command line.
  config.base_lr = 0.03
  config.total_steps = 500
  config.warmup_steps = 100
  config.pp = ml_collections.ConfigDict()
  config.pp.train = 'train'
  config.pp.test = 'test'
  config.pp.resize = 448
  config.pp.crop = 384

  # This value MUST be overridden on the command line.
  config.dataset = ''

  return config