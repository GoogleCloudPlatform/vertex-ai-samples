"""Backbones configurations."""
import dataclasses
from typing import Optional

from official.modeling import hyperparams


@dataclasses.dataclass
class HubModel(hyperparams.Config):
  """Tf-hub model config."""
  handle: Optional[str] = None
  trainable: bool = True
  mean_rgb: Optional[float] = None
  stddev_rgb: Optional[float] = None
  signature: Optional[str] = None
  output_key: Optional[str] = None


@dataclasses.dataclass
class Backbone(hyperparams.OneOfConfig):
  """Configuration for backbones.

  Attributes:
    type: The type of a backbone, such as 'hub_model'.
    hub_model: hub model backbone config.
  """
  type: Optional[str] = 'hub_model'
  hub_model: HubModel = dataclasses.field(default_factory=HubModel)
