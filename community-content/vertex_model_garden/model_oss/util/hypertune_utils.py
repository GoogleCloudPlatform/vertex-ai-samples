"""Utility functions for Vertex Hyperparameter Tuning Jobs."""

import os

from absl import logging


_ENVIRONMENT_VARIABLE_FOR_TRIAL_ID = 'CLOUD_ML_TRIAL_ID'


def get_trial_id_from_environment() -> str:
  """Gets the trial id from environment variable.

  Returns:
    The trial id from environement or '0' if not found.
  """
  if _ENVIRONMENT_VARIABLE_FOR_TRIAL_ID not in os.environ:
    logging.warning(
        'Environment variable %s not found, return 0 as default trial id.',
        _ENVIRONMENT_VARIABLE_FOR_TRIAL_ID,
    )
  return os.environ.get(_ENVIRONMENT_VARIABLE_FOR_TRIAL_ID, '0')
