r"""AutoGluon serving binary.

This module sets up a Flask web server for serving predictions from a
trained AutoGluon model. The server exposes two endpoints:

1.  `/ping`: A health check endpoint that returns "pong" to
    indicate that the server is running.
2.  `/predict`: An endpoint that accepts POST requests with JSON content.
    Each request should contain one or more instances for which the
    predictions are desired. The endpoint returns the predictions and
    associated probabilities in a JSON response.

The server expects an environment variable `model_path` that points to
the directory where the AutoGluon model artifacts are
stored. If `model_path` is not provided, it defaults to '/autogluon/models'.
"""

import json
import logging
import os

from autogluon.tabular import TabularPredictor
import flask
import pandas as pd

from util import constants
from util import fileutils

_SUCCESS_STATUS = 200
_ERROR_STATUS = 500
_PORT = 8501

app = flask.Flask(__name__)
# Check the environment variables.
model_dir = os.getenv('model_path', '/autogluon/models')
logging.info('Model directory passed by the user is: %s', model_dir)
# If the model is on GCS then copy it to a local folder first.
if model_dir.startswith(constants.GCS_URI_PREFIX):
  gcs_path = model_dir[len(constants.GCS_URI_PREFIX) :]
  local_model_dir = os.path.join(constants.LOCAL_MODEL_DIR, gcs_path)
  logging.info('Download %s to %s', model_dir, local_model_dir)
  fileutils.download_gcs_dir_to_local(model_dir, local_model_dir)
  model_dir = local_model_dir
  logging.info('Local model directory is: %s', model_dir)


# Load the predictor at startup.
predictor = TabularPredictor.load(model_dir)


@app.route('/ping', methods=['GET'])
def ping() -> flask.Response:
  """Health check route."""
  return flask.Response('pong', status=_SUCCESS_STATUS)


@app.route('/predict', methods=['POST'])
def predict() -> flask.Response:
  """Prediction route."""
  try:
    # Extract JSON content from the POST request.
    data = flask.request.get_json(force=True)
    instances = data.get('instances', [])

    # Convert instances to DataFrame.
    df_to_predict = pd.DataFrame(instances)

    # Perform prediction.
    predictions = predictor.predict(df_to_predict).tolist()
    response = {'predictions': predictions}

    return flask.Response(
        json.dumps(response),
        status=_SUCCESS_STATUS,
        mimetype='application/json',
    )

  except Exception as e:  # pylint: disable=broad-exception-caught
    return flask.Response(
        json.dumps({'error': str(e)}),
        status=_ERROR_STATUS,
        mimetype='application/json',
    )


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=_PORT)
