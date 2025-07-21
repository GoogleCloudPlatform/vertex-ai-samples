"""Predict server for TimesFM."""

import json
import os
import flask
import predictor
from predictor import PredictionError

# Create the flask app.
app = flask.Flask(__name__)
_OK_STATUS = 200
_INTERNAL_ERROR_STATUS = 500
_BAD_REQUEST_STATUS = 400
_HOST = '0.0.0.0'

# Define the predictor and load the checkpoints.
predictor = predictor.TimesFMPredictor()
predictor.load(os.environ['AIP_STORAGE_URI'])


@app.route(os.environ['AIP_HEALTH_ROUTE'], methods=['GET'])
def health() -> flask.Response:
  return flask.Response(status=_OK_STATUS)


@app.route(os.environ['AIP_PREDICT_ROUTE'], methods=['GET', 'POST'])
def predict() -> flask.Response:
  """Calls TimesFM for prediction.

  Returns:
    A `flask.Response` containing the prediction result in JSON.
  """
  try:
    body = flask.request.get_json(silent=True, force=True)
    preprocessed_inputs = predictor.preprocess(body)
    outputs = predictor.predict(preprocessed_inputs)
    conf_level = preprocessed_inputs.get('conf_level')

    if conf_level is not None:
      postprocessed_outputs = predictor.postprocess_with_conf_level(
          outputs, preprocessed_inputs['conf_level']
      )
    else:
      postprocessed_outputs = predictor.postprocess(outputs)
    return flask.Response(
        json.dumps(postprocessed_outputs),
        status=_OK_STATUS,
        mimetype='application/json',
    )
  except PredictionError as e:
    return flask.Response(
        json.dumps({'error': str(e)}),
        status=e.status_code,
        mimetype='application/json',
    )
  except ValueError as e:
    return flask.Response(
        json.dumps({'error': str(e)}),
        status=_BAD_REQUEST_STATUS,
        mimetype='application/json',
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    return flask.Response(
        json.dumps({'error': str(e)}),
        status=_INTERNAL_ERROR_STATUS,
        mimetype='application/json',
    )


if __name__ == '__main__':
  app.run(host=_HOST, port=os.environ['AIP_HTTP_PORT'])
