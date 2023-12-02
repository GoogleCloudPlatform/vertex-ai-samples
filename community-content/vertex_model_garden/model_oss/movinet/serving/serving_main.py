"""Main executable for MoViNet online / batch predictions."""

from collections.abc import Sequence
import json
import os

from absl import app
from absl import logging
import flask
import tensorflow as tf
import waitress

from movinet.serving import video_serving_lib
from util import constants


flask_app = flask.Flask(__name__)
logging.set_verbosity(logging.INFO)

movinet_model = None

_BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '1'))
_NUM_FRAMES = int(os.environ.get('NUM_FRAMES', '32'))
_FPS = float(os.environ.get('FPS', '5'))
_OVERLAP_FRAMES = int(os.environ.get('OVERLAP_FRAMES', '24'))
_OBJECTIVE = os.environ.get(
    'OBJECTIVE', constants.OBJECTIVE_VIDEO_CLASSIFICATION
).lower()

# VAR parameters.
_CONFIDENCE_THRESHOLD = float(os.environ.get('CONFIDENCE_THRESHOLD', '0.5'))
_MIN_GAP_TIME = float(os.environ.get('MIN_GAP_TIME', '1.5'))


def load_movinet_model() -> None:
  model_path = os.environ.get('MODEL_PATH')

  if not model_path:
    raise app.UsageError('Missing MODEL_PATH environment variable.')

  # We just reload the weights of the fine-tuned diffusion model.
  logging.info('Initialize finetuned models from: %s', model_path)
  global movinet_model
  movinet_model = tf.saved_model.load(model_path)


load_movinet_model()


def error(message: str) -> str:
  """Returns a JSON representing an error response."""
  return json.dumps({
      'success': False,
      'error': message,
  })


# The health check route is required for docker deployment in google cloud.
@flask_app.route('/ping')
def ping() -> flask.Response:
  """Health checks."""
  return flask.Response(status=200)


# The return should be `Response` for docker deployment in google cloud.
@flask_app.route('/predict', methods=['GET', 'POST'])
def predict_model() -> flask.Response:
  """Predictions."""
  if flask.request.method == 'POST':
    contents = flask.request.get_json(force=True)

    logging.info('The input contents are: %s', contents)
    instances = contents.get('instances', [])

    try:
      predictions = []
      for instance in instances:
        executor = video_serving_lib.parse_request(instance)
        prediction = executor.get_prediction(
            movinet_model,
            _BATCH_SIZE,
            _FPS,
            _NUM_FRAMES,
            _OVERLAP_FRAMES,
            _OBJECTIVE,
        )
        if _OBJECTIVE == constants.OBJECTIVE_VIDEO_CLASSIFICATION:
          prediction = video_serving_lib.postprocess_vcn(prediction)
        elif _OBJECTIVE == constants.OBJECTIVE_VIDEO_ACTION_RECOGNITION:
          prediction = video_serving_lib.postprocess_var(
              executor.windows, prediction, _CONFIDENCE_THRESHOLD, _MIN_GAP_TIME
          )
        predictions.append(prediction)
    except ValueError as e:
      return flask.Response(
          error(str(e)), status=500, mimetype='application/json'
      )

    return flask.Response(
        response=json.dumps({
            'success': True,
            'predictions': predictions,
        }),
        status=200,
        mimetype='application/json',
    )
  else:
    return flask.Response(
        response=json.dumps({
            'success': True,
            'isalive': movinet_model is not None,
        }),
        status=200,
        mimetype='application/json',
    )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  # This is used when running locally only. When deploying to Google App
  # Engine, a webserver process such as Gunicorn will serve the app.
  # # Debug deployment.
  # flask_app.run(host='0.0.0.0', port=8501, debug=True)
  # Prod deployment.
  if _OBJECTIVE not in [
      constants.OBJECTIVE_VIDEO_CLASSIFICATION,
      constants.OBJECTIVE_VIDEO_ACTION_RECOGNITION,
  ]:
    raise app.UsageError('Objective must be vcn or var.')
  logging.info(
      'Env: batch_size: %s, num_frames: %s, fps: %s, overlap_frames: %s',
      _BATCH_SIZE,
      _NUM_FRAMES,
      _FPS,
      _OVERLAP_FRAMES,
  )
  waitress.serve(flask_app, host='0.0.0.0', port=8501)


if __name__ == '__main__':
  app.run(main)
