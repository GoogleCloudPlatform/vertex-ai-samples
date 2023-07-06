r"""Servers Keras Stable Diffusion models.

python serve.py --model_path=<model path in gcs>

curl -d \
'{"prompt":"Hello Kitty"}' \
-H "Content-Type: application/json" \
-X POST http://localhost:8501/predict
"""

import base64
import io
import json
import os
from typing import List, Tuple

from absl import app
# The docker builds could not find flask and waitress.
# pylint: disable=import-error
from flask import Flask
from flask import request
from flask import Response
import keras_cv
from PIL import Image
from waitress import serve

from util import constants
from util import fileutils


flask_app = Flask(__name__)

stable_diffusion_model = None


model_path = os.environ.get('MODEL_PATH', '')
if model_path.startswith(constants.GCS_URI_PREFIX):
  print('Downloading models from gcs to local.')
  os.makedirs(constants.LOCAL_MODEL_DIR, exist_ok=True)
  fileutils.download_gcs_dir_to_local(
      os.path.dirname(model_path), constants.LOCAL_MODEL_DIR
  )
  model_path = os.path.join(
      constants.LOCAL_MODEL_DIR, os.path.basename(model_path)
  )

image_width = int(os.environ.get('IMAGE_WIDTH', 512))
image_height = int(os.environ.get('IMAGE_HEIGHT', 512))

print('image_width=', image_width, 'image_height=', image_height)
print('Create Keras stable diffusion models.')
stable_diffusion_model = keras_cv.models.StableDiffusion(
    img_width=image_width,
    img_height=image_height,
    jit_compile=True,
)

if model_path:
  # We just reload the weights of the fine-tuned diffusion model.
  print('Initialize finetuned models from: ', model_path)
  stable_diffusion_model.diffusion_model.load_weights(model_path)


def error(message: str) -> str:
  """Returns a JSON representing an error response."""
  return json.dumps({
      'success': False,
      'error': message,
  })


def check_key_in_json(content: str, keys: List[str]) -> str:
  for key in keys:
    if key not in content:
      return error('No {} in request {}.'.format(key, content))
  return None


def validate_json_key(json_key_string: str) -> Tuple[str, bool]:
  try:
    json_key = json.loads(json_key_string)
  except (ValueError, TypeError):
    return (error('Invalid key found in request'), False)
  return (json_key, True)


# The health check route is required for docker deployment in google cloud.
@flask_app.route('/ping')
def ping() -> Response:
  """Health checks."""
  return Response(status=200)


# The return should be `Response` for docker deployment in google cloud.
@flask_app.route('/predict', methods=['GET', 'POST'])
def predict_model() -> Response:
  """Predictions."""
  if request.method == 'POST':
    contents = request.get_json(force=True)

    print('The input contents are:', contents)
    batch_size = 1
    num_steps = 25
    seed = 1234
    if 'parameters' in contents:
      parameters = contents['parameters']
      if 'batch_size' in parameters:
        batch_size = int(parameters['batch_size'])
      if 'num_steps' in parameters:
        num_steps = int(parameters['num_steps'])
      if 'seed' in parameters:
        seed = int(parameters['seed'])
    print('batch_size=', batch_size, 'num_steps=', num_steps, 'seed=', seed)
    if batch_size < 1:
      return Response(
          response=error('The batch size must be a positive integar.'),
          status=200,
          mimetype='text/plain',
      )
    if num_steps < 1:
      return Response(
          response=error('The num steps must be a positive integar.'),
          status=200,
          mimetype='text/plain',
      )
    predictions = []
    for content in contents['instances']:
      print('Processing:', content)
      prompt = content['prompt']
      generated_image_array = stable_diffusion_model.text_to_image(
          prompt=prompt,
          batch_size=batch_size,
          num_steps=num_steps,
          seed=seed,
      )

      generated_image_bytes_array = []
      for i in range(batch_size):
        generated_image = Image.fromarray(generated_image_array[i])
        # Converts the image to a base64-encoded string.
        buffered_image = io.BytesIO()
        generated_image.save(buffered_image, format='JPEG')
        generated_image_bytes = base64.b64encode(
            buffered_image.getvalue()
        ).decode('utf-8')
        generated_image_bytes_array.append(generated_image_bytes)
      prediction = {
          'prompt': prompt,
          'predicted_image': generated_image_bytes_array,
      }
      predictions.append(prediction)

    return Response(
        response=json.dumps({
            'success': True,
            'predictions': predictions,
        }),
        status=200,
        mimetype='text/plain',
    )
  else:
    return Response(
        response=json.dumps({
            'success': True,
            'isalive': stable_diffusion_model is not None,
        }),
        status=200,
        mimetype='text/plain',
    )


def serve_main(unused_argv):
  """The main function to serve Keras models."""
  del unused_argv
  # This is used when running locally only. When deploying to Google App
  # Engine, a webserver process such as Gunicorn will serve the app.
  # # Debug deployment.
  # flask_app.run(host='0.0.0.0', port=8501, debug=True)
  # Prod deployment.
  serve(flask_app, host='0.0.0.0', port=8501)


if __name__ == '__main__':
  app.run(serve_main)