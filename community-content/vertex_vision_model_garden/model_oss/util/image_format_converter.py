"""Image format converter util lib."""

import base64
import io

from PIL import Image


def image_to_base64(image: Image.Image) -> str:
  """Convert a PIL image to a base64 string."""
  buffer = io.BytesIO()
  image.save(buffer, format="JPEG")
  image_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
  return image_str


def base64_to_image(image_str: str) -> Image.Image:
  """Convert a base64 string to a PIL image."""
  image = Image.open(io.BytesIO(base64.b64decode(image_str)))
  return image
