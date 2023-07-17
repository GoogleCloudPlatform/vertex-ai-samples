"""Video format converter util lib."""

import io
from typing import Sequence
import imageio
import numpy as np
from PIL import Image


def frames_to_video_bytes(frames: Sequence[np.ndarray], fps: int) -> bytes:
  images = [Image.fromarray(array) for array in frames]
  io_obj = io.BytesIO()
  imageio.mimsave(io_obj, images, format=".mp4", fps=fps)
  return io_obj.getvalue()
