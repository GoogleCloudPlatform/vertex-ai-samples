"""Custom handler for huggingface/diffusers models."""

# pylint: disable=g-importing-member
# pylint: disable=logging-fstring-interpolation

import base64
import io
import logging
import os
from typing import Any, List, Sequence, Tuple

from diffusers import ControlNetModel
from diffusers import DiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from diffusers import EulerAncestralDiscreteScheduler
from diffusers import StableDiffusionControlNetPipeline
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionInpaintPipeline
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionUpscalePipeline
from diffusers import TextToVideoZeroPipeline
from diffusers import UniPCMultistepScheduler
import imageio
import numpy as np
from PIL import Image
import torch
from ts.torch_handler.base_handler import BaseHandler

from util import constants
from util import fileutils
from util import image_format_converter
from video_util import video_format_converter

STABLE_DIFFUSION_MODEL = "runwayml/stable-diffusion-v1-5"

# Tasks
TEXT_TO_IMAGE = "text-to-image"
IMAGE_TO_IMAGE = "image-to-image"
IMAGE_INPAINTING = "image-inpainting"
INSTRUCT_PIX2PIX = "instruct-pix2pix"
CONTROLNET = "controlnet"
CONDITIONED_SUPER_RES = "conditioned-super-res"
TEXT_TO_VIDEO_ZERO_SHOT = "text-to-video-zero-shot"
TEXT_TO_VIDEO = "text-to-video"


def frames_to_video_bytes(frames: Sequence[np.ndarray], fps: int) -> bytes:
  images = [Image.fromarray(array) for array in frames]
  io_obj = io.BytesIO()
  imageio.mimsave(io_obj, images, format=".mp4", fps=fps)
  return io_obj.getvalue()


class DiffusersHandler(BaseHandler):
  """Custom handler for TIMM models."""

  def initialize(self, context: Any):
    """Custom initialize."""

    properties = context.system_properties
    self.map_location = (
        "cuda"
        if torch.cuda.is_available() and properties.get("gpu_id") is not None
        else "cpu"
    )
    self.device = torch.device(
        self.map_location + ":" + str(properties.get("gpu_id"))
        if torch.cuda.is_available() and properties.get("gpu_id") is not None
        else self.map_location
    )
    self.manifest = context.manifest

    self.model_id = os.environ["MODEL_ID"]
    if self.model_id.startswith(constants.GCS_URI_PREFIX):
      gcs_path = self.model_id[len(constants.GCS_URI_PREFIX) :]
      local_model_dir = os.path.join(constants.LOCAL_MODEL_DIR, gcs_path)
      logging.info(f"Download {self.model_id} to {local_model_dir}")
      fileutils.download_gcs_dir_to_local(self.model_id, local_model_dir)
      self.model_id = local_model_dir

    self.task = os.environ.get("TASK", TEXT_TO_IMAGE)
    logging.info(f"Using task:{self.task}, model:{self.model_id}")

    if self.task == TEXT_TO_IMAGE:
      pipeline = StableDiffusionPipeline.from_pretrained(
          self.model_id, torch_dtype=torch.float16
      )
      pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
          pipeline.scheduler.config
      )
      pipeline = pipeline.to(self.map_location)
      # Reduce memory footprint.
      pipeline.enable_attention_slicing()
    elif self.task == IMAGE_TO_IMAGE:
      pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
          self.model_id, torch_dtype=torch.float16
      )
      pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
          pipeline.scheduler.config
      )
      pipeline = pipeline.to(self.map_location)
      # Reduce memory footprint.
      pipeline.enable_attention_slicing()
    elif self.task == IMAGE_INPAINTING:
      pipeline = StableDiffusionInpaintPipeline.from_pretrained(
          self.model_id, torch_dtype=torch.float16
      )
      pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
          pipeline.scheduler.config
      )
      pipeline = pipeline.to(self.map_location)
      # Reduce memory footprint.
      pipeline.enable_attention_slicing()
    elif self.task == INSTRUCT_PIX2PIX:
      pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
          self.model_id, torch_dtype=torch.float16
      )
      pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
          pipeline.scheduler.config
      )
      pipeline = pipeline.to(self.map_location)
      # Reduce memory footprint.
      pipeline.enable_attention_slicing()
    elif self.task == CONTROLNET:
      controlnet = ControlNetModel.from_pretrained(
          self.model_id, torch_dtype=torch.float16
      )
      pipeline = StableDiffusionControlNetPipeline.from_pretrained(
          STABLE_DIFFUSION_MODEL,
          controlnet=controlnet,
          torch_dtype=torch.float16,
      )
      pipeline.scheduler = UniPCMultistepScheduler.from_config(
          pipeline.scheduler.config
      )
      pipeline.enable_xformers_memory_efficient_attention()
      pipeline.enable_model_cpu_offload()
      pipeline = pipeline.to(self.map_location)
      # Reduce memory footprint.
      pipeline.enable_attention_slicing()
    elif self.task == CONDITIONED_SUPER_RES:
      pipeline = StableDiffusionUpscalePipeline.from_pretrained(
          self.model_id, torch_dtype=torch.float16
      )
      pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
          pipeline.scheduler.config
      )
      # This is necessary to 4x upscale >=256x256 input images with V100.
      logging.info("Enable xformers memory efficient attention for inference.")
      pipeline.enable_xformers_memory_efficient_attention()
      pipeline = pipeline.to(self.map_location)
      # Reduce memory footprint.
      pipeline.enable_attention_slicing()
    elif self.task == TEXT_TO_VIDEO_ZERO_SHOT:
      pipeline = TextToVideoZeroPipeline.from_pretrained(
          STABLE_DIFFUSION_MODEL, torch_dtype=torch.float16
      )
      # Memory optimization.
      pipeline.enable_xformers_memory_efficient_attention()
      pipeline.enable_model_cpu_offload()
      pipeline = pipeline.to(self.map_location)
    elif self.task == TEXT_TO_VIDEO:
      pipeline = DiffusionPipeline.from_pretrained(
          self.model_id, torch_dtype=torch.float16, variant="fp16"
      )
      pipeline.enable_model_cpu_offload()
      # Memory optimization.
      pipeline.enable_vae_slicing()
      pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
          pipeline.scheduler.config
      )
    else:
      raise ValueError(f"Invalid TASK: {self.task}")

    self.pipeline = pipeline
    self.initialized = True
    logging.info("Handler initialization done.")

  def preprocess(self, data: Any) -> Tuple[Any, Any, Any]:
    """Preprocess input data."""
    prompts = [item["prompt"] for item in data]
    images = None
    mask_images = None

    if "image" in data[0]:
      images = [
          image_format_converter.base64_to_image(item["image"]) for item in data
      ]
    if "mask_image" in data[0]:
      mask_images = [
          image_format_converter.base64_to_image(item["mask_image"])
          for item in data
      ]
    return prompts, images, mask_images

  def inference(self, data: Any, *args, **kwargs) -> List[Image.Image]:
    """Run the inference."""
    prompts, images, mask_images = data
    if self.task == TEXT_TO_IMAGE:
      predicted_images = self.pipeline(prompt=prompts).images
    elif self.task == IMAGE_TO_IMAGE:
      predicted_images = self.pipeline(prompt=prompts, image=images).images
    elif self.task == IMAGE_INPAINTING:
      predicted_images = self.pipeline(
          prompt=prompts, image=images, mask_image=mask_images
      ).images
    elif self.task == INSTRUCT_PIX2PIX:
      predicted_images = self.pipeline(prompt=prompts, image=images).images
    elif self.task == CONTROLNET:
      predicted_images = self.pipeline(
          prompt=prompts, image=images, num_inference_steps=20
      ).images
    elif self.task == CONDITIONED_SUPER_RES:
      predicted_images = self.pipeline(
          prompt=prompts, image=images, num_inference_steps=20
      ).images
    elif self.task == TEXT_TO_VIDEO_ZERO_SHOT:
      # For each given prompt, generate a short video.
      # The pipeline doesn't support multiple prompts in one run yet.
      videos = []
      for prompt in prompts:
        numpy_arrays = self.pipeline(prompt=prompt).images
        numpy_arrays = [(i * 255).astype("uint8") for i in numpy_arrays]
        videos.append(
            frames_to_video_bytes(numpy_arrays, fps=4)
        )
      return videos
    elif self.task == TEXT_TO_VIDEO:
      predicted_images = np.asarray(self.pipeline(prompt=prompts).frames)
      # For multiple prompts, the model concatenates video frames, i.e. the
      # output shape is (num_frames, height, width * len(prompts), channels).
      # Therefore we need to split the output into different videos.
      predicted_images = np.array_split(predicted_images, len(prompts), axis=2)
      videos = [
          frames_to_video_bytes(images, fps=8)
          for images in predicted_images
      ]
      return videos
    else:
      raise ValueError(f"Invalid TASK: {self.task}")
    return predicted_images

  def postprocess(self, data: Any) -> List[str]:
    """Convert the images to base64 string."""
    outputs = []
    for prediction in data:
      if isinstance(prediction, bytes):
        # This is the video bytes.
        outputs.append(base64.b64encode(prediction).decode("utf-8"))
      else:
        outputs.append(image_format_converter.image_to_base64(prediction))
    return outputs


# pylint: enable=logging-fstring-interpolation
