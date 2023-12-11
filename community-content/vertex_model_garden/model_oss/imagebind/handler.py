"""Custom handler for the ImageBind model."""

import logging
import os
from typing import Any, Dict, List

from imagebind import data as data_util
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from PIL import Image
import torch
from torchvision import transforms
from ts.torch_handler import base_handler

from util import constants
from util import fileutils


_VIDEO_KEY_TO_AVOID_CONFLICT_WITH_IMAGE = "video"


class ImageBindHandler(base_handler.BaseHandler):
  """Custom handler for the ImageBind model.

  Attributes:
    map_location: Mapping storage location.
    device: Device on which to run inference.
    manifest: TorchServe manifest.
    task: Task for which to run the ImageBind model.
    model: ImageBind model instance.
  """

  def initialize(self, context: Any) -> None:
    """Initializes the ImageBind model handler.

    Args:
      context: TorchServe context, which contains system information and the
        manifest.

    Raises:
      ValueError: A task that is unsupported by the handler.
    """
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

    self.task = os.environ.get("TASK", constants.FEATURE_EMBEDDING_GENERATION)
    if self.task not in [
        constants.FEATURE_EMBEDDING_GENERATION,
        constants.ZERO_SHOT_CLASSIFICATION,
    ]:
      raise ValueError(f"Invalid task: {self.task}.")
    logging.info(
        "Handler initializing ImageBind pretrained model for task %s.",
        self.task,
    )

    self.model = imagebind_model.imagebind_huge(pretrained=True)
    self.model.eval()
    self.model.to(self.device)

    logging.info("Initialized ImageBind pretrained model.")
    self.initialized = True

  def preprocess(self, data: Any) -> List[Dict[str, Any]]:
    """Preprocesses input data, including text, image, audio and video data.

    Args:
      data: Input data.

    Returns:
      A list of processed data samples, with each sample being a dictionary of
      modality (key): input (value) pairs.
    """
    logging.info("Preprocessing: %d instances received.", len(data))
    preprocessed_sample_list = []
    for item in data:
      preprocessed_sample = {}
      if ModalityType.TEXT in item:
        preprocessed_sample[ModalityType.TEXT] = (
            data_util.load_and_transform_text(
                item[ModalityType.TEXT], self.device
            )
        )
      for image_modality in [
          ModalityType.VISION,
          ModalityType.DEPTH,
          ModalityType.THERMAL,
      ]:
        if image_modality in item:
          image_paths = item[image_modality]
          local_image_paths = fileutils.download_gcs_file_list_to_local(
              image_paths, constants.LOCAL_DATA_DIR
          )
          is_depth_or_thermal = image_modality in [
              ModalityType.DEPTH,
              ModalityType.THERMAL,
          ]
          preprocessed_sample[image_modality] = (
              self._load_and_transform_image_data(
                  local_image_paths,
                  self.device,
                  is_depth_or_thermal=is_depth_or_thermal,
              )
          )
      if ModalityType.AUDIO in item:
        audio_paths = item[ModalityType.AUDIO]
        local_audio_paths = fileutils.download_gcs_file_list_to_local(
            audio_paths, constants.LOCAL_DATA_DIR
        )
        preprocessed_sample[ModalityType.AUDIO] = (
            data_util.load_and_transform_audio_data(
                local_audio_paths, self.device
            )
        )
      if _VIDEO_KEY_TO_AVOID_CONFLICT_WITH_IMAGE in item:
        video_paths = item[_VIDEO_KEY_TO_AVOID_CONFLICT_WITH_IMAGE]
        local_video_paths = fileutils.download_gcs_file_list_to_local(
            video_paths, constants.LOCAL_DATA_DIR
        )
        preprocessed_sample[_VIDEO_KEY_TO_AVOID_CONFLICT_WITH_IMAGE] = (
            data_util.load_and_transform_video_data(
                local_video_paths, self.device
            )
        )
      if ModalityType.IMU in item:
        # Input data in the IMU modality are expected in shape [B, 6, 2000].
        preprocessed_sample[ModalityType.IMU] = torch.tensor(
            item[ModalityType.IMU], dtype=torch.float32, device=self.device
        )
      if preprocessed_sample:
        preprocessed_sample_list.append(preprocessed_sample)
    return preprocessed_sample_list

  def _load_and_transform_image_data(
      self,
      image_paths: List[str],
      device: torch.device,
      is_depth_or_thermal: bool = False,
  ) -> torch.Tensor:
    """Loads and transforms 3-channel images, depth images and thermal images.

    Args:
      image_paths: A list of image paths.
      device: Device onto which to load images.
      is_depth_or_thermal: Whether the images are depth or thermal images.

    Returns:
      A list of processed tensors corresponding to the input images.

    Raises:
      ValueError: The input image_paths is None.
    """
    if image_paths is None:
      raise ValueError("image_paths must not be None.")

    image_outputs = []
    for image_path in image_paths:
      transforms_list = [
          transforms.Resize(
              224, interpolation=transforms.InterpolationMode.BICUBIC
          ),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
      ]
      if not is_depth_or_thermal:
        transforms_list.append(
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            )
        )
      data_transform = transforms.Compose(transforms_list)
      with open(image_path, "rb") as fopen:
        if is_depth_or_thermal:
          image = Image.open(fopen).convert("L")
        else:
          image = Image.open(fopen).convert("RGB")

      image = data_transform(image).to(device)
      image_outputs.append(image)
    return torch.stack(image_outputs, dim=0)

  def inference(
      self, data: List[Dict[str, Any]], *args, **kwargs
  ) -> List[Dict[str, Any]]:
    """Runs inference using the ImageBind model.

    Args:
      data: A list of processed data samples, with each sample being a
        dictionary of modality (key): input (value) pairs.
      *args: Additional inference args.
      **kwargs: Additional inference kwargs.

    Returns:
      A list of model outputs, with each output being a dictionary of
      modality (key): embedding (value) pairs.
    """
    output_list = []
    with torch.no_grad():
      for inputs in data:
        if _VIDEO_KEY_TO_AVOID_CONFLICT_WITH_IMAGE in inputs:
          # Allows inference on both image and video data, which both fall under
          # ModalityType.VISION.
          video_inputs = {
              ModalityType.VISION: inputs[
                  _VIDEO_KEY_TO_AVOID_CONFLICT_WITH_IMAGE
              ]
          }
          video_embeddings = self.model(video_inputs)
          video_embeddings[_VIDEO_KEY_TO_AVOID_CONFLICT_WITH_IMAGE] = (
              video_embeddings[ModalityType.VISION]
          )
          del video_embeddings[ModalityType.VISION]
          del inputs[_VIDEO_KEY_TO_AVOID_CONFLICT_WITH_IMAGE]
        else:
          video_embeddings = {}
        embeddings = self.model(inputs)
        embeddings.update(video_embeddings)
        output_list.append(embeddings)
    return output_list

  def postprocess(self, output_list: List[Dict[str, Any]]) -> List[Any]:
    """Postprocesses model outputs for the task of interest.

    For feature embedding generation, returns the embeddings for each modality
    for each input.
    For zero-shot classification, generates classification probabilities
    between the inputs of a pair of modalities for all possible pairings.

    Args:
      output_list: A list of model outputs, with each output being a dictionary
        of modality (key): embedding (value) pairs.

    Returns:
      A list of postprocessed model outputs for the task of interest, with each
      output corresponding to an input.

    Raises:
      ValueError: Fewer than two modalities are provided for zero-shot
      classification, or the task is not supported.
    """
    preds = []
    if self.task == constants.FEATURE_EMBEDDING_GENERATION:
      for item in output_list:
        preds.append({k: v.tolist() for k, v in item.items()})
    elif self.task == constants.ZERO_SHOT_CLASSIFICATION:
      for item in output_list:
        modalities = list(item.keys())
        if len(modalities) < 2:
          raise ValueError(
              "Two or more modalities are needed for task"
              f" {constants.ZERO_SHOT_CLASSIFICATION}."
          )
        pairwise_probs = {}
        for m1 in modalities:
          for m2 in modalities:
            if m1 == m2:
              continue
            probs = torch.softmax(item[m1] @ item[m2].T, dim=-1)
            pairwise_probs[
                f"Classify each input in {m1} (row) against inputs in"
                f" {m2} (column)"
            ] = probs.tolist()
        preds.append(pairwise_probs)
    else:
      raise ValueError(f"Task {self.task} is not supported by the handler.")
    return preds
