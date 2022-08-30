# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adapts a pretrained TIMM image classification model to the CPR framework.

Documentation for the TIMM (Torch IMage Models) library is here:
  https://rwightman.github.io/pytorch-image-models/

Its source can also be found here:
  https://github.com/rwightman/pytorch-image-models
"""

import base64
import binascii
import io
import os
from typing import Dict, List, Union

from fastapi import HTTPException
from google.cloud.aiplatform import prediction as cpr
from pathlib import Path
import PIL
import smart_open
import timm
import torch
import torch.nn.functional as F

with open(Path(__file__).parent.absolute().joinpath("imagenet.txt")) as f:
    IMAGENET_CLASSES = f.read().splitlines()


class TimmPredictor(cpr.predictor.Predictor):
    """Predictor class for image models based on TIMM."""

    TIMM_MODEL_NAME = os.getenv("TIMM_MODEL_NAME", default="vit_small_patch32_224")
    WEIGHTS_FILE = "state_dict.pth"
    NUM_TOP_CLASSES_TO_RETURN = 5

    def __init__(self):
        self._cuda = torch.cuda.device_count() > 0

    def load(self, artifacts_uri: str = ""):
        """Initializes the model and preprocessing transforms.

        Args:
          artifacts_uri: Directory where state dict is stored. Can be a
          GCS URI or local path.
        """
        if artifacts_uri:
            artifact_path = os.path.join(artifacts_uri)
            if not (os.path.isdir(artifact_path) or artifact_path.startswith("gs://")):
                raise ValueError("Provided artifact_uri is not a directory.")
        else:
            artifact_path = os.getcwd()

        artifact_path = os.path.join(artifact_path, self.WEIGHTS_FILE)
        with smart_open.open(artifact_path, "rb") as f:
            self._model = torch.load(f)

        if self._cuda:
            self._model.cuda()

        config = timm.data.resolve_data_config(model=self.TIMM_MODEL_NAME, args=[])
        self._transform = timm.data.create_transform(
            is_training=False, use_prefetcher=False, **config
        )

    def preprocess(self, request_dict: Dict[str, List[str]]) -> torch.Tensor:
        """Performs preprocessing.

        By default, the server expects a request body consisting of a valid JSON
        object. This will be parsed by the handler before it's evaluated by the
        preprocess method.

        Args:
          request_dict: Parsed request body. We expect that the input consists of
            a list of base64-encoded image files under the "instances" key. (Any
            image format that PIL.image.open can handle is okay.)

        Returns:
          torch.Tensor containing the preprocessed images as a batch. If GPU is
          available, the result tensor will be stored on GPU.
        """

        if "instances" not in request_dict:
            raise HTTPException(
                status_code=400,
                detail='Request must contain "instances" as a top-level key.',
            )

        tensors = []

        for (i, image) in enumerate(request_dict["instances"]):
            # We use Base64 encoding to handle image data.
            # This is probably the best we can do while still using JSON input.
            # Overriding the input format requires building a custom Handler.
            try:
                image_bytes = base64.b64decode(image, validate=True)
            except (binascii.Error, TypeError) as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Base64 decoding of the input image at index {i} failed:"
                    f" {str(e)}",
                )

            try:
                pil_image = PIL.Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except PIL.UnidentifiedImageError:
                raise HTTPException(
                    status_code=400,
                    detail=f"The input image at index {i} could not be identified as an"
                    " image file.",
                )

            tensors.append(self._transform(pil_image))

        with torch.inference_mode():
            result = torch.stack(tensors)
            if self._cuda:
                result = result.cuda()
        return result

    def predict(self, instances: torch.Tensor) -> torch.Tensor:
        """Performs prediction.

        Args:
          instances: torch.Tensor with type torch.float32 and shape
            [?, 3, 224, 224], containing the pre-processed input images.

        Returns:
          Vector of scores with type torch.float32 and shape [?, 1000],
            representing the model's estimate of the likelihood that the
            input belongs to the Imagenet class with that index.
        """
        with torch.inference_mode():
            class_scores = self._model(instances)
        return class_scores

    def postprocess(
        self, class_scores: torch.Tensor
    ) -> Dict[str, List[Dict[str, Union[str, int, float]]]]:
        """Translate the model output into a classification result.

        Args:
          class_scores: torch.Tensor with type torch.float32 and shape
            [?, 1000], containing the scores assigned to each class by
            the model.

        Returns:
          Dictionary containing the list of classification results. Each
            classification result contains the probabilities, class names, and
            class indices of the classes with the top class scores as reported by
            the model.
        """
        class_probs = F.softmax(class_scores, dim=1)
        top_k = class_probs.topk(self.NUM_TOP_CLASSES_TO_RETURN)
        top_k_values = top_k.values.numpy().tolist()
        top_k_indices = top_k.indices.numpy().tolist()
        predictions = [
            dict(
                probabilities=values,
                indices=indices,
                class_names=[IMAGENET_CLASSES[int(class_num)] for class_num in indices],
            )
            for (values, indices) in zip(top_k_values, top_k_indices)
        ]
        return {"predictions": predictions}
