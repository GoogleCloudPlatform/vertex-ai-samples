"""Custom handler for Pic2Word."""

from argparse import Namespace  # pylint: disable=g-importing-member
import os
from typing import Any, List

from absl import logging
from data import CustomFolder
from eval_utils import visualize_results
from model.clip import load
from model.model import convert_weights
from model.model import IM2TEXT
from params import get_project_root
import torch
from torch.utils.data import DataLoader
from ts.torch_handler.base_handler import BaseHandler

from util import fileutils

# The COCO dataset is stored in a publicly accessible bucket.
_COCO_STORAGE_DIR = "gs://pic2word-bucket/data/coco/"
_COCO_LOCAL_DIR = "/home/model-server/composed_image_retrieval/data/coco/"
_COCO_VAL2017_PATH = "coco/val2017"
_COCO_DATASET_NAME = "coco"
_MODEL_NAME = "ViT-L/14"
_LOCAL_QUERY_PATH = "./query/"
_IMAGE_OUTPUT_LOCAL_DIR = "demo_out/images"
_OUTPUT_LOCAL_DIR = "./demo_out/"
_DATA_DIR = "data"
_CHECKPOINT_DIR = "checkpoint/pic2word_model.pt"
_REQUEST_PROMPTS = "prompts"
_REQUEST_OUTPUT_STORAGE_DIR = "output_storage_dir"
_REQUEST_IMAGE_PATH = "image_path"
_REQUEST_IMAGE_FILE_NAME = "image_file_name"
_RESPONSE_MSG = "Successfully retrieved images."
_PICKLE_DIR_PATH = "gs://pic2word-bucket/pickle/"


class ModelHandler(BaseHandler):
  """A custom model handler implementation."""

  def __init__(self):
    self.initialized = False
    self.gpu = 0
    self.model = None
    self.dataloader = None
    self.prompt = None
    self.output_storage_dir = None

  def initialize(self, context: Any):
    """Initialize."""
    logging.info("Initializing pic2word.")
    # Download pickle file for COCO
    fileutils.download_gcs_dir_to_local(_PICKLE_DIR_PATH, "./data")

    # Download COCO dataset. The model looks for this folder specifically
    # during image retrieval to generate a response for each request.
    # This is a publicly accessible bucket.
    fileutils.download_gcs_dir_to_local(
        _COCO_STORAGE_DIR,
        _COCO_LOCAL_DIR,
    )

    # Load the model.

    self.initialized = True

    torch.cuda.set_device(self.gpu)
    model, _, preprocess_val = load(_MODEL_NAME, jit=False)

    img2text = IM2TEXT(
        embed_dim=model.embed_dim,
        output_dim=model.token_embedding.weight.shape[1],
    )

    model.cuda(self.gpu)
    img2text.cuda(self.gpu)

    convert_weights(model)
    convert_weights(img2text)

    self.model = model
    self.img2text = img2text

    # Load the dataset
    logging.info("Loading dataset.")

    root_project = os.path.join(get_project_root(), _DATA_DIR)
    dataset = CustomFolder(
        os.path.join(root_project, _COCO_VAL2017_PATH), transform=preprocess_val
    )

    # Initialize the dataloader. This is used to create the pickle file from
    # the dataset.
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
    )

    self.dataloader = dataloader

    logging.info("Finished initializing Pic2Word server.")

  def preprocess(self, data: Any) -> str:
    """Preprocess input data."""
    logging.info("Preprocessing Pic2Word inference request.")
    query = data[0]

    self.output_storage_dir = query[_REQUEST_OUTPUT_STORAGE_DIR]
    prompts = query[_REQUEST_PROMPTS]
    prompts = prompts.split(",")
    self.prompt = prompts

    image_path = query[_REQUEST_IMAGE_PATH]
    # The query image is only supported via GCS bucket upload.
    fileutils.download_gcs_dir_to_local(image_path, _LOCAL_QUERY_PATH)
    image_file_name = query[_REQUEST_IMAGE_FILE_NAME]

    query_file = f"./query/{image_file_name}"

    logging.info("Setting model args.")

    args = {
        "openai-pretrained": True,
        "resume": _CHECKPOINT_DIR,
        "retrieval_data": _COCO_DATASET_NAME,
        "query_file": query_file,
        "demo_out": _OUTPUT_LOCAL_DIR,
        "prompts": prompts,
        "distributed": False,
        "dp": False,
        "gpu": 0,
        "model": _MODEL_NAME,
        "world_size": 1,
    }
    model_input = Namespace(**args)

    logging.info("Finished preprocessing Pic2Word inference request.")
    return model_input

  def inference(self, model_input: Any):
    """Runs inference."""
    logging.info("Running model-inference.")
    visualize_results(
        model=self.model,
        img2text=self.img2text,
        args=model_input,
        prompt=self.prompt,
        dataloader=self.dataloader,
    )

  def postprocess(self):
    """Upload the output images to the bucket."""
    logging.info("Running request postprocess.")
    fileutils.upload_local_dir_to_gcs(
        _IMAGE_OUTPUT_LOCAL_DIR, self.output_storage_dir
    )

  def handle(self, data: Any, context: Any) -> List[str]:  # pylint: disable=unused-argument
    """Runs preprocess, inference, and post-processing."""
    logging.info("Received Pic2Word inference request")
    model_input = self.preprocess(data)
    self.inference(model_input)
    self.postprocess()
    logging.info("Done handling input.")
    return [_RESPONSE_MSG]