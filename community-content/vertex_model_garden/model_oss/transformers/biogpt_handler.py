"""Custom handler for huggingface/biogpt models."""

import os
from typing import Any, List

from absl import logging
import torch
from transformers import BioGptForCausalLM, BioGptTokenizer
from transformers import pipeline
from ts.torch_handler.base_handler import BaseHandler

from util import constants
from util import fileutils

# Tasks
TEXT_GENERATION = "text-generation"

# prompt specific parameters
MAX_LENGTH = 200
NUM_RETURN_SEQUENCES = 10

# Default Model ID
DEFAULT_MODEL_ID = "microsoft/biogpt"

logging.set_verbosity(os.environ.get("LOG_LEVEL", logging.INFO))


class BioGPTHandler(BaseHandler):
  """Custom handler for BioGPT models."""

  def initialize(self, context: Any):
    """Initializes the handler."""
    logging.info("Start to initialize the BioGPT handler.")
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
    self.max_length = int(os.environ.get("MAX_LENGTH", MAX_LENGTH))
    self.num_return_sequences = int(
        os.environ.get("NUM_RETURN_SEQUENCES", NUM_RETURN_SEQUENCES)
    )
    self.base_model_id = os.environ.get("BASE_MODEL_ID", None)
    self.model_id = self.base_model_id
    if not self.base_model_id:
      self.model_id = os.environ.get("MODEL_ID", "microsoft/biogpt")
    if fileutils.is_gcs_path(self.model_id):
      fileutils.download_gcs_dir_to_local(
          self.model_id,
          constants.LOCAL_BASE_MODEL_DIR,
          skip_hf_model_bin=True,
      )
      self.model_id = constants.LOCAL_BASE_MODEL_DIR

    logging.info(f"Using base model:{self.model_id}")

    self.pipeline = None
    self.tokenizer = None

    self.tokenizer = BioGptTokenizer.from_pretrained(self.model_id)
    logging.debug("Initialized the BioGPT tokenizer.")
    model = BioGptForCausalLM.from_pretrained(self.model_id)
    logging.debug("Initialized the base model.")
    self.pipeline = pipeline(
        TEXT_GENERATION, model=model, tokenizer=self.tokenizer
    )

    self.initialized = True
    logging.info("The BioGPT handler was initialized.")

  def preprocess(self, data: Any) -> str:
    """Preprocess input data."""
    prompt = data[0]["prompt"]
    return prompt

  def inference(self, data: Any, *args, **kwargs) -> str:
    """Run the inference."""
    logging.debug(f"Inference prompts={data}")
    predicted_results = self.pipeline(
        data,
        max_length=self.max_length,
        num_return_sequences=self.num_return_sequences,
        do_sample=True,
    )[0]["generated_text"]
    return predicted_results

  def postprocess(self, data: Any) -> List[str]:
    """Postprocesses output data."""
    output = data.replace("<|endoftext|></s>", "")
    return [output]