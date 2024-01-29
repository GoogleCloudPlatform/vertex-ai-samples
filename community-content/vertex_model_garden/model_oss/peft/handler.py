"""Custom handler for huggingface/peft models."""

# pylint: disable=g-importing-member
# pylint: disable=logging-fstring-interpolation

import logging
import os
from typing import Any, List

from absl import logging
from diffusers import DPMSolverMultistepScheduler
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from PIL import Image
import torch
import transformers
from transformers import AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from ts.torch_handler.base_handler import BaseHandler

from util import constants
from util import fileutils
from util import image_format_converter

# Tasks
TEXT_TO_IMAGE_LORA = "text-to-image-lora"
SEQUENCE_CLASSIFICATION_LORA = "sequence-classification-lora"
CAUSAL_LANGUAGE_MODELING_LORA = "causal-language-modeling-lora"
INSTRUCT_LORA = "instruct-lora"

# Inference parameters.
_NUM_INFERENCE_STEPS = 25
_MAX_LENGTH_DEFAULT = 200
_TOP_K_DEFAULT = 10
_TOP_P_DEFAULT = 1.0
_TEMPERATURE_DEFAULT = 1.0


class PeftHandler(BaseHandler):
  """Custom handler for Peft models."""

  def initialize(self, context: Any):
    """Initializes the handler."""
    logging.info("Start to initialize the PEFT handler.")
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
    self.precision_mode = os.environ.get(
        "PRECISION_LOADING_MODE", constants.PRECISION_MODE_16
    )
    self.task = os.environ.get("TASK", CAUSAL_LANGUAGE_MODELING_LORA)
    self.base_model_id = os.environ.get(
        "BASE_MODEL_ID", "openlm-research/open_llama_7b"
    )
    if fileutils.is_gcs_path(self.base_model_id):
      fileutils.download_gcs_dir_to_local(
          self.base_model_id,
          constants.LOCAL_BASE_MODEL_DIR,
          skip_hf_model_bin=True,
      )
      self.base_model_id = constants.LOCAL_BASE_MODEL_DIR
    self.finetuned_lora_model_path = os.environ.get(
        "FINETUNED_LORA_MODEL_PATH", ""
    )
    if fileutils.is_gcs_path(self.finetuned_lora_model_path):
      fileutils.download_gcs_dir_to_local(
          self.finetuned_lora_model_path, constants.LOCAL_MODEL_DIR
      )
      self.finetuned_lora_model_path = constants.LOCAL_MODEL_DIR

    logging.info(
        f"Using task:{self.task}, base model:{self.base_model_id}, lora model:"
        f" {self.finetuned_lora_model_path}, and precision"
        f" {self.precision_mode}."
    )

    self.pipeline = None
    self.model = None
    self.tokenizer = None
    if self.task == TEXT_TO_IMAGE_LORA:
      pipeline = StableDiffusionPipeline.from_pretrained(
          self.base_model_id, torch_dtype=torch.float16
      )
      logging.debug("Initialized the base model for text to image.")
      pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
          pipeline.scheduler.config
      )
      logging.debug("Initialized the scheduler for text to image.")
      if self.finetuned_lora_model_path:
        pipeline.unet.load_attn_procs(self.finetuned_lora_model_path)
      logging.debug("Initialized the LoRA model for text to image.")
      # This is to reduce GPU memory requirements.
      pipeline.enable_xformers_memory_efficient_attention()
      pipeline = pipeline.to(self.map_location)
      # Reduces memory footprint.
      pipeline.enable_attention_slicing()
      self.pipeline = pipeline
      logging.info("Initialized the text to image pipelines.")
    elif self.task == SEQUENCE_CLASSIFICATION_LORA:
      tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
      logging.debug("Initialized the tokenizer for sequence classification.")
      model = AutoModelForSequenceClassification.from_pretrained(
          self.base_model_id, torch_dtype=torch.float16
      )
      logging.debug("Initialized the base model for sequence classification.")
      if self.finetuned_lora_model_path:
        model = PeftModel.from_pretrained(model, self.finetuned_lora_model_path)
        logging.debug("Initialized the LoRA model for sequence classification.")
      model.to(self.map_location)
      self.model = model
      self.tokenizer = tokenizer
    elif (
        self.task == CAUSAL_LANGUAGE_MODELING_LORA or self.task == INSTRUCT_LORA
    ):
      tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
      logging.debug("Initialized the tokenizer.")
      if self.task == CAUSAL_LANGUAGE_MODELING_LORA:
        if self.precision_mode == constants.PRECISION_MODE_32:
          model = AutoModelForCausalLM.from_pretrained(
              self.base_model_id,
              return_dict=True,
              torch_dtype=torch.float32,
              device_map="auto",
          )
        elif self.precision_mode == constants.PRECISION_MODE_16:
          model = AutoModelForCausalLM.from_pretrained(
              self.base_model_id,
              return_dict=True,
              torch_dtype=torch.bfloat16,
              device_map="auto",
          )
        elif self.precision_mode == constants.PRECISION_MODE_8:
          quantization_config = BitsAndBytesConfig(
              load_in_8bit=True, int8_threshold=0
          )
          model = AutoModelForCausalLM.from_pretrained(
              self.base_model_id,
              return_dict=True,
              torch_dtype=torch.float16,
              device_map="auto",
              quantization_config=quantization_config,
          )
        else:
          quantization_config = BitsAndBytesConfig(
              load_in_4bit=True,
              bnb_4bit_quant_type="nf4",
              bnb_4bit_compute_dtype=torch.bfloat16,
          )
          model = AutoModelForCausalLM.from_pretrained(
              self.base_model_id,
              return_dict=True,
              device_map="auto",
              torch_dtype=torch.bfloat16,
              quantization_config=quantization_config,
          )
      else:
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            return_dict=True,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
      logging.debug("Initialized the base model.")
      if self.finetuned_lora_model_path:
        model = PeftModel.from_pretrained(model, self.finetuned_lora_model_path)
        logging.debug("Initialized the LoRA model.")
      pipeline = transformers.pipeline(
          "text-generation",
          model=model,
          tokenizer=tokenizer,
      )
      self.tokenizer = tokenizer
      self.pipeline = pipeline
    else:
      raise ValueError(f"Invalid TASK: {self.task}")

    self.initialized = True
    logging.info("The PEFT handler was initialized.")

  def preprocess(self, data: Any) -> Any:
    """Preprocesses input data."""
    # Assumes that the parameters are same in one request. We parse the
    # parameters from the first instance for all instances in one request.
    max_length = _MAX_LENGTH_DEFAULT
    max_new_tokens = None
    top_k = _TOP_K_DEFAULT
    top_p = _TOP_P_DEFAULT
    temperature = _TEMPERATURE_DEFAULT

    prompts = [item["prompt"] for item in data]
    if "max_length" in data[0]:
      max_length = data[0]["max_length"]
    if "top_k" in data[0]:
      top_k = data[0]["top_k"]
    if "top_p" in data[0]:
      top_p = data[0]["top_p"]
    if "temperature" in data[0]:
      temperature = data[0]["temperature"]

    return prompts, max_length, max_new_tokens, top_k, top_p, temperature

  def inference(self, data: Any, *args, **kwargs) -> List[Image.Image]:
    """Runs the inference."""
    prompts, max_length, max_new_tokens, top_k, top_p, temperature = data
    logging.debug(
        f"Inference prompts={prompts}, max_length={max_length}, top_k={top_k}."
    )
    if self.task == TEXT_TO_IMAGE_LORA:
      predicted_results = self.pipeline(
          prompt=prompts, num_inference_steps=_NUM_INFERENCE_STEPS
      ).images
    elif self.task == SEQUENCE_CLASSIFICATION_LORA:
      encoded_input = self.tokenizer(prompts, return_tensors="pt")
      encoded_input.to(self.map_location)
      with torch.no_grad():
        outputs = self.model(**encoded_input)
      predictions = outputs.logits.argmax(dim=-1)
      predicted_results = predictions.tolist()
    elif (
        self.task == CAUSAL_LANGUAGE_MODELING_LORA or self.task == INSTRUCT_LORA
    ):
      predicted_results = self.pipeline(
          prompts,
          max_length=max_length,
          max_new_tokens=max_new_tokens,
          do_sample=True,
          temperature=temperature,
          top_k=top_k,
          top_p=top_p,
          num_return_sequences=1,
          eos_token_id=self.tokenizer.eos_token_id,
      )
    else:
      raise ValueError(f"Invalid TASK: {self.task}")
    return predicted_results

  def postprocess(self, data: Any) -> List[str]:
    """Postprocesses output data."""
    if self.task == TEXT_TO_IMAGE_LORA:
      # Converts the images to base64 string.
      outputs = [
          image_format_converter.image_to_base64(image) for image in data
      ]
    else:
      outputs = data
    return outputs


# pylint: enable=logging-fstring-interpolation
