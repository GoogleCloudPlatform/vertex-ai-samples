"""Custom handler for huggingface/peft models."""

# pylint: disable=g-importing-member
# pylint: disable=logging-fstring-interpolation

import os
import time
from typing import Any, List, Tuple

from absl import logging
from awq import AutoAWQForCausalLM
from diffusers import DPMSolverMultistepScheduler
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from PIL import Image
import psutil
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

if os.path.exists(constants.SHARED_MEM_DIR):
  logging.info(
      "SharedMemorySizeMb: %s",
      psutil.disk_usage(constants.SHARED_MEM_DIR).free / 1e6,
  )

# Tasks
TEXT_TO_IMAGE_LORA = "text-to-image-lora"
SEQUENCE_CLASSIFICATION_LORA = "sequence-classification-lora"
CAUSAL_LANGUAGE_MODELING_LORA = "causal-language-modeling-lora"
INSTRUCT_LORA = "instruct-lora"

# Inference parameters.
_NUM_INFERENCE_STEPS = 25
_MAX_LENGTH_DEFAULT = 200
_MAX_TOKENS_DEFAULT = None
_TEMPERATURE_DEFAULT = 1.0
_TOP_P_DEFAULT = 1.0
_TOP_K_DEFAULT = 10

logging.set_verbosity(os.environ.get("LOG_LEVEL", logging.INFO))


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
    self.base_model_id = os.environ.get("BASE_MODEL_ID", None)
    self.model_id = self.base_model_id
    if not self.base_model_id:
      self.model_id = os.environ.get("MODEL_ID", "")
    self.quantization = os.environ.get("QUANTIZATION", None)
    logging.info(f"Load base model id from MODEL_ID:{self.model_id}.")
    if not self.model_id:
      self.model_id = os.environ.get("AIP_STORAGE_URI", "")
      logging.info(f"Load base model id from AIP_STORAGE_URI: {self.model_id}.")
    if not self.model_id:
      raise ValueError("Base model id is must be set.")
    if fileutils.is_gcs_path(self.model_id):
      fileutils.download_gcs_dir_to_local(
          self.model_id,
          constants.LOCAL_BASE_MODEL_DIR,
          skip_hf_model_bin=True,
      )
      self.model_id = constants.LOCAL_BASE_MODEL_DIR
    self.finetuned_lora_model_path = os.environ.get(
        "FINETUNED_LORA_MODEL_PATH", ""
    )
    if fileutils.is_gcs_path(self.finetuned_lora_model_path):
      fileutils.download_gcs_dir_to_local(
          self.finetuned_lora_model_path, constants.LOCAL_MODEL_DIR
      )
      self.finetuned_lora_model_path = constants.LOCAL_MODEL_DIR

    logging.info(
        f"Using task:{self.task}, base model:{self.model_id}, lora model:"
        f" {self.finetuned_lora_model_path}, and precision"
        f" {self.precision_mode}."
    )

    self.pipeline = None
    self.model = None
    self.tokenizer = None
    start_time = time.perf_counter()
    logging.info("Started PEFT handler initialization at: %s", start_time)
    if self.task == TEXT_TO_IMAGE_LORA:
      pipeline = StableDiffusionPipeline.from_pretrained(
          self.model_id, torch_dtype=torch.float16
      )
      logging.debug("Initialized the base model for text to image.")
      pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
          pipeline.scheduler.config
      )
      logging.debug("Initialized the scheduler for text to image.")
      # This is to reduce GPU memory requirements.
      pipeline.enable_xformers_memory_efficient_attention()
      pipeline = pipeline.to(self.map_location)
      # Reduces memory footprint.
      pipeline.enable_attention_slicing()
      if self.finetuned_lora_model_path:
        pipeline.load_lora_weights(self.finetuned_lora_model_path)
      logging.debug("Initialized the LoRA model for text to image.")
      self.pipeline = pipeline
      logging.info("Initialized the text to image pipelines.")
    elif self.task == SEQUENCE_CLASSIFICATION_LORA:
      tokenizer = AutoTokenizer.from_pretrained(self.model_id)
      logging.debug("Initialized the tokenizer for sequence classification.")
      model = AutoModelForSequenceClassification.from_pretrained(
          self.model_id, torch_dtype=torch.float16
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
      tokenizer = AutoTokenizer.from_pretrained(self.model_id)
      logging.debug("Initialized the tokenizer.")
      if self.task == CAUSAL_LANGUAGE_MODELING_LORA:
        if self.quantization == constants.AWQ:
          model = AutoAWQForCausalLM.from_quantized(self.model_id)
        elif self.quantization == constants.GPTQ or not self.quantization:
          if self.precision_mode == constants.PRECISION_MODE_32:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                return_dict=True,
                torch_dtype=torch.float32,
                device_map="auto",
            )
          elif self.precision_mode == constants.PRECISION_MODE_16B:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                return_dict=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
          elif self.precision_mode == constants.PRECISION_MODE_16:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                return_dict=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )
          elif self.precision_mode == constants.PRECISION_MODE_8:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True, int8_threshold=0
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
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
                self.model_id,
                return_dict=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                quantization_config=quantization_config,
            )
        else:
          raise ValueError(f"Invalid QUANTIZATION value: {self.quantization}")
      else:
        try:
          model = AutoModelForCausalLM.from_pretrained(
              self.model_id,
              torch_dtype=torch.bfloat16,
              trust_remote_code=True,
              device_map="auto",
          )
        except:  # pylint: disable=bare-except
          model = AutoModelForCausalLM.from_pretrained(
              self.model_id,
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
    end_time = time.perf_counter()
    logging.info("The PEFT handler was initialize at: %s", end_time)
    logging.info("Handler initiation took %s seconds", end_time - start_time)

  def preprocess(self, data: Any) -> Any:
    """Preprocesses input data."""
    # Assumes that the parameters are same in one request. We parse the
    # parameters from the first instance for all instances in one request.
    # For generation length: `max_length` defines the maximum length of the
    # sequence to be generated, including both input and output tokens.
    # `max_length` is overridden by `max_new_tokens` if also set.
    # `max_new_tokens` defines the maximum number of new tokens to generate,
    # ignoring the current number of tokens.
    # Reference:
    # https://github.com/huggingface/transformers/blob/574a5384557b1aaf98ddb13ea9eb0a0ee8ff2cb2/src/transformers/generation/configuration_utils.py#L69-L73
    max_length = _MAX_LENGTH_DEFAULT
    max_tokens = _MAX_TOKENS_DEFAULT
    temperature = _TEMPERATURE_DEFAULT
    top_p = _TOP_P_DEFAULT
    top_k = _TOP_K_DEFAULT

    prompts = [item["prompt"] for item in data]
    if "max_length" in data[0]:
      max_length = data[0]["max_length"]
    if "max_tokens" in data[0]:
      max_tokens = data[0]["max_tokens"]
    if "temperature" in data[0]:
      temperature = data[0]["temperature"]
    if "top_p" in data[0]:
      top_p = data[0]["top_p"]
    if "top_k" in data[0]:
      top_k = data[0]["top_k"]

    return prompts, max_length, max_tokens, temperature, top_p, top_k

  def inference(
      self, data: Any, *args, **kwargs
  ) -> Tuple[List[str], List[Image.Image]]:
    """Runs the inference."""
    prompts, max_length, max_tokens, temperature, top_p, top_k = data
    logging.debug(
        f"Inference prompts={prompts}, max_length={max_length},"
        f" max_tokens={max_tokens}, temperature={temperature}, top_p={top_p},"
        f" top_k={top_k}."
    )
    if self.task == TEXT_TO_IMAGE_LORA:
      predicted_results = self.pipeline(
          prompt=prompts, num_inference_steps=_NUM_INFERENCE_STEPS
      ).images
    elif self.task == SEQUENCE_CLASSIFICATION_LORA:
      encoded_input = self.tokenizer(prompts, return_tensors="pt", padding=True)
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
          max_new_tokens=max_tokens,
          do_sample=True,
          temperature=temperature,
          top_p=top_p,
          top_k=top_k,
          num_return_sequences=1,
          eos_token_id=self.tokenizer.eos_token_id,
          return_full_text=False,
      )
    else:
      raise ValueError(f"Invalid TASK: {self.task}")
    return prompts, predicted_results

  def postprocess(self, data: Any) -> List[str]:
    """Postprocesses output data."""
    prompts, predicted_results = data
    if self.task == TEXT_TO_IMAGE_LORA:
      # Converts the images to base64 string.
      outputs = [
          image_format_converter.image_to_base64(image)
          for image in predicted_results
      ]
    elif self.task == SEQUENCE_CLASSIFICATION_LORA:
      outputs = predicted_results
    else:
      outputs = []
      for prompt, predicted_result in zip(prompts, predicted_results):
        formatted_output = self._format_text_generation_output(
            prompt=prompt, output=predicted_result[0]["generated_text"]
        )
        outputs.append(formatted_output)
    return outputs

  def _format_text_generation_output(self, prompt: str, output: str) -> str:
    """Formats text generation output."""
    output = output.strip("\n")
    return f"Prompt:\n{prompt.strip()}\nOutput:\n{output}"


# pylint: enable=logging-fstring-interpolation