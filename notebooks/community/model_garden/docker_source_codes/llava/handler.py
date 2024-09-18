"""Customer handler for LLava 1.5 OSS model.

The code is based on here: https://github.com/haotian-liu/LLaVA
handler based on:
https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/run_llava.py
There are two supported variant:
1. liuhaotian/llava-v1.5-13b: 13B params
2. liuhaotian/llava-v1.5-7b: 7B params
"""

import os
import re
from typing import Any, Dict, List

from llava import constants as llava_constants
from llava import conversation
from llava import mm_utils
from llava.model import builder
import model_handler_setup
import torch
from ts.torch_handler import base_handler

from util import constants
from util import image_format_converter


DEFAULT_MODEL_ID = "liuhaotian/llava-v1.5-7b"


class LlavaHandler(base_handler.BaseHandler):
  """Custom handler for LLava model."""

  def initialize(self, context: Any):
    """Initializes model, tokenizer, and other components."""
    self.map_location = model_handler_setup.get_map_location(context=context)
    self.device = model_handler_setup.get_model_device(
        map_location=self.map_location, context=context
    )
    self.manifest = context.manifest
    self.model_id = model_handler_setup.get_model_id(
        default_model_id=DEFAULT_MODEL_ID
    )

    # Allows 4bit and 8bit quantiziation using BnB nf4.
    precision = os.environ.get("PRECISION_MODE")
    load_8bit = precision == constants.PRECISION_MODE_8
    load_4bit = precision == constants.PRECISION_MODE_4

    self.tokenizer, self.model, self.image_processor, self.context_len = (
        builder.load_pretrained_model(
            model_path=self.model_id,
            model_base=None,
            model_name=mm_utils.get_model_name_from_path(self.model_id),
            load_8bit=load_8bit,
            load_4bit=load_4bit,
        )
    )

  def preprocess(self, data: List[Dict[str, Any]]) -> Any:
    """Runs the preprocessing to tokenize image and the prompt."""
    if len(data) > 1:
      raise ValueError(
          "LLava original repo currently does not support batch inference."
          " https://github.com/haotian-liu/LLaVA/issues/754"
      )
    data = data[0]
    prompt, base64_image = data["prompt"], data["base64_image"]

    # Adds proper image token to the prompt.
    image_token_se = (
        llava_constants.DEFAULT_IM_START_TOKEN
        + llava_constants.DEFAULT_IMAGE_TOKEN
        + llava_constants.DEFAULT_IM_END_TOKEN
    )
    if llava_constants.IMAGE_PLACEHOLDER in prompt:
      if self.model.config.mm_use_im_start_end:
        prompt = re.sub(
            llava_constants.IMAGE_PLACEHOLDER, image_token_se, prompt
        )
      else:
        prompt = re.sub(
            llava_constants.IMAGE_PLACEHOLDER,
            llava_constants.DEFAULT_IMAGE_TOKEN,
            prompt,
        )
    else:
      if self.model.config.mm_use_im_start_end:
        prompt = image_token_se + "\n" + prompt
      else:
        prompt = llava_constants.DEFAULT_IMAGE_TOKEN + "\n" + prompt

    # Formats the prompt as a conversation to be fed to the model.
    conv = conversation.conv_llava_v1.copy()
    conv.append_message(role=conv.roles[0], message=prompt)
    conv.append_message(role=conv.roles[1], message=None)
    prompt = conv.get_prompt()

    # Tokenizes the prompt that includes special image token as well.
    input_ids = (
        mm_utils.tokenizer_image_token(
            prompt=prompt,
            tokenizer=self.tokenizer,
            image_token_index=llava_constants.IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        )
        .unsqueeze(0)
        .to(self.device)
    )

    images = [
        image_format_converter.base64_to_image(image_str=base64_image).convert(
            "RGB"
        )
    ]
    # Gets the image embedding.
    images_tensor = mm_utils.process_images(
        images=images,
        image_processor=self.image_processor,
        model_cfg=self.model.config,
    ).to(self.device, dtype=torch.float16)

    self.stop_str = conversation.conv_llava_v1.sep2
    self.keywords = [self.stop_str]

    return input_ids, images_tensor

  def inference(
      self, input_ids: List[torch.Tensor], images_tensor: torch.Tensor
  ) -> List[torch.Tensor]:
    """Runs the inference."""
    stopping_criteria = mm_utils.KeywordsStoppingCriteria(
        keywords=self.keywords, tokenizer=self.tokenizer, input_ids=input_ids
    )

    with torch.inference_mode():
      output_ids = self.model.generate(
          input_ids=input_ids,
          images=images_tensor,
          do_sample=False,
          temperature=0,
          top_p=None,
          num_beams=1,
          max_new_tokens=512,
          use_cache=True,
          stopping_criteria=[stopping_criteria],
      )

    return output_ids

  def postprocess(
      self, output_ids: List[torch.Tensor], input_token_len: int
  ) -> List[str]:
    """Runs the postprocessing to convert token ids to string."""
    outputs = self.tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0]
    outputs = outputs.strip()
    if outputs.endswith(self.stop_str):
      outputs = outputs[: -len(self.stop_str)]
    outputs = outputs.strip()

    return [outputs]

  def handle(self, data: List[Dict[str, Any]], context: Any) -> List[str]:
    """Handles an incoming request by passing it through `preprocess`, `inference`, and `postprocess`."""
    input_ids, images_tensor = self.preprocess(data=data)
    model_output = self.inference(
        input_ids=input_ids, images_tensor=images_tensor
    )

    input_token_len = input_ids.shape[1]
    return self.postprocess(
        output_ids=model_output, input_token_len=input_token_len
    )
