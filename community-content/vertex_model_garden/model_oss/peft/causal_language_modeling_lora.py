"""Causal language modeling with LoRA models."""

# pylint: disable=g-importing-member

from datasets import load_dataset
from peft import get_peft_model
from peft import LoraConfig
import torch
from torch import nn
import transformers
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import TrainingArguments
from util import constants


def finetune_causal_language_modeling(
    pretrained_model_id: str,
    dataset_name: str,
    output_dir: str,
    precision_mode: str = None,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    warmup_steps: int = 10,
    max_steps: int = 10,
    learning_rate: float = 2e-4,
    local_pretrained_model_id: str = None,
) -> None:
  """Finetunes causal language modelings."""
  if precision_mode == constants.PRECISION_MODE_32:
    model = AutoModelForCausalLM.from_pretrained(
        local_pretrained_model_id
        if local_pretrained_model_id
        else pretrained_model_id,
        torch_dtype=torch.float32,
        device_map="auto",
    )
  elif precision_mode == constants.PRECISION_MODE_16:
    model = AutoModelForCausalLM.from_pretrained(
        local_pretrained_model_id
        if local_pretrained_model_id
        else pretrained_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
  elif precision_mode == constants.PRECISION_MODE_8:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True, int8_threshold=0
    )
    model = AutoModelForCausalLM.from_pretrained(
        local_pretrained_model_id
        if local_pretrained_model_id
        else pretrained_model_id,
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
        local_pretrained_model_id
        if local_pretrained_model_id
        else pretrained_model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )

  tokenizer = AutoTokenizer.from_pretrained(
      local_pretrained_model_id
      if local_pretrained_model_id
      else pretrained_model_id
  )
  if "llama" in pretrained_model_id:
    tokenizer.pad_token = "[PAD]"

  for param in model.parameters():
    # Freezes the model - train adapters later.
    param.requires_grad = False
    if param.ndim == 1:
      # Casts the small parameters (e.g. layernorm) to fp32 for stability.
      param.data = param.data.to(torch.float32)

  # Reduces the number of stored activations.
  model.gradient_checkpointing_enable()
  model.enable_input_require_grads()

  class CastOutputToFloat(nn.Sequential):

    def forward(self, x):
      return super().forward(x).to(torch.float32)

  model.lm_head = CastOutputToFloat(model.lm_head)

  config = LoraConfig(
      r=lora_rank,
      lora_alpha=lora_alpha,
      target_modules=["q_proj", "v_proj"],
      lora_dropout=lora_dropout,
      bias="none",
      task_type="CAUSAL_LM",
  )

  model = get_peft_model(model, config)
  model.print_trainable_parameters()

  data = load_dataset(dataset_name)
  data = data.map(
      lambda samples: tokenizer(samples["quote"]),
      batched=True,
  )

  trainer = transformers.Trainer(
      model=model,
      train_dataset=data["train"],
      args=TrainingArguments(
          per_device_train_batch_size=4,
          gradient_accumulation_steps=4,
          warmup_steps=warmup_steps,
          max_steps=max_steps,
          learning_rate=learning_rate,
          fp16=True,
          logging_steps=1,
          output_dir=output_dir,
          ddp_find_unused_parameters=False,
      ),
      data_collator=transformers.DataCollatorForLanguageModeling(
          tokenizer,
          mlm=False,
      ),
  )
  # Silence the warnings. Please re-enable for inference!
  model.config.use_cache = False
  trainer.train()

  model.save_pretrained(output_dir)
