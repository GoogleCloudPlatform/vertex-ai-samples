"""Instruct/Chat with LoRA models."""

# pylint: disable=g-importing-member
from datasets import load_dataset
from peft import LoraConfig
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import TrainingArguments
from trl import SFTTrainer


def finetune_instruct(
    pretrained_model_id: str,
    dataset_name: str,
    output_dir: str,
    lora_rank: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    warmup_ratio: int = 0.03,
    max_steps: int = 10,
    max_seq_length: int = 512,
    learning_rate: float = 2e-4,
) -> None:
  """Finetunes instruct."""
  dataset = load_dataset(dataset_name, split="train")

  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.float16,
  )

  model = AutoModelForCausalLM.from_pretrained(
      pretrained_model_id,
      quantization_config=bnb_config,
      trust_remote_code=True,
  )
  model.config.use_cache = False

  tokenizer = AutoTokenizer.from_pretrained(
      pretrained_model_id, trust_remote_code=True
  )
  tokenizer.pad_token = tokenizer.eos_token

  peft_config = LoraConfig(
      lora_alpha=lora_alpha,
      lora_dropout=lora_dropout,
      r=lora_rank,
      bias="none",
      task_type="CAUSAL_LM",
      target_modules=[
          "query_key_value",
          "dense",
          "dense_h_to_4h",
          "dense_4h_to_h",
      ],
  )

  per_device_train_batch_size = 4
  gradient_accumulation_steps = 4
  optim = "paged_adamw_32bit"
  save_steps = 10
  logging_steps = 10
  max_grad_norm = 0.3
  lr_scheduler_type = "constant"

  training_arguments = TrainingArguments(
      output_dir=output_dir,
      per_device_train_batch_size=per_device_train_batch_size,
      gradient_accumulation_steps=gradient_accumulation_steps,
      optim=optim,
      save_steps=save_steps,
      logging_steps=logging_steps,
      learning_rate=learning_rate,
      fp16=True,
      max_grad_norm=max_grad_norm,
      max_steps=max_steps,
      warmup_ratio=warmup_ratio,
      group_by_length=True,
      lr_scheduler_type=lr_scheduler_type,
  )

  trainer = SFTTrainer(
      model=model,
      train_dataset=dataset,
      peft_config=peft_config,
      dataset_text_field="text",
      max_seq_length=max_seq_length,
      tokenizer=tokenizer,
      args=training_arguments,
  )
  for name, module in trainer.model.named_modules():
    if "norm" in name:
      module = module.to(torch.float32)
  trainer.train()
