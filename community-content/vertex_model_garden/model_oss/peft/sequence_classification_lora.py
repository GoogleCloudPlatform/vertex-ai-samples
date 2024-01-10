"""Sequence classification with LoRA models."""

# pylint: disable=g-importing-member

from datasets import load_dataset
import evaluate
from peft import get_peft_model
from peft import LoraConfig
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup


def finetune_sequence_classification(
    pretrained_model_id: str,
    dataset_name: str,
    output_dir: str,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    num_epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
) -> None:
  """Finetunes sequence classification."""
  task = "mrpc"
  device = "cuda"

  peft_config = LoraConfig(
      task_type="SEQ_CLS",
      inference_mode=False,
      r=lora_rank,
      lora_alpha=lora_alpha,
      lora_dropout=lora_dropout,
  )
  if any(k in pretrained_model_id for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
  else:
    padding_side = "right"

  tokenizer = AutoTokenizer.from_pretrained(
      pretrained_model_id, padding_side=padding_side
  )
  if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

  datasets = load_dataset(dataset_name, task)
  metric = evaluate.load(dataset_name, task)

  def tokenize_function(examples):
    # max_length=None => use the model max length (it's actually the default)
    outputs = tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        truncation=True,
        max_length=None,
    )
    return outputs

  tokenized_datasets = datasets.map(
      tokenize_function,
      batched=True,
      remove_columns=["idx", "sentence1", "sentence2"],
  )

  # We also rename the 'label' column to 'labels' which is the expected name for
  # labels by the models of the transformers library.
  tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

  def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")

  # Instantiate dataloaders.
  train_dataloader = DataLoader(
      tokenized_datasets["train"],
      shuffle=True,
      collate_fn=collate_fn,
      batch_size=batch_size,
  )
  eval_dataloader = DataLoader(
      tokenized_datasets["validation"],
      shuffle=False,
      collate_fn=collate_fn,
      batch_size=batch_size,
  )

  model = AutoModelForSequenceClassification.from_pretrained(
      pretrained_model_id, return_dict=True
  )
  model = get_peft_model(model, peft_config)
  model.print_trainable_parameters()

  optimizer = AdamW(params=model.parameters(), lr=learning_rate)

  # Instantiate scheduler
  lr_scheduler = get_linear_schedule_with_warmup(
      optimizer=optimizer,
      num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs),
      num_training_steps=(len(train_dataloader) * num_epochs),
  )

  model.to(device)
  for epoch in range(num_epochs):
    model.train()
    for _, batch in enumerate(tqdm(train_dataloader)):
      batch.to(device)
      outputs = model(**batch)
      loss = outputs.loss
      loss.backward()
      optimizer.step()
      lr_scheduler.step()
      optimizer.zero_grad()

    model.eval()
    for _, batch in enumerate(tqdm(eval_dataloader)):
      batch.to(device)
      with torch.no_grad():
        outputs = model(**batch)
      predictions = outputs.logits.argmax(dim=-1)
      references = batch["labels"]
      metric.add_batch(
          predictions=predictions,
          references=references,
      )

    eval_metric = metric.compute()
    print(f"epoch {epoch}:", eval_metric)

  model.save_pretrained(output_dir)
