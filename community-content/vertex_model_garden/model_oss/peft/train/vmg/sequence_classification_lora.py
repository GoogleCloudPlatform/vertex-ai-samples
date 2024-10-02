"""Sequence classification with LoRA models."""

from typing import Sequence

from absl import app
from absl import flags
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

from util import dataset_validation_util


_PRETRAINED_MODEL_ID = flags.DEFINE_string(
    "pretrained_model_id",
    None,
    "The pretrained model id. Supported models can be causal language modeling"
    " models from https://github.com/huggingface/peft/tree/main. Note, there"
    " might be different paddings for different models. This tool assumes the"
    " pretrained_model_id contains model name, and then choose proper padding"
    " methods. e.g. it must contain `llama` for `Llama2 models`.",
)

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "The output directory.",
)

_DATASET_NAME = flags.DEFINE_string(
    "dataset_name",
    None,
    "The dataset name in huggingface.",
)

_LORA_RANK = flags.DEFINE_integer(
    "lora_rank",
    16,
    "The rank of the update matrices, expressed in int. Lower rank results in"
    " smaller update matrices with fewer trainable parameters, referring to"
    " https://huggingface.co/docs/peft/conceptual_guides/lora.",
)

_LORA_ALPHA = flags.DEFINE_integer(
    "lora_alpha",
    32,
    "LoRA scaling factor, referring to"
    " https://huggingface.co/docs/peft/conceptual_guides/lora.",
)

_LORA_DROPOUT = flags.DEFINE_float(
    "lora_dropout",
    0.05,
    "dropout probability of the LoRA layers, referring to"
    " https://huggingface.co/docs/peft/task_guides/token-classification-lora.",
)

_NUM_EPOCHS = flags.DEFINE_integer(
    "num_epochs",
    None,
    "The number of training epochs.",
)

_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size",
    32,
    "The batch size.",
)

_LEARNING_RATE = flags.DEFINE_float(
    "learning_rate",
    2e-4,
    "The learning rate after the potential warmup period.",
)


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


def main(unused_argv: Sequence[str]) -> None:
  if dataset_validation_util.is_gcs_path(_PRETRAINED_MODEL_ID.value):
    pretrained_model_id = dataset_validation_util.download_gcs_uri_to_local(
        _PRETRAINED_MODEL_ID.value
    )
  else:
    pretrained_model_id = _PRETRAINED_MODEL_ID.value
  pretrained_model_path = dataset_validation_util.force_gcs_fuse_path(
      pretrained_model_id
  )
  output_dir = dataset_validation_util.force_gcs_fuse_path(_OUTPUT_DIR.value)

  finetune_sequence_classification(
      pretrained_model_id=pretrained_model_path,
      dataset_name=_DATASET_NAME.value,
      output_dir=output_dir,
      lora_rank=_LORA_RANK.value,
      lora_alpha=_LORA_ALPHA.value,
      lora_dropout=_LORA_DROPOUT.value,
      num_epochs=int(_NUM_EPOCHS.value),
      batch_size=_BATCH_SIZE.value,
      learning_rate=_LEARNING_RATE.value,
  )


if __name__ == "__main__":
  app.run(main)
