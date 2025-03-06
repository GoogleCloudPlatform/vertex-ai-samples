"""Library for running evaluations during training."""

from collections.abc import Callable, Mapping, MutableMapping, Sequence
import dataclasses
import string
from typing import Type

from absl import logging
import evaluate
import numpy as np
import torch
import transformers

from util import dataset_validation_util
from util import constants


_STRING_TRANSLATOR = str.maketrans("", "", string.punctuation)

_GREATER_IS_BETTER_MAP = {
    "loss": False,
    "perplexity": False,
    "bleu": True,
    "google_bleu": True,
    "rouge1": True,
    "rouge2": True,
    "rougeL": True,
    "rougeLsum": True,
}


@dataclasses.dataclass(frozen=True)
class EvalConfig:
  """Configuration for running evaluations during training.

  Attributes:
    steps: The number of steps to run evaluation.
    tasks: The list of tasks to run evaluation on.
    per_device_batch_size: The per device batch size for evaluation.
    num_fewshot: The number of few-shot examples to use for evaluation.
    limit: The maximum number of examples to evaluate.
    metric_name: The name of the metric to compute.
    tokenize_dataset: Whether to tokenize the dataset.
    dataset_path: The path to the dataset.
    split: The split of the dataset to evaluate.
    template: The template to use for the dataset.
    column: The column name of the dataset.
    metric_for_best_model: The metric to use for loading the best model.
  """

  steps: int
  per_device_batch_size: int
  num_fewshot: int | None
  limit: float | None
  metric_name: Sequence[str]
  tokenize_dataset: bool
  dataset_path: str = ""
  split: str = "test"
  template: str = ""
  column: str = constants.DEFAULT_TRAIN_COLUMN
  metric_for_best_model: str | None = None


def create_trainer(
    cls: Type[transformers.Trainer],
    eval_config: EvalConfig | None,
    tokenizer: transformers.PreTrainedTokenizerBase | None,
    args: transformers.TrainingArguments,
    **kwargs,
) -> transformers.Trainer:
  """Creates a trainer. If eval config is provided, injects evaluation loop.

  Args:
    cls: The trainer class.
    eval_config: The evaluation config.
    tokenizer: The tokenizer.
    args: The training arguments.
    **kwargs: The keyword arguments.

  Returns:
    A trainer.
  """
  if not eval_config:
    return cls(args=args, **kwargs)

  args.eval_strategy = "steps"
  args.eval_steps = eval_config.steps
  args.per_device_eval_batch_size = eval_config.per_device_batch_size
  args.metric_for_best_model = eval_config.metric_for_best_model
  args.greater_is_better = _GREATER_IS_BETTER_MAP.get(
      eval_config.metric_for_best_model, None
  )
  args.save_strategy = (
      transformers.trainer_utils.SaveStrategy.STEPS
      if eval_config.metric_for_best_model is None
      else transformers.trainer_utils.SaveStrategy.BEST
  )

  kwargs["tokenizer"] = tokenizer

  try:
    eval_dataset = dataset_validation_util.load_dataset_with_template(
        dataset_name=eval_config.dataset_path,
        split=eval_config.split,
        input_column=eval_config.column,
        template=eval_config.template,
        tokenizer=tokenizer,
    )
    if eval_config.limit is not None:
      if eval_config.limit >= 1:
        limit = int(eval_config.limit)
      else:
        limit = int(eval_config.limit * len(eval_dataset))
      eval_dataset = eval_dataset.select(range(limit))
    if tokenizer is not None:
      eval_dataset = dataset_validation_util.get_filtered_dataset(
          dataset=eval_dataset,
          input_column=eval_config.column,
          max_seq_length=kwargs["max_seq_length"],
          tokenizer=tokenizer,
      )
    if eval_config.tokenize_dataset:
      eval_dataset = eval_dataset.map(
          lambda samples: tokenizer(samples[eval_config.column])
      )
    kwargs["eval_dataset"] = eval_dataset
  except (OSError, ValueError, IndexError) as e:
    logging.warning(
        "Failed to load eval dataset %s. Evaluation will be skipped.\n%s",
        eval_config.dataset_path,
        e,
    )
    del args.evaluation_strategy
    del args.eval_steps
    del args.per_device_eval_batch_size
  return cls(args=args, **kwargs)


def _cleanup_text(text: str) -> str:
  """Cleans up the prediction and references text.

  Args:
    text: The text to clean up.

  Returns:
    Cleaned up text.
  """
  text = text.translate(_STRING_TRANSLATOR)
  text = text.strip()
  text = " ".join(text.split())
  return text.lower()


def create_compute_metrics(
    tokenizer: transformers.PreTrainedTokenizerBase,
    eval_metrics: Mapping[str, evaluate.EvaluationModule],
) -> Callable[[transformers.EvalPrediction], MutableMapping[str, float]]:
  """Creates a compute_metrics function using Hugging Face evaluate library.

  Args:
    tokenizer: The tokenizer for decoding predictions.
    eval_metrics: The eval metrics to compute.

  Returns:
    Function that computes comprehensive metrics.
  """

  def _preprocess_data(
      predictions: np.ndarray, labels: np.ndarray
  ) -> tuple[Sequence[str], Sequence[str]]:
    """Preprocesses predictions and lavels before evaluation.

    Args:
      predictions: The predictions to preprocess.
      labels: The labels to preprocess.

    Returns:
      A tuple (preprocessed predictions, labels).
    """
    # Handle padding and special tokens.
    predictions = np.where(
        predictions != -100, predictions, tokenizer.pad_token_id
    )
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode to text.
    pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Clean up text.
    cleaned_pred_texts = [_cleanup_text(text) for text in pred_texts]
    cleaned_label_texts = [_cleanup_text(text) for text in label_texts]

    return cleaned_pred_texts, cleaned_label_texts

  def _compute_metrics_with_tokenizer(
      eval_pred: transformers.EvalPrediction,
  ) -> MutableMapping[str, float]:
    """Computes metrics using Hugging Face evaluate library.

    Args:
      eval_pred: The evaluation prediction.

    Returns:
      A dictionary of metrics.
    """
    predictions, perplexities = eval_pred.predictions
    labels = eval_pred.label_ids

    pred_texts, label_texts = _preprocess_data(predictions, labels)

    metrics = {}

    for eval_metric, computed_eval_metric in eval_metrics.items():
      match eval_metric:
        case "perplexity":
          # We don't use the perplexity from HF Evaluate since it loads the
          # model again. This causes an increase in the GPU utilization and
          # hence an OOM. Due to this, we compute the perplexity ourselves
          # using the eval_loss over the unmasked tokens in
          # preprocess_logits_for_metrics fn.
          metrics[eval_metric] = np.mean(perplexities)
        case "bleu" | "google_bleu":
          num_valid_labels = len(list(filter(None, label_texts)))
          if num_valid_labels:
            eval_score = computed_eval_metric.compute(
                predictions=pred_texts,
                references=[[text] for text in label_texts],
            )
            metrics[eval_metric] = eval_score[eval_metric]
          else:
            metrics[eval_metric] = 0.0
        case "rouge1" | "rouge2" | "rougeL" | "rougeLsum":
          rouge_scores = computed_eval_metric.compute(
              predictions=pred_texts,
              references=label_texts,
              use_stemmer=True,
          )
          metrics[eval_metric] = rouge_scores[eval_metric]

    pred_lengths = [len(pred.split()) for pred in pred_texts]
    label_lengths = [len(label.split()) for label in label_texts]

    metrics["gen_len"] = np.mean(pred_lengths)
    metrics["ref_len"] = np.mean(label_lengths)
    metrics["length_ratio"] = np.mean(
        [len(p) / len(r) if r else 0 for p, r in zip(pred_texts, label_texts)]
    )

    # Round all metrics to 4 decimal places.
    return {k: round(float(v), 4) for k, v in metrics.items()}

  return _compute_metrics_with_tokenizer


def preprocess_logits_for_metrics(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """Preprocesses the logits before caching them for eval metric calculation.

  Args:
    logits: Logits predicted by the model.
    labels: Ground truth labels.

  Returns:
    A tuple (pred_ids, perplexities).
  """
  # Calculate prediction IDs.
  pred_ids = logits.argmax(dim=-1)

  # This step shifts the logits and labels to align them correctly, where we are
  # predicting the next token in a sequence. The last logit doesn't have a
  # corresponding label, and the first label doesn't have a preceding logit to
  # predict it. This calculation of perplexity is inspired from
  # https://github.com/huggingface/evaluate/blob/main/metrics/perplexity/perplexity.py.
  shift_logits = logits[..., :-1, :].contiguous()
  shift_labels = labels[..., 1:].contiguous()
  attn_mask = shift_labels != -100
  loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

  perplexities = torch.exp(
      (loss_fct(shift_logits.transpose(1, 2), shift_labels) * attn_mask).sum(1)
      / attn_mask.sum(1)
  )

  return (pred_ids, perplexities)
