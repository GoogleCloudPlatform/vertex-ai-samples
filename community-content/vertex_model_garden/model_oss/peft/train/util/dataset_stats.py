"""Calculate dataset statistics like token, example and character counts."""

from collections.abc import Mapping, Sequence
import dataclasses
import json
from typing import Any
import datasets
import numpy as np
import transformers
from util import dataset_validation_util

_MAX_NUM_DATASET_SAMPLES = 6


@dataclasses.dataclass
class SupervisedTuningDatasetBucket:
  """Represents a histogram bucket for tuning dataset distribution stats."""

  count: float = 0
  left: float = 0
  right: float = 0


@dataclasses.dataclass
class SupervisedTuningDatasetDistribution:
  """Represents a histogram with summary statistics for tuning dataset distribution stats."""

  sum: int = 0
  billable_sum: int = 0
  min: float = 0
  max: float = 0
  mean: float = 0
  median: float = 0
  p5: float = 0
  p95: float = 0
  buckets: list[SupervisedTuningDatasetBucket] = dataclasses.field(
      default_factory=list
  )


# Represents detailed tuning dataset statistics.
@dataclasses.dataclass
class SupervisedTuningDataStats:
  """Represents detailed tuning dataset stats."""

  tuning_dataset_example_count: int = 0
  total_tuning_character_count: int = 0
  total_billable_token_count: int = 0
  tuning_step_count: int = 0
  # Represents a histogram and some summary statistics of the number of input
  # tokens across examples.
  user_input_token_distribution: SupervisedTuningDatasetDistribution | None = (
      None
  )
  # Represents a histogram and some summary statistics for the number of output
  # tokens across examples.
  user_output_token_distribution: SupervisedTuningDatasetDistribution | None = (
      None
  )
  # Represents the number of "messages" (a single-turn conversation will have a
  # single message) across examples.
  user_message_per_example_distribution: (
      SupervisedTuningDatasetDistribution | None
  ) = None
  user_dataset_examples: list[str] = dataclasses.field(default_factory=list)


def get_dataset_stats(
    *,
    raw: Any,
    templated: Any,
    template: str,
    tokenizer: transformers.PreTrainedTokenizer,
    column: str,
    effective_batch_size: int,
) -> Mapping[str, Any]:
  """Calculates dataset statistics for managed fine-tuning, e.g., total number of tokens."""
  tokenized_dataset = templated.map(lambda x: tokenizer(x[column]))
  inputs = tokenized_dataset["input_ids"]
  tuning_dataset_example_count = int(len(inputs))
  total_billable_token_count = int(np.sum([len(ex) for ex in inputs]))
  total_tuning_character_count = int(
      np.sum([len(ex[column]) for ex in templated])
  )
  tuning_step_count = (
      tuning_dataset_example_count + effective_batch_size - 1
  ) // effective_batch_size

  # Assume that data is represented as ChatCompletions or Vertex Text-Bison
  # formats to extract per-example input/output tokens.
  user_inputs = []
  user_outputs = []
  user_input_messages_counts = []

  for ex in raw:
    if "messages" in ex:
      messages = ex["messages"]
      if messages:
        # For ChatCompletions assume the last turn (i.e. the instruction
        # response) is the expected output.
        user_inputs.append({**ex, "messages": messages[:-1]})
        user_outputs.append({**ex, "messages": messages[-1:]})
        # Exclude everything but the last message for the number of input
        # messages.
        user_input_messages_counts.append(len(messages[:-1]))
    elif "input_text" in ex:
      # For Vertex Text-Bison, the `output_text` field is the expected output.
      user_inputs.append({**ex, "output_text": ""})
      user_outputs.append(
          {**ex, "input_text": ex["output_text"], "output_text": ""}
      )
      # Vertex Text-Bison goes from input -> output; i.e. there is only a single
      # input "message".
      user_input_messages_counts.append(1)

  def calc_histogram(
      counts: Sequence[int],
  ) -> SupervisedTuningDatasetDistribution:
    mean = np.mean(counts)
    median = np.median(counts).item()
    max_count = np.max(counts).item()
    min_count = np.min(counts).item()
    count_sum = np.sum(counts).item()
    p5 = np.percentile(counts, 0.05).item()
    p95 = np.percentile(counts, 0.95).item()
    hist, bin_edges = np.histogram(counts, bins=10)

    return SupervisedTuningDatasetDistribution(
        sum=count_sum,
        billable_sum=count_sum,
        min=min_count,
        max=max_count,
        mean=mean,
        median=median,
        p5=p5,
        p95=p95,
        buckets=[
            SupervisedTuningDatasetBucket(
                count=hist[i].item(),
                left=bin_edges[i].item(),
                right=bin_edges[i + 1].item(),
            )
            for i in range(len(hist))
        ],
    )

  # Tokenize input and output messages separately to generate separate summary
  # statistics about them.
  user_input_token_distribution = None
  if user_inputs:
    user_input_dataset = dataset_validation_util.format_dataset(
        datasets.Dataset.from_list(user_inputs), column, template, tokenizer
    )
    user_input_tokenized_dataset = user_input_dataset.map(
        lambda x: tokenizer(x[column])
    )
    user_input_tokens = user_input_tokenized_dataset["input_ids"]
    user_input_token_counts = np.array([len(ex) for ex in user_input_tokens])
    user_input_token_distribution = calc_histogram(user_input_token_counts)

  user_output_token_distribution = None
  if user_outputs:
    user_output_dataset = dataset_validation_util.format_dataset(
        datasets.Dataset.from_list(user_outputs), column, template, tokenizer
    )
    user_output_tokenized_dataset = user_output_dataset.map(
        lambda x: tokenizer(x[column])
    )
    user_output_tokens = user_output_tokenized_dataset["input_ids"]
    user_output_token_counts = np.array([len(ex) for ex in user_output_tokens])
    user_output_token_distribution = calc_histogram(user_output_token_counts)

  user_messages_per_example_distribution = None
  if user_input_messages_counts:
    user_input_messages_counts = np.array(user_input_messages_counts)
    user_messages_per_example_distribution = calc_histogram(
        user_input_messages_counts
    )

  user_dataset_examples = [
      json.dumps(ex)
      for ex in raw.shuffle().select(
          range(min(len(raw), _MAX_NUM_DATASET_SAMPLES))
      )
  ]

  dataset_stats = SupervisedTuningDataStats(
      tuning_dataset_example_count=tuning_dataset_example_count,
      total_tuning_character_count=total_tuning_character_count,
      total_billable_token_count=total_billable_token_count,
      tuning_step_count=tuning_step_count,
      user_input_token_distribution=user_input_token_distribution,
      user_output_token_distribution=user_output_token_distribution,
      user_message_per_example_distribution=user_messages_per_example_distribution,
      user_dataset_examples=user_dataset_examples,
  )
  return dataclasses.asdict(dataset_stats)
