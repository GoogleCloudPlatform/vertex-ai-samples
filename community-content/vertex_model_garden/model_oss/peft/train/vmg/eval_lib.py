"""Library for running evaluations during training."""

import dataclasses
from typing import Any, Optional, Type

from absl import logging
import datasets
from lm_eval import evaluator
from lm_eval import tasks
from lm_eval import utils
from lm_eval.api import model as lm_model
from lm_eval.api import registry
from lm_eval.models import huggingface
from peft import peft_model
import transformers
from transformers import trainer

from util import dataset_validation_util
from util import constants


_DESCRIPTION_EVALUATION = "evaluation"
_BUILTIN_EVAL_TASK = "builtin_eval"


@dataclasses.dataclass(frozen=True)
class EvalConfig:
  steps: int
  tasks: list[str]
  per_device_batch_size: int
  num_fewshot: Optional[int]
  limit: Optional[float]
  metric_name: str
  tokenize_dataset: bool
  dataset_path: str = ""
  split: str = "test"
  template: str = ""
  column: str = constants.DEFAULT_INSTRUCT_COLUMN_IN_DATASET


class PeftCausalLMModel(huggingface.HFLM):
  """PeftCausalLMModel that supports loading an in-memory model."""

  AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

  def __init__(
      self,
      model: peft_model.PeftModelForCausalLM,
      tokenizer: transformers.PreTrainedTokenizerBase,
      batch_size_per_gpu: int,
  ):
    lm_model.LM.__init__(self)
    self._model = model
    self.tokenizer = tokenizer
    self.vocab_size = tokenizer.vocab_size
    tokenizer.pad_token_id = tokenizer.eos_token_id
    self._config = model.config
    self.batch_size_per_gpu = batch_size_per_gpu
    self._device = model.device
    self._max_length = None  # Will be automatically determined from config.
    self._add_special_tokens = (
        None  # Will be automatically determined from AUTO_MODEL_CLASS.
    )


def create_trainer(
    cls: Type[transformers.Trainer],
    eval_config: Optional[EvalConfig],
    tokenizer: Optional[transformers.PreTrainedTokenizerBase],
    args: trainer.TrainingArguments,
    **kwargs,
) -> transformers.Trainer:
  """Creates a trainer. If eval config is provided, injects evaluation loop."""
  if not eval_config:
    return cls(args=args, **kwargs)

  args.eval_strategy = "steps"
  args.eval_steps = eval_config.steps
  args.per_device_eval_batch_size = eval_config.per_device_batch_size
  kwargs["tokenizer"] = tokenizer

  if eval_config.tasks == [_BUILTIN_EVAL_TASK]:
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

  class LMEvalTrainer(cls):
    """Trainer with lm_eval injected as the eval library."""

    def __init__(self, **kwargs):
      super().__init__(**kwargs)
      task_names = utils.pattern_match(eval_config.tasks, registry.ALL_TASKS)
      logging.info("Selected Eval Tasks: %s", task_names)
      task_args = {}
      if eval_config.num_fewshot is not None:
        task_args["num_fewshot"] = eval_config.num_fewshot
      if eval_config.dataset_path:
        task_args["dataset_path"] = "json"
        task_args["dataset_kwargs"] = {
            "data_files": {"test": eval_config.dataset_path},
        }
      self._eval_task_dict = tasks.get_task_dict(task_names, **task_args)

    def evaluation_loop(
        self,
        dataloader: trainer.DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> trainer.EvalLoopOutput:
      """Custom evaluation loop that invokes lm_eval."""
      if description.lower() != _DESCRIPTION_EVALUATION:
        return super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )

      model = self._wrap_model(self.model, training=False)
      lm = PeftCausalLMModel(
          model,
          self.tokenizer or self.data_collator.tokenizer,
          eval_config.per_device_batch_size,
      )
      results: dict[str, Any] = evaluator.evaluate(
          lm=lm,
          task_dict=self._eval_task_dict,
          limit=eval_config.limit,
      )["results"]
      metric_name = eval_config.metric_name
      # Compute average value if there are multiple tasks.
      metric_values: list[float] = []
      for result in results.values():
        for key, value in result.items():
          if key.split(",")[0] == metric_name:
            metric_values.append(value)
      if not metric_values:
        raise ValueError(
            f"Metric {metric_name} not found in eval response: {results}"
        )
      metric_average = sum(metric_values) / len(metric_values)
      logging.info("%s value: %f\n%s", metric_name, metric_average, results)
      return trainer.EvalLoopOutput(
          # Only metrics field is set. Other fields are dummy values.
          predictions=None,
          label_ids=None,
          metrics={f"{metric_key_prefix}_{metric_name}": metric_average},
          num_samples=0,
      )

  # Use empty eval dataset as a placeholder.
  return LMEvalTrainer(
      args=args, eval_dataset=datasets.Dataset.from_dict({"test": []}), **kwargs
  )
