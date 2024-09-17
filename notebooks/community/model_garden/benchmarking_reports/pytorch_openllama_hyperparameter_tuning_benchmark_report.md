# Benchmark report on hyperparameter tuning the OpenLLaMA models on Google Cloud Vertex Model Garden

Changyu Zhu, Software Engineer, Google Cloud

Dustin Luong, Software Engineer, Google Cloud

Gary Wei, Software Engineer, Google Cloud

Genquan Duan, Software Engineer, Google Cloud

## Introduction

Fine-tuning of LLMs can be non-trivial to find an optimal configuration of
machine types, training parameters, and other hyperparameters that achieves a
good balance between cost efficiency and model performance. To facilitate users
in conducting tuning experiments, this report benchmarks fine-tuning OpenLLaMA
models with [Vertex AI Hyperparameter Tuning Service](https://cloud.google.com/vertex-ai/docs/training/hyperparameter-tuning-overview), demonstrating both efficiency
and effectiveness. Similar hyperparameter tuning techniques can apply to other models as well.

## Key takeaways

- **The hyperparameter tuning service finds good parameters**: The best model found by the hyperparameter tuning service has an average improvement of around 4% in accuracy in *ARC*, *HellaSwag*, and *TruthfulQA* datasets, while only tuning the learning rate.

- **Hyperparameter tuning works with QLoRA on limited resources**: 4bit QLoRA is sufficient for hyperparameter tuning to find a set of good parameters. In this way, all OpenLLaMA models can run on 1 single `NVIDIA_L4` GPU. It is also possible to train for more steps on the good parameters discovered by hyperparameter tuning, avoiding the waste of computing resources on fine-tuning with suboptimal hyperparameters.

- **Hyperparameter tuning is cost-effective**: While `NVIDIA_L4` is slower than `NVIDIA_TESLA_V100`, it costs less and avoids the overhead of multi-GPU training since it has more GPU memory. Finding a good 3B/7B/13B OpenLLaMA model costs $28.5671, $47.8016, and $87.9208, respectively.

## Benchmarking setup

This section describes the experiment setup of the hyperparameter tuning experiments. The default tuning parameters are:

### Machine configuration

- Machine type: g2-standard-8
- Machine count: 1
- Accelerator type: NVIDIA_L4
- Accelerator count: 1

### Modeling

We benchmark all 3 OpenLLaMA models:

- [open_llama_3b](https://huggingface.co/openlm-research/open_llama_3b)
- [open_llama_7b](https://huggingface.co/openlm-research/open_llama_7b)
- [open_llama_13b](https://huggingface.co/openlm-research/open_llama_13b)

We use the Huggingface [PEFT](https://github.com/huggingface/peft) library for fine-tuning.

### Training dataset

We use the dataset [timdettmers/openassistant-guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco) loaded directly via Huggingface.

### Training parameters

The set of training parameters used during benchmarking:

- Batch size: 4
- Precision mode: 4bit QLoRA
- LoRA rank: 32
- LoRA alpha: 64
- Max sequence length: 512
- Max train steps: 1000

### Evaluation dataset

We use the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) library injected into the training loop for evaluation. The hyperparameter tuning job will pick the model according to the evaluation metrics.

- Eval task: [ARC Challenge](https://huggingface.co/datasets/ai2_arc)
- Eval metric: acc_norm
- Max eval examples: 10000

### Standalone evaluation dataset

After finding the best model with Vertex hyperparameter tuning service, we run standalone evaluations with the model on the following datasets:

- [ARC Challenge](https://huggingface.co/datasets/ai2_arc)
- [HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag)
- [TruthfulQA](https://huggingface.co/datasets/EleutherAI/truthful_qa_mc)

### Hyperparameter tuning

We only tune the learning rate hyperparameter. It is considered a floating point value in the continuous range [1e-5, 1e-4]. We run 8 trials in total, with a parallelism of 1 or 2.

### Code example

The following code example launches an example hyperparameter tuning job of OpenLLaMA 7B model.

```py
from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt


TRAIN_DOCKER_URI = 'us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-peft-train:20231130_0936_RC00'
output_dir = "gs://path/to/output/dir"
base_model_id = "openlm-research/open_llama_7b"
dataset_name = "timdettmers/openassistant-guanaco"
hpt_precision_mode = "4bit"
machine_type = "g2-standard-8"
accelerator_type = "NVIDIA_L4"
accelerator_count = 1
eval_task = "arc_challenge"
eval_metric_name = "acc_norm"
max_steps = 1000
eval_limit = 10000

flags = {
    "learning_rate": 1e-5,
    "precision_mode": hpt_precision_mode,
    "task": "instruct-lora",
    "pretrained_model_id": base_model_id,
    "output_dir": output_dir,
    "warmup_steps": 10,
    "max_steps": max_steps,
    "lora_rank": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "dataset_name": dataset_name,
    "eval_steps": max_steps + 1,  # Only evaluates at the end.
    "eval_tasks": eval_task,
    "eval_limit": eval_limit,
    "eval_metric_name": eval_metric_name,
}
worker_pool_specs = [
    {
        "machine_spec": {
            "machine_type": machine_type,
            "accelerator_type": accelerator_type,
            "accelerator_count": accelerator_count,
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": TRAIN_DOCKER_URI,
            "args": ["--{}={}".format(k, v) for k, v in flags.items()],
        },
    }
]
metric_spec = {"model_performance": "maximize"}
parameter_spec = {
    "learning_rate": hpt.DoubleParameterSpec(
        min=1e-5, max=1e-4, scale="linear"
    ),
}

train_job = aiplatform.CustomJob(
    display_name=job_name,
    worker_pool_specs=worker_pool_specs,
    staging_bucket=STAGING_BUCKET,
)

train_hpt_job = aiplatform.HyperparameterTuningJob(
    display_name=f"{job_name}_hpt",
    custom_job=train_job,
    metric_spec=metric_spec,
    parameter_spec=parameter_spec,
    max_trial_count=8,
    parallel_trial_count=2,
)

train_hpt_job.run()
```

## Benchmark results

### Fine-tuning cost

The fine-tuning cost is calculated from `us-central1` pricing and may be subject to changes.

| Model         | Train time | Trials | Parallel Trials | Hourly cost | Cost     | Eval acc_norm (ARC-Challenge) |
|---------------|------------|--------|-----------------|-------------|----------|-------------------------------|
| OpenLLaMA 3B  | 16 hrs     | 8      | 2               | $1.7072     | $28.5671 | 39.9%                         |
| OpenLLaMA 7B  | 28 hrs     | 8      | 2               | $1.7072     | $47.8016 | 45.8%                         |
| OpenLLaMA 13B | 103 hrs    | 8      | 1               | $0.8536     | $87.9208 | 47.6%                         |

### Fine-tuning performance

Here are the evaluation results of the best model found by hyperparameter tuning, compared with the baseline model. The column `Eval acc_norm` is calculated during training, which is always lower than that during standalone evaluation, because the model is loaded and evaluated at a lower precision (4bit during training / float16 during standalone evaluation).

| Model         | Eval acc_norm (ARC-Challenge) | ARC    | hellaswag | Truthfulqa_mc | ∆ARC   | ∆Hellaswag | ∆Truthfulqa_mc | ∆Average |
|---------------|-------------------------------|--------|-----------|---------------|--------|------------|----------------|----------|
| OpenLLaMA 3B  | 39.9%                         | 41.47% | 69.97%    | 38.31%        | +1.62% | +7.32%     | +3.34%         | +4.09%   |
| OpenLLaMA 7B  | 45.8%                         | 49.83% | 75.53%    | 41.53%        | +2.82% | +3.55%     | +6.68%         | +4.35%   |
| OpenLLaMA 13B | 47.6%                         | 52.20% | 78.90%    | 44.27%        | +1.01% | +3.67%     | +6.19%         | +3.62%   |

## Related documents

1. [Benchmark report on fine tuning the OpenLLaMA 7B model on Google Cloud Vertex Model Garden
](
https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/community-content/vertex_model_garden/benchmarking_reports/pytorch_openllama_7b_finetune_benchmark_report.md)
