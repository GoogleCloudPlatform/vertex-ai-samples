# GenAI Evaluation Metric Registry

Complete catalog of evaluation metrics available in the `vertexai._genai` SDK.
Source of truth: `vertexai/_genai/_evals_metric_loaders.py` and
`vertexai/_genai/_evals_metric_handlers.py`.

## Metric Type Hierarchy

```
Metric (base)
├── LLMMetric           — LLM-as-a-judge with prompt_template
├── CodeExecutionMetric — Sandboxed remote Python function
└── (base Metric)       — Local callable or predefined name
```

Access predefined metrics via `types.RubricMetric.<NAME>` (preferred).
`types.PrebuiltMetric` is an alias with identical behavior.

## Predefined API Metrics (AutoRater)

Server-side evaluation via Vertex AI AutoRater. No judge model needed.

### Single-Turn

| Metric name | What it measures | Required fields |
|---|---|---|
| `general_quality_v1` | Overall response quality | `prompt`, `response` |
| `text_quality_v1` | Text quality (grammar, clarity) | `prompt`, `response` |
| `instruction_following_v1` | How well response follows instructions | `prompt`, `response` |
| `grounding_v1` | Factual grounding in provided context | `prompt`, `response`, context |
| `safety_v1` | Safety assessment | `prompt`, `response` |
| `hallucination_v1` | Hallucination detection | `prompt`, `response` |
| `tool_use_quality_v1` | Quality of tool/function calling | `agent_data` with tool calls |

### Multi-Turn / Agent

| Metric name | What it measures | Required fields |
|---|---|---|
| `multi_turn_general_quality_v1` | Overall multi-turn quality | `agent_data` (1+ turns) |
| `multi_turn_text_quality_v1` | Text quality across turns | `agent_data` (1+ turns) |
| `multi_turn_tool_use_quality_v1` | Tool call quality across trajectory | `agent_data` with function calls |
| `multi_turn_trajectory_quality_v1` | Quality of agent's action sequence | `agent_data` with full trajectory |
| `multi_turn_task_success_v1` | Whether agent completed the task | `agent_data` with task context |

### Agent Final Response

| Metric name | What it measures | Required fields |
|---|---|---|
| `final_response_match_v2` | Reference-based final response matching | `agent_data`, `reference` |
| `final_response_reference_free_v1` | Final response quality (no reference) | `agent_data` |
| `final_response_quality_v1` | Final response quality | `agent_data` |

### Multimodal

| Metric name | What it measures | Required fields |
|---|---|---|
| `gecko_text2image_v1` | Text-to-image quality | image content |
| `gecko_text2video_v1` | Text-to-video quality | video content |

### Accessing predefined metrics

```python
from vertexai import types

# Via RubricMetric (preferred)
metric = types.RubricMetric.MULTI_TURN_TRAJECTORY_QUALITY

# Via PrebuiltMetric (alias — identical behavior)
metric = types.PrebuiltMetric.MULTI_TURN_TRAJECTORY_QUALITY

# With version override
metric = types.RubricMetric.GENERAL_QUALITY(version="v2")
```

## Computation-Based Metrics

No LLM judge. Deterministic comparison of `response` vs `reference`.

| Metric name | What it measures | Notes |
|---|---|---|
| `exact_match` | Exact string match | Case-sensitive |
| `bleu` | BLEU score (translation/generation) | Standard BLEU |
| `rouge_1` | ROUGE-1 (unigram overlap) | Summarization |
| `rouge_l_sum` | ROUGE-L (longest common subsequence) | Summary-level |
| `tool_call_valid` | Whether tool calls are syntactically valid | Agent evals |
| `tool_name_match` | Whether tool names match reference | Agent evals |
| `tool_parameter_key_match` | Whether tool parameter keys match | Agent evals |
| `tool_parameter_kv_match` | Whether tool parameter key-value pairs match | Agent evals |

```python
# Usage
metric = types.Metric(name="exact_match")
metric = types.Metric(name="tool_call_valid")
```

## Translation Metrics

| Metric name | Default version | Notes |
|---|---|---|
| `comet` | `COMET_22_SRC_REF` | Requires `prompt` (source), `response`, `reference` |
| `metricx` | `METRICX_24_SRC_REF` | Requires `prompt` (source), `response`, `reference` |

## RubricMetric / PrebuiltMetric (GCS-Loaded LLM Recipes)

These resolve first against the API predefined list, then fall back to
GCS-hosted LLM metric YAML definitions.

| Property | Resolution |
|---|---|
| `GENERAL_QUALITY` | API predefined |
| `TEXT_QUALITY` | API predefined |
| `INSTRUCTION_FOLLOWING` | API predefined |
| `SAFETY` | API predefined |
| `HALLUCINATION` | API predefined |
| `TOOL_USE_QUALITY` | API predefined |
| `MULTI_TURN_GENERAL_QUALITY` | API predefined |
| `MULTI_TURN_TEXT_QUALITY` | API predefined |
| `MULTI_TURN_TOOL_USE_QUALITY` | API predefined |
| `MULTI_TURN_TRAJECTORY_QUALITY` | API predefined |
| `MULTI_TURN_TASK_SUCCESS` | API predefined |
| `FINAL_RESPONSE_MATCH` | API predefined (v2) |
| `FINAL_RESPONSE_REFERENCE_FREE` | API predefined |
| `FINAL_RESPONSE_QUALITY` | API predefined |
| `COHERENCE` | GCS-loaded LLM recipe |
| `FLUENCY` | GCS-loaded LLM recipe |
| `VERBOSITY` | GCS-loaded LLM recipe |
| `SUMMARIZATION_QUALITY` | GCS-loaded LLM recipe |
| `QUESTION_ANSWERING_QUALITY` | GCS-loaded LLM recipe |
| `MULTI_TURN_CHAT_QUALITY` | GCS-loaded LLM recipe |
| `MULTI_TURN_SAFETY` | GCS-loaded LLM recipe |

Any arbitrary name can be tried via `RubricMetric.<NAME>` — it will
attempt resolution against the API list and then GCS.

## Custom Metrics

### Custom Local Function

Runs client-side. Fastest iteration, no API call.

```python
def my_evaluator(instance: dict) -> float:
    response_text = instance.get("response", "")
    return 1.0 if "thank you" in response_text.lower() else 0.0

metric = types.Metric(
    name="politeness_check",
    custom_function=my_evaluator,
)
```

### CodeExecutionMetric (Remote Sandboxed)

Runs server-side in a secure sandbox. Must contain `def evaluate(instance)`.

```python
metric = types.CodeExecutionMetric(
    name="link_validator",
    custom_function='''
import re
def evaluate(instance: dict) -> dict:
    text = instance.get("response", "")
    links = re.findall(r"https?://\\S+", text)
    valid = all(link.startswith("https://") for link in links)
    return {"score": 1.0 if valid else 0.0, "explanation": f"Found {len(links)} links"}
''',
)
```

### LLMMetric (LLM-as-a-Judge)

Uses a judge model to evaluate with a custom prompt template.

```python
metric = types.LLMMetric(
    name="helpfulness",
    prompt_template="""
Evaluate whether the response is helpful for the given query.

Query: {prompt}
Response: {response}

Score 1 if helpful, 0 if not. Explain your reasoning.
""",
    judge_model="gemini-2.5-flash",
    judge_model_sampling_count=3,
)

# Load from YAML/JSON file
metric = types.LLMMetric.load("path/to/metric_config.yaml")
```

### MetricPromptBuilder (Structured Judge Prompt)

Builds structured LLM judge prompts from criteria, rating scores, and
evaluation steps. Preferred over raw `prompt_template` strings for complex
rubrics.

```python
metric = types.LLMMetric(
    name="structured_quality",
    prompt_template=types.MetricPromptBuilder(
        criteria={
            "Accuracy": "Response contains factually correct information",
            "Completeness": "Response addresses all aspects of the query",
        },
        rating_scores={
            "1": "Poor — fails on both criteria",
            "3": "Acceptable — meets one criterion",
            "5": "Excellent — meets both criteria",
        },
    ),
    judge_model="gemini-2.5-flash",
)
```

### Registered Metric (Server-Side Resource)

For reusable metrics shared across teams.

```python
# Create once
resource = client.evals.create_evaluation_metric(metric_config)

# Use by resource name
metric = types.Metric(
    name="team_quality",
    metric_resource_name="projects/.../evaluationMetrics/...",
)
```

## Metric Selection Guide

| Agent Type | Recommended Metrics |
|---|---|
| **RAG agent** | `hallucination_v1`, `grounding_v1`, `general_quality_v1` |
| **Tool-use agent** | `tool_use_quality_v1`, `multi_turn_task_success_v1`, `tool_call_valid`, `tool_name_match` |
| **Multi-turn conversational** | `multi_turn_general_quality_v1`, `multi_turn_text_quality_v1`, `safety_v1` |
| **Code generation** | `CodeExecutionMetric` (custom), `exact_match`, `instruction_following_v1` |
| **Summarization** | `RubricMetric.SUMMARIZATION_QUALITY`, `rouge_l_sum` |
| **Translation** | `comet`, `metricx` |

## Pairwise Comparison

There is no `PairwiseMetric` class. For model comparison, provide multiple
`EvaluationDataset` instances and use `calculate_win_rates()`:

```python
result_a = client.evals.evaluate(dataset=dataset_a, metrics=[...])
result_b = client.evals.evaluate(dataset=dataset_b, metrics=[...])
win_rates = calculate_win_rates(result_a, result_b)
```

## Handler Dispatch Order

When the SDK receives a metric, it checks in this order:

1. `CodeExecutionMetric` with `custom_function` (str) or `remote_custom_function`
2. `Metric` with `custom_function` (local `Callable`)
3. `Metric` with `metric_resource_name` (registered)
4. Name in computation metrics (`exact_match`, `bleu`, etc.)
5. Name in translation metrics (`comet`, `metricx`)
6. Name in predefined API metrics (`general_quality_v1`, etc.)
7. `LLMMetric` with `prompt_template` (custom LLM judge)
