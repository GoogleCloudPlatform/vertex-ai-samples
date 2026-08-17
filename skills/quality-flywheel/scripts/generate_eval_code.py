#!/usr/bin/env python3
"""Generate a runnable Vertex Evaluation SDK script from a dataset and metrics.

Reads an evaluation dataset JSON and generates a complete Python script
that executes the evaluation and prints results.

Usage:
  python generate_eval_code.py --dataset dataset.json --metrics
  hallucination_v1,safety_v1
  python generate_eval_code.py --dataset dataset.json --metrics hallucination_v1
  --output eval_script.py
  python generate_eval_code.py --dataset dataset.json  # defaults to
  general_quality_v1

The generated script can be run directly:
  python eval_script.py --project my-project --location us-central1
"""

import argparse
import json
import os
import sys
from typing import Any

# Predefined metrics accessed via types.RubricMetric.<NAME>
_RUBRIC_METRICS = frozenset({
    "general_quality_v1",
    "text_quality_v1",
    "instruction_following_v1",
    "grounding_v1",
    "safety_v1",
    "hallucination_v1",
    "tool_use_quality_v1",
    "multi_turn_general_quality_v1",
    "multi_turn_text_quality_v1",
    "multi_turn_tool_use_quality_v1",
    "multi_turn_trajectory_quality_v1",
    "multi_turn_task_success_v1",
    "final_response_match_v2",
    "final_response_reference_free_v1",
    "final_response_quality_v1",
})

# Computation metrics accessed via types.Metric(name="...")
_COMPUTATION_METRICS = frozenset({
    "exact_match",
    "bleu",
    "rouge_1",
    "rouge_l_sum",
    "tool_call_valid",
    "tool_name_match",
    "tool_parameter_key_match",
    "tool_parameter_kv_match",
})

# Map from metric name to RubricMetric constant name
_RUBRIC_CONSTANT_MAP = {
    "general_quality_v1": "GENERAL_QUALITY",
    "text_quality_v1": "TEXT_QUALITY",
    "instruction_following_v1": "INSTRUCTION_FOLLOWING",
    "grounding_v1": "GROUNDING",
    "safety_v1": "SAFETY",
    "hallucination_v1": "HALLUCINATION",
    "tool_use_quality_v1": "TOOL_USE_QUALITY",
    "multi_turn_general_quality_v1": "MULTI_TURN_GENERAL_QUALITY",
    "multi_turn_text_quality_v1": "MULTI_TURN_TEXT_QUALITY",
    "multi_turn_tool_use_quality_v1": "MULTI_TURN_TOOL_USE_QUALITY",
    "multi_turn_trajectory_quality_v1": "MULTI_TURN_TRAJECTORY_QUALITY",
    "multi_turn_task_success_v1": "MULTI_TURN_TASK_SUCCESS",
    "final_response_match_v2": "FINAL_RESPONSE_MATCH",
    "final_response_reference_free_v1": "FINAL_RESPONSE_REFERENCE_FREE",
    "final_response_quality_v1": "FINAL_RESPONSE_QUALITY",
}


def _metric_to_code(metric_name: str) -> str:
  """Convert a metric name to its Python SDK expression."""
  if metric_name in _RUBRIC_CONSTANT_MAP:
    return f"types.RubricMetric.{_RUBRIC_CONSTANT_MAP[metric_name]}"
  if metric_name in _COMPUTATION_METRICS:
    return f'types.Metric(name="{metric_name}")'
  # Unknown metric — try as RubricMetric constant
  return f'types.Metric(name="{metric_name}")'


def _detect_dataset_type(dataset: dict[str, Any]) -> str:
  """Detect whether dataset is single-turn or multi-turn."""
  cases = dataset.get("eval_cases", [])
  if not cases:
    return "unknown"
  first = cases[0]
  if "agent_data" in first or "agentData" in first:
    return "multi_turn"
  if "prompt" in first:
    return "single_turn"
  return "unknown"


def generate_script(
    dataset_path: str,
    metrics: list[str],
    dataset: dict[str, Any],
) -> str:
  """Generate a complete evaluation Python script."""
  dataset_type = _detect_dataset_type(dataset)
  n_cases = len(dataset.get("eval_cases", []))

  metrics_code = ",\n        ".join(_metric_to_code(m) for m in metrics)

  script = f'''#!/usr/bin/env python3
"""Auto-generated Vertex Evaluation script.

Dataset: {dataset_path} ({n_cases} case(s), {dataset_type})
Metrics: {", ".join(metrics)}

Run: python <this_script>.py --project <PROJECT_ID> --location <LOCATION>
"""

import argparse
import json
import sys

import vertexai
from vertexai import types
from google.genai import types as genai_types


def load_dataset(path: str) -> types.EvaluationDataset:
    """Load evaluation dataset from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return types.EvaluationDataset.model_validate(data)


def run_evaluation(project: str, location: str, dataset_path: str):
    """Run evaluation and print results."""
    print(f"Initializing Vertex AI client (project={{project}}, location={{location}})...")
    client = vertexai.Client(project=project, location=location)

    print(f"Loading dataset from {{dataset_path}}...")
    dataset = load_dataset(dataset_path)

    metrics = [
        {metrics_code}
    ]

    print(f"Running evaluation with {{len(metrics)}} metric(s)...")
    result = client.evals.evaluate(dataset=dataset, metrics=metrics)

    # Print summary metrics
    print("\\n" + "=" * 60)
    print("SUMMARY METRICS")
    print("=" * 60)
    if result.summary_metrics:
        for summary in result.summary_metrics:
            print(f"  {{summary.metric_name}}:")
            print(f"    Mean Score: {{summary.mean_score}}")
            print(f"    Pass Rate:  {{summary.pass_rate}}")
            print(f"    Std Dev:    {{summary.stdev_score}}")
            print(f"    Valid/Total: {{summary.num_cases_valid}}/{{summary.num_cases_total}}")
            print()

    # Print per-case results
    print("=" * 60)
    print("PER-CASE RESULTS")
    print("=" * 60)
    if result.eval_case_results:
        for case_result in result.eval_case_results:
            print(f"\\n  Case {{case_result.eval_case_index}}:")
            if case_result.response_candidate_results:
                for candidate in case_result.response_candidate_results:
                    if candidate.metric_results:
                        for metric_name, metric_result in candidate.metric_results.items():
                            print(f"    {{metric_name}}:")
                            print(f"      Score: {{metric_result.score}}")
                            if metric_result.explanation:
                                explanation = metric_result.explanation[:200]
                                print(f"      Explanation: {{explanation}}...")
                            if metric_result.rubric_verdicts:
                                for v in metric_result.rubric_verdicts:
                                    status = "PASS" if v.verdict else "FAIL"
                                    rubric_id = v.evaluated_rubric.rubric_id if v.evaluated_rubric else "?"
                                    print(f"      Rubric {{rubric_id}}: {{status}}")
                                    if v.reasoning:
                                        print(f"        Reasoning: {{v.reasoning[:150]}}...")

    # Output as JSON for programmatic consumption
    print("\\n" + "=" * 60)
    print("JSON OUTPUT")
    print("=" * 60)
    json_output = {{
        "summary": [
            {{
                "metric": s.metric_name,
                "mean_score": s.mean_score,
                "pass_rate": s.pass_rate,
                "stdev": s.stdev_score,
            }}
            for s in (result.summary_metrics or [])
        ],
    }}
    print(json.dumps(json_output, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(description="Run Vertex Evaluation")
    parser.add_argument("--project", "-p", required=True, help="GCP Project ID")
    parser.add_argument("--location", "-l", default="us-central1", help="GCP Location")
    parser.add_argument("--dataset", "-d", default="{dataset_path}", help="Dataset JSON path")
    args = parser.parse_args()
    run_evaluation(args.project, args.location, args.dataset)


if __name__ == "__main__":
    main()
'''
  return script


def main():
  parser = argparse.ArgumentParser(
      description="Generate a runnable Vertex Eval SDK script."
  )
  parser.add_argument(
      "--dataset",
      "-d",
      required=True,
      help="Path to the evaluation dataset JSON file.",
  )
  parser.add_argument(
      "--metrics",
      "-m",
      default="general_quality_v1",
      help="Comma-separated metric names (default: general_quality_v1).",
  )
  parser.add_argument(
      "--output",
      "-o",
      help="Output script path. Prints to stdout if not specified.",
  )
  args = parser.parse_args()

  if not os.path.exists(args.dataset):
    print(f"ERROR: File not found: {args.dataset}", file=sys.stderr)
    sys.exit(1)

  try:
    with open(args.dataset) as f:
      dataset = json.load(f)
  except json.JSONDecodeError as e:
    print(f"ERROR: Invalid JSON in {args.dataset}: {e}", file=sys.stderr)
    sys.exit(1)

  metrics = [m.strip() for m in args.metrics.split(",")]

  # Validate metric names
  all_known = _RUBRIC_METRICS | _COMPUTATION_METRICS
  unknown = [m for m in metrics if m not in all_known]
  if unknown:
    print(
        f"WARNING: Unknown metric(s): {', '.join(unknown)}. "
        "They will be passed as types.Metric(name=...).",
        file=sys.stderr,
    )

  script = generate_script(args.dataset, metrics, dataset)

  if args.output:
    with open(args.output, "w") as f:
      f.write(script)
    os.chmod(args.output, 0o755)
    print(f"Generated eval script: {args.output}", file=sys.stderr)
    print(
        f"Run: python {args.output} --project <PROJECT> --location <LOCATION>",
        file=sys.stderr,
    )
  else:
    print(script)


if __name__ == "__main__":
  main()
