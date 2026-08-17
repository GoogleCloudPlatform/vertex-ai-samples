# Quality Flywheel Skill - Manual Test Plan

## Prerequisites

- Read `SKILL.md` to understand the workflow
- Have a GCP project with the GenAI Evaluation API enabled
- Access to an environment with the skill installed

---

## Test 1: Cold Start — No Data

**Prompt:** "I want to evaluate my customer support agent but I don't
have any test data. Can you help me set up evals from scratch?"

**Verify:**
- Agent asks for GCP Project ID and Location (or attempts env vars)
- Agent recommends synthetic data generation via `generate_user_scenarios`
- Agent explains the EvalCase / EvaluationDataset format
- Agent suggests appropriate metrics for customer support

---

## Test 2: Metric Selection

**Prompt:** "My agent is a multi-turn travel booking assistant that
uses tools like search_flights and book_hotel. Which metrics should
I use to evaluate it?"

**Verify:**
- Agent recommends multi-turn agent metrics (trajectory quality, task success)
- Agent recommends tool-specific computation metrics (tool_call_valid, tool_name_match)
- Agent uses `types.RubricMetric.<NAME>` syntax
- Agent does NOT recommend only single-turn text quality metrics

---

## Test 3: Custom Metric

**Prompt:** "I need a metric that checks if my agent always includes
a disclaimer when giving financial advice. Can you write one?"

**Verify:**
- Agent creates a `CodeExecutionMetric` or `LLMMetric`
- The metric logic specifically checks for disclaimer presence
- Code is syntactically valid Python
- Agent shows how to use it with `client.evals.evaluate()`

---

## Test 4: End-to-End Execution

**Prompt:** "Here's my agent — it's a simple Q&A bot. I have 3 test
questions. Can you write an eval script and run it?"

Provide sample questions when asked.

**Verify:**
- Agent writes a complete Python script with proper imports
- Script uses `vertexai.Client()` initialization
- Agent saves the script to a file and executes it
- Agent reads and analyzes the actual output (no hallucinated results)
- Agent uses standard Python imports (not internal paths)

---

## Test 5: Result Analysis

**Prompt:** "My eval results show 0.3 on tool_use_quality and 0.8
on general_quality. The agent keeps calling the wrong tool when
users ask about order status. What should I fix?"

**Verify:**
- Agent identifies tool_use_quality as the critical failure
- Agent suggests specific tool description or system prompt improvements
- Agent recommends checking `tool_name_match` for granular diagnosis
- Agent references failure patterns (from references/failure_patterns.md)

---

## Cleanup

No persistent state created. Scripts written to /tmp/ can be deleted.
