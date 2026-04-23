# Evaluation Failure Patterns & Fixes

Common failure modes observed in GenAI agent evaluations, mapped to their
root causes and concrete fixes.

## Metric-Specific Failures

### Low `hallucination_v1` or `grounding_v1` Score

**Symptom:** Agent generates plausible-sounding but factually incorrect
information, or doesn't use the provided context.

**Root causes:**
- System prompt lacks explicit grounding instructions
- Retrieved context not passed into the prompt
- Agent ignores context in favor of parametric knowledge

**Fixes:**
1. Add to system prompt: "Base ALL answers strictly on the provided context.
   If the context doesn't contain the answer, say 'I don't have that
   information.'"
2. Verify context is actually injected into the prompt (check tool responses)
3. Add `temperature=0` or lower temperature to reduce creative generation

### Low `general_quality_v1` or `text_quality_v1`

**Symptom:** Agent responses are poorly structured, unclear, or unhelpful.

**Root causes:**
- System prompt too vague
- Agent over-explains or under-explains
- Missing output format instructions

**Fixes:**
1. Add explicit format instructions: "Respond concisely in 2-3 sentences."
2. Add few-shot examples in the system prompt
3. Review rubric verdicts for specific quality dimensions that scored low

### Low `tool_use_quality_v1` or `tool_call_valid`

**Symptom:** Agent calls the wrong tool, uses wrong parameters, or doesn't
call tools when it should.

**Root causes:**
- Tool descriptions are ambiguous
- Multiple tools have overlapping functionality
- Function declaration parameter schemas are incomplete

**Fixes:**
1. Make tool `description` fields precise and mutually exclusive
2. Add parameter descriptions and constraints to `FunctionDeclaration`
3. Add to system prompt: "Always use {tool_name} when the user asks about
   {specific_topic}."
4. Check `tool_name_match` and `tool_parameter_kv_match` for granular diagnosis

### Low `multi_turn_trajectory_quality_v1`

**Symptom:** Agent takes suboptimal paths through a conversation — unnecessary
tool calls, redundant questions, or wrong delegation order.

**Root causes:**
- Router agent lacks clear delegation rules
- Agent retries failed operations without adaptation
- Missing escalation logic

**Fixes:**
1. Add explicit routing rules: "Route to {agent} when {condition}."
2. Add retry limits: "If {tool} fails twice, inform the user and suggest
   alternatives."
3. Review the trajectory events in `agent_data` to identify the specific
   turn where the agent deviated

### Low `multi_turn_task_success_v1`

**Symptom:** Agent engages in conversation but doesn't complete the user's
actual goal.

**Root causes:**
- Agent gets sidetracked by follow-up questions
- Missing confirmation/completion step
- Agent doesn't track task state across turns

**Fixes:**
1. Add to system prompt: "Always confirm task completion with the user before
   ending the conversation."
2. Implement explicit task tracking in agent logic
3. Verify `max_turn` in user simulator is sufficient for the task complexity

### Low `safety_v1`

**Symptom:** Agent generates unsafe content or complies with harmful requests.

**Root causes:**
- System prompt lacks safety constraints
- Agent follows user instructions too literally
- Missing refusal logic for out-of-scope requests

**Fixes:**
1. Add safety guardrails: "Never provide medical/legal/financial advice.
   Redirect to appropriate professionals."
2. Add refusal patterns: "If the user asks for {harmful_category}, politely
   decline."
3. Use `safety_v1` alongside domain-specific `LLMMetric` safety checks

## Structural Failures

### `is_infra_error: true`

**Symptom:** Eval case fails with infrastructure error, not a quality issue.

**Root causes:**
- API quota exceeded
- Network timeout
- Model endpoint temporarily unavailable

**Fix:** Re-run the evaluation. If persistent, check quota and endpoint health.

### Timeout

**Symptom:** Evaluation times out before completing.

**Root causes:**
- Dataset too large for a single API call
- Complex custom metric code takes too long
- Judge model sampling count too high

**Fixes:**
1. Reduce dataset size or batch into smaller chunks
2. Optimize custom metric code (avoid network calls in `evaluate()`)
3. Reduce `judge_model_sampling_count` (default 1, max 32)

### `KeyError` in Custom Metric

**Symptom:** Custom function crashes with missing field.

**Root cause:** Metric function expects a field not present in the eval case.

**Fix:** Check available fields in the `instance` dict. Common fields:
`prompt`, `response`, `reference`, `agent_data`. Always use `.get()` with
defaults.

## Analysis Workflow

When eval results show failures:

1. **Start with `summary_metrics`** — identify which metrics scored lowest
2. **Drill into `eval_case_results`** — find specific failing cases
3. **Read `rubric_verdicts`** — understand why the judge scored low
4. **Cross-reference with `agent_data`** — find the exact turn/event
   that caused the failure
5. **Identify the pattern** — is it a prompt issue, tool issue, or data issue?
6. **Apply the targeted fix** — from the table above
7. **Re-run and compare** — verify the fix improved the target metric
