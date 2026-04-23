# Evaluation Dataset Schema

Canonical formats for evaluation datasets in the Google GenAI Evaluation SDK.
Source of truth: `vertexai/_genai/types/evals.py` and
`vertexai/_genai/types/common.py`.

## Core Types

```
EvaluationDataset
├── eval_cases: list[EvalCase]       # Primary: list of cases
└── eval_dataset_df: pd.DataFrame    # Alternative: pandas DataFrame

EvalCase
├── prompt: str                      # Single-turn: the user query
├── response: str                    # Single-turn: the model response
├── reference: str                   # Ground truth (for reference-based metrics)
├── agent_data: AgentData            # Multi-turn: full conversation trajectory
└── (extra fields allowed)           # Custom fields for custom metrics

AgentData
├── agents: dict[str, AgentConfig]   # Agent definitions
└── turns: list[ConversationTurn]    # Ordered conversation turns

ConversationTurn
├── turn_index: int                  # 0-based turn number
└── events: list[AgentEvent]         # Events within this turn

AgentEvent
├── author: str                      # "user", agent_id, or "tool"
└── content: genai_types.Content     # Content with role and parts
```

## Single-Turn Dataset

For simple prompt-response evaluation (e.g., QA, summarization).

```python
from vertexai import types

dataset = types.EvaluationDataset(eval_cases=[
    types.EvalCase(
        prompt="What is the capital of France?",
        response="The capital of France is Paris.",
        reference="Paris",
    ),
    types.EvalCase(
        prompt="Summarize this article: ...",
        response="The article discusses...",
    ),
])
```

### From pandas DataFrame

```python
import pandas as pd
from vertexai import types

df = pd.DataFrame({
    "prompt": ["What is 2+2?", "Name the planets"],
    "response": ["4", "Mercury, Venus, Earth, ..."],
    "reference": ["4", "Mercury, Venus, Earth, Mars, ..."],
})
dataset = types.EvaluationDataset(eval_dataset_df=df)
```

### Required fields by metric type

| Metric category | Required fields |
|---|---|
| Predefined (single-turn) | `prompt`, `response` |
| Computation-based | `response`, `reference` |
| Translation | `prompt` (source), `response`, `reference` |
| Custom LLM/code | Fields referenced in your template/function |

## Multi-Turn Dataset (AgentData)

For evaluating multi-turn agent conversations with tool calls.

```python
from vertexai import types
from google.genai import types as genai_types

agent_data = types.evals.AgentData(
    agents={
        "support_agent": types.evals.AgentConfig(
            agent_id="support_agent",
            instruction="You are a helpful support agent.",
            tools=[genai_types.Tool(function_declarations=[
                genai_types.FunctionDeclaration(
                    name="lookup_order",
                    description="Look up order status by ID",
                    parameters=genai_types.Schema(
                        type="OBJECT",
                        properties={"order_id": genai_types.Schema(type="STRING")},
                    ),
                )
            ])],
        )
    },
    turns=[
        types.evals.ConversationTurn(
            turn_index=0,
            events=[
                # User message
                types.evals.AgentEvent(
                    author="user",
                    content=genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text="Where is my order #12345?")]
                    ),
                ),
                # Agent calls tool
                types.evals.AgentEvent(
                    author="support_agent",
                    content=genai_types.Content(
                        role="model",
                        parts=[genai_types.Part(
                            function_call=genai_types.FunctionCall(
                                name="lookup_order",
                                args={"order_id": "12345"},
                            )
                        )]
                    ),
                ),
                # Tool response
                types.evals.AgentEvent(
                    author="support_agent",
                    content=genai_types.Content(
                        role="tool",
                        parts=[genai_types.Part(
                            function_response=genai_types.FunctionResponse(
                                name="lookup_order",
                                response={"status": "shipped", "eta": "tomorrow"},
                            )
                        )]
                    ),
                ),
                # Agent final response
                types.evals.AgentEvent(
                    author="support_agent",
                    content=genai_types.Content(
                        role="model",
                        parts=[genai_types.Part(
                            text="Your order #12345 has been shipped and should arrive tomorrow!"
                        )]
                    ),
                ),
            ],
        ),
    ],
)

eval_case = types.EvalCase(agent_data=agent_data)
dataset = types.EvaluationDataset(eval_cases=[eval_case])
```

## Multi-Agent Dataset

For evaluating systems with multiple collaborating agents.

```python
agent_data = types.evals.AgentData(
    agents={
        "router": types.evals.AgentConfig(
            agent_id="router",
            agent_type="RouterAgent",
            instruction="Route requests to the appropriate specialist.",
        ),
        "flight_bot": types.evals.AgentConfig(
            agent_id="flight_bot",
            agent_type="SpecialistAgent",
            instruction="Search and book flights.",
            tools=[genai_types.Tool(function_declarations=[
                genai_types.FunctionDeclaration(name="search_flights")
            ])],
        ),
    },
    turns=[
        types.evals.ConversationTurn(
            turn_index=0,
            events=[
                types.evals.AgentEvent(
                    author="user",
                    content=genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text="Book a flight to NYC")]
                    ),
                ),
                # Router delegates
                types.evals.AgentEvent(
                    author="router",
                    content=genai_types.Content(
                        role="model",
                        parts=[genai_types.Part(
                            function_call=genai_types.FunctionCall(
                                name="delegate_to_agent",
                                args={"agent_name": "flight_bot"},
                            )
                        )]
                    ),
                ),
            ],
        ),
        types.evals.ConversationTurn(
            turn_index=1,
            events=[
                # Specialist works
                types.evals.AgentEvent(
                    author="flight_bot",
                    content=genai_types.Content(
                        role="model",
                        parts=[genai_types.Part(
                            function_call=genai_types.FunctionCall(
                                name="search_flights",
                                args={"destination": "NYC"},
                            )
                        )]
                    ),
                ),
            ],
        ),
    ],
)
```

## Synthetic Data Generation

### Generate User Scenarios (Cold Start)

```python
scenarios = client.evals.generate_user_scenarios(
    agents={
        "my_agent": types.evals.AgentConfig(
            agent_id="my_agent",
            instruction="You are a helpful customer support agent.",
        )
    },
    root_agent_id="my_agent",
    user_scenario_generation_config=types.evals.UserScenarioGenerationConfig(
        user_scenario_count=10,
        simulation_instruction="Simulate a customer asking about order status.",
        environment_data="Orders can be: pending, shipped, delivered, cancelled.",
        model_name="gemini-2.5-flash",
    ),
)
```

### Run Inference (Populate Responses)

```python
dataset_with_responses = client.evals.run_inference(
    agent=my_agent_callable,
    src=scenarios,
    config={
        "user_simulator_config": {
            "model_name": "gemini-2.5-flash",
            "max_turn": 5,
        }
    },
)
```

## Common Mistakes

| Mistake | Fix |
|---|---|
| Using `role="assistant"` | Use `role="model"` (Vertex convention) |
| Missing `turn_index` | Always set sequential 0-based indices |
| Tool response without `function_response` | Wrap in `genai_types.FunctionResponse` |
| Using `response` field for multi-turn | Use `agent_data` with full trajectory |
| Mixing `prompt` and `agent_data` | Use one or the other per EvalCase |
