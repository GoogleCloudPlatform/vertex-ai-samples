<!--
 Copyright 2026 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

# Vertex AI Supported Models and Recommendations

This reference catalog provides technical specifications, tuning
recommendations, and deployment hardware requirements for supported models in
Vertex AI.

## Supported Models Catalog

> [!WARNING] **CRITICAL AGENT INSTRUCTION**
> Do NOT use this catalog to recommend a specific model to the user until they
> have explicitly confirmed their **Model Category** as Open Model.

Available open models can be found in Google Cloud [documentation](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/open-model-tuning#supported-models).
This is the list of open models that are available for tuning; do not suggest
any other open models besides the one listed here.
Each model has some [limitations](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/open-model-tuning#limitations) for tuning.

## Model Selection Guidelines

**Identify Task**: Check a few samples from the dataset to identify the task.

Choose a model family based on your task type:

- **Qwen**: Best for code generation or complex math-based tasks.
- **Gemma**: Optimized for chat-based interactions, creative writing and multilingual tasks.
- **Llama (Instruct)**: Strong general-purpose chat/instruction models.
- **Llama (Base/Scout)**: Best for continuation tasks or building custom instruction-tuned models.

**Complexity Heuristics**:

- **Simple (QA, Extraction)**: 1B - 3B models.
- **Intermediate (Summarization, Reasoning)**: 8B - 17B models.
- **Complex (Multi-turn, Tool use, Deep reasoning)**: 27B - 70B models.

## Baseline Hyperparameter Recommendations

These values are starting points and should be adjusted based on your dataset
size.

| Model | Tuning Mode | Learning Rate | Epochs | Adapter Size (PEFT) |
| :--- | :--- | :--- | :--- | :--- |
| Gemma 3 1B IT | Full | 2.0E-5 | 3 | N/A |
| Gemma 3 4B IT | Full | 1.0E-5 | 3 | N/A |
| Gemma 3 12B IT | Full | 1.0E-5 | 3 | N/A |
| Gemma 3 27B IT | PEFT | 2.0E-4 | 3 | 32 |
| Gemma 3 27B IT | Full | 2.0E-4 | 3 | N/A |
| Llama 3.1 8B | PEFT | 2.0E-4 | 3 | 16 |
| Llama 3.1 8B | Full | 2.0E-4 | 3 | N/A |
| Llama 3.1 8B Instruct | PEFT | 2.0E-4 | 3 | 16 |
| Llama 3.1 8B Instruct | Full | 2.0E-4 | 3 | N/A |
| Llama 3.2 1B Instruct | Full | 1.5E-6 | 3 | N/A |
| Llama 3.2 3B Instruct | Full | 1.0E-7 | 3 | N/A |
| Llama 3.3 70B Instruct | PEFT | 5.0E-5 | 3 | 16 |
| Llama 3.3 70B Instruct | Full | 5.0E-5 | 3 | N/A |
| Llama 4 Scout 17B 16E | PEFT | 2.0E-5 | 3 | 16 |
| Qwen 3 4B | Full | 7.5e-5 | 3 | N/A |
| Qwen 3 8B | Full | 5e-5 | 3 | N/A |
| Qwen 3 14B | Full | 4e-5 | 3 | N/A |
| Qwen 3 32B | PEFT | 2.0E-4 | 3 | 16 |
| Qwen 3 32B | Full | 2.5e-5 | 3 | N/A |