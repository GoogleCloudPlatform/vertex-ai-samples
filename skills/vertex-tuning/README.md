# Vertex AI Model Tuning Skill

This skill allows agents to fine-tune Large Language Models (LLMs) using
Vertex AI's managed tuning service. It encapsulates the workflow for data
preparation, job submission, monitoring, and model deployment.

## Setup

### 1. Clone the repo

Add instruction here with the correct repo handle

> git clone repo-link

### 2. Installing Agent Skills

Install the skill to your favorite AI assisted coding tool.

- [Antigravity](https://antigravity.google/docs/skills#:~:text=and%20follows%20them.-,Where%20skills%20live,-Antigravity%20supports%20two)
- [Gemini CLI](https://geminicli.com/docs/cli/skills/#managing-skills)
- [Claude Code](https://code.claude.com/docs/en/skills#where-skills-live)

#### Verification

To confirm the installation was successful ask:

> "What skills do you have?"

The agent should respond with a list including your newly added skill
`vertex-tuning` alongside its default capabilities.

## Getting started

Try the following prompt to get started:

> "I want to fine-tune a Llama 3.1 8B model for text classification on Vertex.
Can you help me with this?"

## Features

*   **Data Preparation**: Converts and validates datasets (JSONL format) for Vertex AI.
*   **Model Tuning**: Submits tuning jobs for supported models (Llama, Gemma,
    Qwen, etc.) with customizable hyperparameters (PEFT/Full).
*   **Model Deployment**: Deploys tuned models to Vertex AI Endpoints for serving.
*   **Guidance**: Provides recommendations for models and hyperparameters based on the task and dataset.

## Directory Structure

*   `scripts/`: Python scripts for each stage of the workflow.
    *   `prepare_dataset.py`: Converts, splits, and validates datasets.
    *   `tune_model.py`: Submits the tuning job to Vertex AI.
    *   `deploy_model.py`: Deploys the tuned model to an endpoint.
*   `references/`: Documentation and catalogs.
    *   `models.md`: Supported models, hardware requirements, and hyperparameter baselines.
    *   `data_prep.md`: Data formatting guidelines.
    *   `tuning_guide.md`: Detailed tuning advice.
