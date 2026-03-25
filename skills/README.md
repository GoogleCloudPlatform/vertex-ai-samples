# 🤖 Agentic Skills for Google Cloud Vertex AI

This directory contains a suite of AI Agent **Skills** configured to help agents understand, navigate, and execute tasks on Google Cloud Vertex AI.

> [!NOTE]
> These skills act as specialized prompt contexts, enabling your LLM agents to perform complex Cloud AI workflows autonomously.

## Table of Contents

- [Top Starter Skills](#top-starter-skills)
- [Tool Configuration](#tool-configuration)
- [Architecture & Routing](#architecture--routing)
  - [Primary Router: `vertex-ai`](#primary-router-vertex-ai)
  - [Sub-Skills](#sub-skills)
- [How It Works Under The Hood](#how-it-works-under-the-hood)
- [Installation & Usage](#installation--usage)
  - [Example Prompts](#example-prompts)
- [How to Add a New Skill](#how-to-add-a-new-skill)
- [Best Practices for Authoring Skills](#best-practices-for-authoring-skills)

## 🌟 Top Starter Skills

- 🚀 `@vertex-deploy` for deploying Open Models from Model Garden directly to endpoints.
- 🛠️ `@genai-sdk` for learning how to use the latest Google Gen AI SDK.
- 🎯 `@vertex-tuning` for fine-tuning models with your own data.
- ⚡ `@vertex-inference` for executing Generative AI inference.

## ⚙️ Tool Configuration

These skills follow the universal **SKILL.md** format and work with any AI coding assistant that supports agentic skills.

| Tool                  | Type | Invocation Example                  | Path                                                                      |
| :-------------------- | :--- | :---------------------------------- | :------------------------------------------------------------------------ |
| **Gemini CLI**  | CLI  | `(User Prompt) Use skill-name...` | `.gemini/skills/`                                                       |
| **Antigravity** | IDE  | `(Agent Mode) Use skill...`       | Global:`~/.gemini/antigravity/skills/` · Workspace: `.agent/skills/` |
| **Claude Code** | CLI  | `>> /skill-name help me...`       | `.claude/skills/`                                                       |
| **Cursor**      | IDE  | `@skill-name (in Chat)`           | `.cursor/skills/`                                                       |


## Architecture & Routing

These skills are designed hierarchically to guide the LLM agent from a broad user intent down to the specific technical implementation steps.

### Primary Router: `vertex-ai`
The entry point for any general Vertex AI task. The `vertex-ai/SKILL.md` file acts as a traffic controller. When an agent receives a generic request (e.g., "I want to use Vertex AI"), it reads this file to determine the next step based on a Decision Tree:
- **Deploying a model** → Routes to `vertex-deploy`
- **Using Gemini API & Gen AI SDK** → Routes to `genai-sdk`
- **Generating text, chat, or embeddings** → Routes to `vertex-inference`
- **Fine-tuning a model** → Routes to `vertex-tuning`

### 🧰 Sub-Skills Directory

| Category / Skill | Purpose | Capabilities |
| :--- | :--- | :--- |
| **`genai-sdk`** | Guides the usage of Gemini API with the Gen AI SDK across multiple languages. | Core inference, Live API, function calling, structured output, caching, and batch prediction. |
| **`vertex-deploy`** | Instructions and bash scripts for deploying Open Models or custom weights to a dedicated Endpoint. | `gcloud ai model-garden models deploy`, checking status, cost estimates, undeploying. |
| **`vertex-inference`** | Code samples and instructions for authenticating and executing Generative AI inference. | First-Party (Gemini) via `google-genai` and Third-Party OpenMaaS via OpenAI SDK. |
| **`vertex-tuning`** | A secondary router specifically for model fine-tuning. | Directs the agent to specific tuning procedures for Open Models or First-Party Gemini models. |

## How It Works Under The Hood

Agent skills are essentially specialized prompt contexts. When building agents (like those using Semantic Router, LangChain, or other LLM orchestration frameworks), the agent is instructed to read the relevant `SKILL.md` files.

1. **Trigger:** The agent recognizes a keyword (like "Vertex AI") and reads `skills/vertex-ai/SKILL.md`.
2. **Determine Intent:** The LLM processes the user prompt against the conditions in the Router's Decision Tree.
3. **Navigate:** The LLM actively uses file-reading tools to navigate to the correct underlying `SKILL.md` (e.g., `skills/vertex-inference/SKILL.md`).
4. **Execute:** The agent reads the highly-detailed instructions, caveats, and code snippets in the sub-skill to fulfill the user's request.

## Installation & Usage

To use these skills with your AI agent:

1. **Clone the Repository**: Clone this repository to your local development environment or production workspace.
2. **Expose to Agent**: Ensure the `skills/` directory is accessible to your AI assistant. This typically involves mounting or copying the folder into your agent's workspace (e.g., `.agents/skills/` or equivalent custom routing directory).
3. **Trigger the Skills**: Prompt your agent with a Vertex AI related task. The agent should automatically scan for skills, find the primary `vertex-ai/SKILL.md` router, and follow the documented workflow.

### Example Prompts

Here are some example prompts you can use to trigger the routing logic and test the skills:

**Testing Deployment:**
> "I want to deploy a Llama 3.3 model from Model Garden to a Vertex AI endpoint. Can you help me write the script?"

**Testing Inference (Gen AI SDK):**
> "Can you show me how to connect to Vertex AI and get text embeddings using the new Gemini SDK?"

**Testing Tuning:**
> "I need to fine-tune a Gemini 1.5 Pro model using Vertex AI. Where should I start?"

## How to Add a New Skill

If Vertex AI introduces a new capability (e.g., Vertex AI Search or Vertex AI Pipelines) and you want to teach the agent to handle it:

1. **Create the Skill Directory:**
   ```bash
   mkdir vertex-search
   touch vertex-search/SKILL.md
   ```
2. **Write the Instructions (`SKILL.md`):**
   - Add frontmatter (`name` and `description`).
   - Write clear, step-by-step instructions.
   - Include code snippets and strict rules using markdown alerts (e.g., `> [!WARNING]`).
3. **Update the Router:**
   - Open `vertex-ai/SKILL.md`.
   - Add a new branch to the "Workflow Decision Tree" that tells the LLM when to route to `../vertex-search/SKILL.md`.

## Best Practices for Authoring Skills

- **Be Directive:** Use imperative verbs ("Do this", "You MUST check that"). AI agents perform best when instructions are explicitly strict.
- **Use Code Samples:** Scripts prevent the agent from hallucinating SDK syntax. Always include fully functional snippets.
- **Handle Errors:** Include a Troubleshooting section in your skills. If a quota error is common, tell the LLM exactly how to interpret a 429 error and explain it to the user.
