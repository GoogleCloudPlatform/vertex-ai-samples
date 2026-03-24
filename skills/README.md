# Vertex AI Skills Developer Guide

This directory contains a suite of AI Agent "Skills" configured to help agents understand, navigate, and execute tasks on Google Cloud Vertex AI.

## Architecture & Routing

These skills are designed hierarchically to guide the LLM agent from a broad user intent down to the specific technical implementation steps.

### Primary Router: `vertex-ai`
The entry point for any general Vertex AI task. The `vertex-ai/SKILL.md` file acts as a traffic controller. When an agent receives a generic request (e.g., "I want to use Vertex AI"), it reads this file to determine the next step based on a Decision Tree:
- **Deploying a model** → Routes to `vertex-deploy`
- **Using Gemini API & Gen AI SDK** → Routes to `genai-sdk`
- **Generating text, chat, or embeddings** → Routes to `vertex-inference`
- **Fine-tuning a model** → Routes to `vertex-tuning`

### Sub-Skills
1. **`genai-sdk`**
   - **Purpose:** Guides the usage of Gemini API on Google Cloud Vertex AI with the Gen AI SDK across multiple languages (Python, JS/TS, Go, Java, C#).
   - **Capabilities:** Core inference, Live API, function calling, structured output, caching, and batch prediction.

2. **`vertex-deploy`**
   - **Purpose:** Instructions and bash scripts for deploying Open Models from Model Garden or custom weights to a dedicated Vertex AI Endpoint.
   - **Capabilities:** Handling `gcloud ai model-garden models deploy`, checking operation status, calculating quota/cost estimates, and undeploying models to save costs.

2. **`vertex-inference`**
   - **Purpose:** Code samples and instructions for authenticating and executing Generative AI inference.
   - **Capabilities:** Supports both First-Party (Gemini) using the `google-genai` SDK and Third-Party OpenMaaS (Llama, DeepSeek, Qwen) using the standard OpenAI SDK configured with a Vertex AI endpoint.

3. **`vertex-tuning`**
   - **Purpose:** A secondary router specifically for model fine-tuning.
   - **Capabilities:** Directs the agent to specific tuning procedures depending on whether the user wants to tune an Open Model (e.g., Llama) or a First-Party Gemini model.

## How It Works Under The Hood

Agent skills are essentially specialized prompt contexts. When building agents (like those using Semantic Router, LangChain, or other LLM orchestration frameworks), the agent is instructed to read the relevant `SKILL.md` files.

1. **Trigger:** The agent recognizes a keyword (like "Vertex AI") and reads `skills/vertex-ai/SKILL.md`.
2. **Determine Intent:** The LLM processes the user prompt against the conditions in the Router's Decision Tree.
3. **Navigate:** The LLM actively uses file-reading tools to navigate to the correct underlying `SKILL.md` (e.g., `skills/vertex-inference/SKILL.md`).
4. **Execute:** The agent reads the highly-detailed instructions, caveats, and code snippets in the sub-skill to fulfill the user's request.

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
