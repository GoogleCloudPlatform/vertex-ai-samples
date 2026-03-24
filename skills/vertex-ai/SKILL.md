---
name: vertex-ai
description: >
  Primary Router for Vertex AI skills. Use this skill when the user wants to work
  with Google Cloud Vertex AI (e.g., deploying models, running inference, or tuning models).
  This skill routes to vertex-deploy, vertex-inference, or vertex-tuning.
---

# Vertex AI Skills (Primary Router)

## Overview

This skill acts as the primary router for all Vertex AI related tasks. Google Cloud Vertex AI provides capabilities for deploying open models, running inference against Generative AI models (like Gemini and open models), and fine-tuning models.

Your first step is to determine the user's main objective and then read the appropriate sub-skill.

## Workflow Decision Tree

1.  **Objective Category**: What does the user want to do with Vertex AI?

    -   **Use Gemini SDK (Gen AI SDK)**: If the user explicitly asks about using the Gemini API on Vertex AI with the Gen AI SDK, or asks for features like Live API, multimedia generation, caching, or batch prediction, stop reading this file and IMMEDIATELY read the skill instructions located at `../genai-sdk/SKILL.md`.
    -   **Deploy Models**: If the user wants to deploy an Open Model from Model Garden or deploy custom weights to create a Vertex AI Endpoint, stop reading this file and IMMEDIATELY read the skill instructions located at `../vertex-deploy/SKILL.md`.
    -   **Run Inference (Text Generation, Chat, Embeddings)**: If the user wants to connect to Vertex AI to generate text, chat, or get embeddings using models like Gemini, Llama, DeepSeek, etc., stop reading this file and IMMEDIATELY read the skill instructions located at `../vertex-inference/SKILL.md`.
    -   **Model Tuning / Fine-Tuning**: If the user wants to fine-tune an Open Model or a Gemini Model, stop reading this file and IMMEDIATELY read the skill instructions located at `../vertex-tuning/SKILL.md`.

    -   **Unknown / Not Listed**: If the objective is not clearly one of the above, **STOP**. Ask the user to clarify whether they are looking to deploy a model, run inference, or tune a model on Vertex AI.
