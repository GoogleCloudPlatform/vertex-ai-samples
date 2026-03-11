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

---
name: vertex-tuning
description: >
  Vertex AI Model Tuning Router. Use this skill when the user wants to fine-tune
  models using Vertex AI. This skill routes to either vertex-tuning-open-model
  or vertex-tuning-gemini.
---

# Vertex AI Model Tuning (Router)

## Overview

This skill acts as a router for Vertex AI tuning tasks. The tuning procedures
for Open Models and Gemini Models differ significantly. Your first step is to
determine which category the user intends to tune and then read the
corresponding sub-skill.

## Workflow Decision Tree

1.  **Model Category**: Has the user explicitly stated whether they want to tune
    an **Open Model** or a **Gemini Model**?

    -   **No** → **STOP**. Ask the user if they want to tune an Open Model or a
        Gemini Model. Do not proceed or recommend any specific models until this
        is confirmed.
    -   **Yes** (Open Model) → The user wants to tune an Open Model. Stop
        reading this file and IMMEDIATELY read the skill instructions located at
        `open-model/SKILL.md`. Follow the instructions inside that skill to
        complete the task.
    -   **Yes** (Gemini Model) → The user wants to tune a Gemini Model. Stop
        reading this file and IMMEDIATELY read the skill instructions located at
        `gemini/SKILL.md`. Follow the instructions inside that skill to complete
        the task.
