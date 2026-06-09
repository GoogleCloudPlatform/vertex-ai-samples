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
name: vertex-tuning-gemini
description: >
  Vertex AI Gemini Model Tuning. Use when you need to fine-tune Gemini models
  using Vertex AI's infrastructure.
---

# Vertex AI Gemini Model Tuning

## Overview

This skill provides procedural knowledge for fine-tuning Gemini Large Language
Models using Vertex AI's tuning service. It covers the entire lifecycle from
environment setup and data preparation to job configuration, monitoring, and
deployment.

## Workflow Decision Tree

1.  **Environment Check**: Has the environment (Auth, APIs, IAM, Venv) been
    initialized?

    -   **No** → Go to [Phase 0: Environment & IAM Setup](#phase-0).
    -   **Yes** → Proceed.

2.  **Dataset Status**: Is the dataset ready in JSONL format and uploaded to
    GCS?

    -   **No** → Go to [Phase 1: Dataset Preparation & Upload](#phase-1).
    -   **Yes** → Proceed.

3.  **Configuration**: Have the target Gemini model and hyperparameters been
    decided?

    -   **No** → Go to
        [Phase 2: Model Configuration & Recommendation](#phase-2).
    -   **Yes** → Proceed.

4.  **Job Status**: Has the tuning job been submitted?

    -   **No** → Go to
        [Phase 3: Tuning Job Execution](#phase-3-tuning-job-execution).
    -   **Yes** → Proceed.

5.  **Job Completion**: Is the tuning job complete?

    -   **No** → Go to [Phase 4: Monitoring](#phase-4-monitoring).
    -   **Yes** → Proceed.

6.  **Deployment**: Has the tuned model been deployed (if required)?

    -   **No** → Go to [Phase 5: Model Deployment](#phase-5-model-deployment).
    -   **Yes** → Task Complete.

--------------------------------------------------------------------------------

## Phase 0: Environment & IAM Setup {#phase-0}

Ensure the foundational environment is ready before proceeding.

### 0.1 Authentication & Project Context

-   Check if `gcloud` CLI is installed. If it is not installed, prompt the user
    for permission to install it before proceeding.
-   Verify `gcloud auth list`. If not authenticated, run `gcloud auth login`.
-   Ensure `project` and `location` are known. Use `gcloud config get project`
    to retrieve the current project (and `gcloud config get compute/region` for
    region).
-   **CRITICAL: Ask for Confirmation.** You must prompt the user to confirm the
    retrieved project and region before proceeding, in case they want to switch
    to a different one.

### 0.2 Enable APIs

Ensure `aiplatform.googleapis.com` and `storage.googleapis.com` are enabled.
```bash
gcloud services enable aiplatform.googleapis.com storage.googleapis.com --project=YOUR_PROJECT
```

### 0.3 IAM Permissions

Verify the following identities have the required roles.

-   **Vertex AI Service Agent**:
    `service-PROJECT_NUMBER@gcp-sa-aiplatform.iam.gserviceaccount.com`
-   **Managed OSS Fine Tuning Service Agent**:
    `service-PROJECT_NUMBER@gcp-sa-vertex-moss-ft.iam.gserviceaccount.com`
-   **User Identity**: The account running the commands.

### 0.4 Virtual Environment

Create and use a virtual environment named `tuning_agent_venv` in the home
directory. Install dependencies from `references/requirements.txt`.
```bash
python3 -m venv ~/tuning_agent_venv
source ~/tuning_agent_venv/bin/activate
pip install -r references/requirements.txt
```

--------------------------------------------------------------------------------

## Phase 1: Dataset Preparation & Upload {#phase-1}

Vertex AI requires valid JSONL format in GCS.

### 1.0 Dataset Discovery & Confirmation

-   **Ask the User First:** Ask the user if they already have a dataset they
    want to use.
-   **Auto-Discovery:** If the user does not have a dataset, search the
    authenticated project's GCS buckets to find if any existing file has a
    reasonable dataset that can do the job the user prompted initially.
-   **CRITICAL: Ask for Confirmation.** Do not proceed with dataset preparation
    or upload until you present the found or provided dataset to the user and
    they confirm the dataset to use.

### 1.1 Formatting & Validation

-   **Conversion**: If data is in CSV or JSON, use `scripts/prepare_dataset.py`
    to convert.
-   **Validation**: If data is already in JSONL, validate it before uploading:
    `bash python3 scripts/prepare_dataset.py \ --input my_data.jsonl \ --format
    messages_gemini \ --validate_only`
-   Refer to [Data Preparation Guide](references/data_prep.md) for required
    schemas.

### 1.2 Upload

Upload formatted `.jsonl` files to GCS using a unique directory (e.g., with a
datetime timestamp) to avoid overwriting outputs from different runs.

```bash
gcloud storage cp dataset.jsonl gs://YOUR_BUCKET/tuning_agent_job_<datetime>/dataset.jsonl
```

--------------------------------------------------------------------------------

## Phase 2: Model Configuration & Recommendation {#phase-2}

Help the user choose the best Gemini model and parameters.
**Always seek user confirmation before submitting the job.**

-   If the user does not specify a specific model in their prompt, calculate
    recommendations based on the **Models Catalog**.
-   **Prompt for Confirmation:** Present the recommended model to the user and
    ask for their confirmation before configuring hyperparameters.

### 2.1 Configuration

-   *Hyperparameter recommendation guidelines and options for Gemini models will
    be populated here.*
-   **Prompt for Confirmation:** Present the proposed configuration to the user
    and ask for their approval before proceeding.

--------------------------------------------------------------------------------

## Phase 3: Tuning Job Execution {#phase-3-tuning-job-execution}

Submit the Gemini model tuning job using `scripts/tune_gemini_model.py`.

```bash
python3 scripts/tune_gemini_model.py
```

--------------------------------------------------------------------------------

## Phase 4: Monitoring {#phase-4-monitoring}

Monitor the job via the Cloud Console link provided in the script output or by
polling the job status.

--------------------------------------------------------------------------------

## Phase 5: Model Deployment {#phase-5-model-deployment}

Once the Gemini model tuning job is `SUCCEEDED`, deploy the model using
`scripts/deploy_gemini_model.py`.

```bash
python3 scripts/deploy_gemini_model.py
```

--------------------------------------------------------------------------------

## Resources

-   [Data Preparation Guide](references/data_prep.md)
-   [Models Catalog](references/models.md)
-   [Tuning Guide](references/tuning_guide.md)
-   `scripts/prepare_dataset.py`: Data conversion & validation.
-   `scripts/tune_gemini_model.py`: Gemini model tuning job submission.
-   `scripts/deploy_gemini_model.py`: Gemini model deployment.
