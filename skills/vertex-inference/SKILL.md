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
name: Vertex AI GenAI Inference
description: Instructions for connecting to and performing inference with Google Cloud Vertex AI GenAI models, including Gemini and OpenMaaS (Llama, DeepSeek, Qwen, etc.).
---

# Vertex AI GenAI Inference Skill

This skill provides instructions for authenticating and connecting to Google Cloud Vertex AI to use Generative AI models. It covers both First-Party (Gemini) and Third-Party (OpenMaaS) models.

> [!TIP]
> **Sample Scripts**: This skill includes fully functional sample scripts in the `scripts/` directory (e.g., `scripts/openmaas_openai_sdk.py`). When running these scripts, **ALWAYS** create and use a local virtual environment:
> ```bash
> python3 -m venv .venv && source .venv/bin/activate
> pip install -r scripts/requirements.txt
> ```
>
> **Verify All Scripts**: You can run all scripts at once to verify your setup:
> ```bash
> ./scripts/verify_all.sh
> ```

> [!IMPORTANT]
> **CRITICAL: Model IDs & Availability**
> *   **Gemini Models**: See [Gemini Models](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/migrate) for valid Model IDs and Regions.
> *   **OpenMaaS Models**: See [Use Open Models on Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/maas/use-open-models) for Llama, DeepSeek, Qwen, etc.
> *   **Incomplete Lists**: The Model IDs listed in this skill are **examples only** and may be incomplete or outdated.
> *   **Action**: Always verify the Model ID and Region using the links above before generating code.

## 1. Authentication (CRITICAL)

Before running any code, ensure you are authenticated with Application Default Credentials (ADC) and have the necessary API enabled.

1.  **Login**:
    ```bash
    gcloud auth application-default login
    ```
2.  **Enable API** (if not already enabled):
    ```bash
    gcloud services enable aiplatform.googleapis.com
    ```

## 2. Gemini Models

For Gemini models (e.g., `gemini-1.5-pro`, `gemini-2.0-flash`), the **GenAI SDK** (`google-genai`) is the **PREFERRED** method. The legacy `vertexai` SDK is still supported but GenAI SDK is recommended for new projects.

> [!IMPORTANT]
> **Preview Models (including Gemini 2.0)** are often **ONLY** available in the `global` region. Stable models are available in `us-central1` and other regions.

### Choosing the Right SDK

*   **Gemini Models**: **GenAI SDK** (`google-genai`) is **PREFERRED**. Use OpenAI SDK for compatibility, or Legacy SDK (`vertexai`) if needed.
*   **OpenMaaS Models**: **OpenAI SDK** is **HIGHLY RECOMMENDED**. Use GenAI SDK or Legacy SDK if you have specific infrastructure requirements.

### Installation

```bash
pip install google-genai
```

### Python Example (GenAI SDK - Preferred)

See [`scripts/gemini_genai_sdk.py`](scripts/gemini_genai_sdk.py) for the complete code.

### Alternative: OpenAI SDK (Chat Completions)

Use the standard OpenAI SDK with the Vertex AI endpoint. This is great for cross-compatibility.

See [`scripts/gemini_openai_sdk.py`](scripts/gemini_openai_sdk.py) for the complete code.

### Legacy: Vertex AI SDK

The legacy `vertexai` SDK is still widely used but `google-genai` is preferred for new Gemini projects.

See [`scripts/gemini_vertexai_sdk.py`](scripts/gemini_vertexai_sdk.py) for the complete code.

**Documentation**: [Google GenAI SDK](https://github.com/googleapis/python-genai)

**Documentation**: [Vertex AI Gemini Models](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models)

## 3. OpenMaaS Models (Llama, DeepSeek, Qwen, etc.)

For OpenMaaS (Model-as-a-Service) models, the **HIGHLY RECOMMENDED** approach is to use the standard **OpenAI SDK** with a specific Vertex AI endpoint.

> [!WARNING]
> While `GenerativeModel` *can* support some OpenMaaS models, it is **discouraged**. Use the OpenAI SDK for best compatibility (especially for Chat Completions).

### Installation

```bash
pip install openai google-auth
```

### Authentication for OpenAI SDK

You **MUST** use a Google Cloud OAuth access token as the API key for the OpenAI SDK.

```python
import google.auth
from google.auth.transport.requests import Request

def get_gcp_access_token():
    creds, _ = google.auth.default()
    creds.refresh(Request())
    return creds.token

> [!NOTE]
> Google Cloud access tokens typically expire after 1 hour. The `get_gcp_access_token()` function above retrieves a *fresh* token at the time it is called.
> For long-running applications, you implement a refresh mechanism. See [Refresh the access token](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/openai-sdk-auth#refresh-token) for details.
```

### Configuration (Base URL)

-   **Global Endpoint** (Recommended for most models requiring global availability):
    `https://aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/global/endpoints/openapi`
-   **Regional Endpoint**:
    `https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/openapi`

### Python Example (OpenMaaS - Chat Completions)

See [`scripts/openmaas_openai_sdk.py`](scripts/openmaas_openai_sdk.py) for the complete code.

> [!TIP]
> **Alternative: Environment Variables**
> You can set environment variables in your shell instead of updated the code.
> ```bash
> export OPENAI_BASE_URL="https://aiplatform.googleapis.com/v1/projects/YOUR_PROJECT_ID/locations/global/endpoints/openapi"
> export OPENAI_API_KEY="$(gcloud auth print-access-token)"
> ```
> Then initialize the client without arguments: `client = OpenAI()`

### Python Example (OpenMaaS - Completions API)

The following models support the legacy Completions API: `zai-org/glm-5-maas`, `moonshotai/kimi-k2-thinking-maas`, `minimaxai/minimax-m2-maas`, `deepseek-ai/deepseek-v3.1-maas`, and `deepseek-ai/deepseek-v3.2-maas`.

```python
response = client.completions.create(
    model="deepseek-ai/deepseek-v3.2-maas",
    prompt="Once upon a time",
    max_tokens=100
)
print(response.choices[0].text)
```

### Python Example (OpenMaaS - Embeddings)

```python
# Verify specific Embedding Model ID on Model Garden (e.g., intfloat/multilingual-e5-small)
response = client.embeddings.create(
    model="intfloat/multilingual-e5-large-maas",
    input="The quick brown fox jumps over the lazy dog",
)
print(response.data[0].embedding)
```

### Alternative: GenAI SDK

The `google-genai` SDK can also access OpenMaaS models via the `vertexai` backend.

See [`scripts/openmaas_genai_sdk.py`](scripts/openmaas_genai_sdk.py) for the complete code.

> [!IMPORTANT]
> **Model ID Format**: For GenAI SDK with OpenMaaS, you **MUST** use the full path: `publishers/PUBLISHER/models/MODEL` (e.g., `publishers/zai-org/models/glm-5-maas`).

### Legacy: Vertex AI SDK (OpenMaaS)

For OpenMaaS, you can also use `GenerativeModel` (if supported).

See [`scripts/openmaas_vertexai_sdk.py`](scripts/openmaas_vertexai_sdk.py) for the complete code.

> [!IMPORTANT]
> **Model ID Format**: For Vertex AI SDK with OpenMaaS, you **MUST** use the full path: `publishers/PUBLISHER/models/MODEL`.

### Model Reference & Availability

**Documentation**: [Use Open Models on Vertex AI](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/maas/use-open-models)

> [!TIP]
> **Self-Deployment for Control**: If you need **dedicated hardware** (GPUs/TPUs), **guaranteed capacity**, or **specific regional placement** not offered by MaaS, you can **Self-Deploy** these models to Vertex AI Endpoints. Search for the model in Model Garden and click "Deploy" to select your machine type.


> [!IMPORTANT]
> **Finding Inference Examples**: The list above is a starting point. For the **definitive** inference snippets (especially for Chat Completions payload structure):
> 1.  Consult the [Use Open Models on Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/maas/use-open-models) list.
> 2.  Click the link for your specific model (e.g., "DeepSeek-V3") to visit its **Model Garden** page.
> 3.  Look for the **"Sample Code"** or **"Use this model"** button on the Model Garden page to get the exact `curl` or Python code for that specific model version.

> [!NOTE]
> This list is **INCOMPLETE**. See [Use Open Models on Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/maas/use-open-models) for the full list of supported models.

| Model Family | Model ID Examples | Location | Notes |
| :--- | :--- | :--- | :--- |
| **Llama 4** | `meta/llama-4-maverick-17b-128e-instruct-maas` | `us-east5` | |
| **Llama 4** | `meta/llama-4-scout-17b-16e-instruct-maas` | `us-east5` | |
| **Llama 3.3** | `meta/llama-3.3-70b-instruct-maas` | `us-central1` | |
| **DeepSeek** | `deepseek-ai/deepseek-v3.2-maas` | `global` | Global ONLY |
| **DeepSeek** | `deepseek-ai/deepseek-v3.1-maas` | `us-west2` | US-West2 ONLY |
| **DeepSeek** | `deepseek-ai/deepseek-r1-0528-maas` | `us-central1` | |
| **Qwen 3** | `qwen/qwen3-coder-480b-a35b-instruct-maas` | `global` | |
| **Qwen 3** | `qwen/qwen3-next-80b-a3b-instruct-maas` | `global` | |
| **Kimi** | `moonshotai/kimi-k2-thinking-maas` | `global` | |
| **MiniMax** | `minimaxai/minimax-m2-maas` | `global` | |
| **GLM** | `zai-org/glm-4.7-maas`, `zai-org/glm-5-maas` | `global` | |

## 4. Troubleshooting & Common Error Codes

### 429: Resource Exhausted

*   **Cause**: OpenMaaS and Gemini models use **Dynamic Shared Quota (DSQ)**. Resources are pooled and allocated dynamically based on availability. A 429 error indicates the shared pool is temporarily exhausted, not necessarily that *your* specific project quota is hit (though it can be).
*   **Solution**: Implement strict **exponential backoff and retry** strategies.
*   **High Throughput**: For production workloads requiring high throughput or guaranteed capacity, consider **Provisioned Throughput (PT)**.
*   **Important**: Quota increases through normal cloud processes (Cloud Console) are **NOT** applicable for DSQ constraints.
*   **Documentation**: [Quotas and limits (DSQ)](https://cloud.google.com/vertex-ai/generative-ai/docs/quotas)

### 400: User Validation Error

*   **Cause**: Invalid request format, unsupported parameter, or incorrect Model ID.
*   **Action**: Double-check your request payload and parameters. Verify the Model ID and Region are correct.

### 404: Not Found / Model Not Available

*   **Cause**: The model is not enabled, or not available in the specified project or region.
*   **Action**:
    1.  **Check Location Availability**:
        *   **OpenMaaS**: Verify the model is available in your region. See [Model Availability by Location](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/locations#genai-open-models).
        *   **Gemini**:
            *   **Source of Truth**: Always check [Gemini Model Locations](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations#google-models) for the authoritative list.
            *   **Preview Models**: All Preview models (e.g., Gemini 2.0, experimental versions) are often **ONLY** available in the `us-central1` or `global` regions.
            *   **Stable Models**: (e.g., Gemini 1.5 Pro) Available in `us-central1`, `europe-west4`, and many other regions.
            *   **Important**: If you get a 404/400 error, try switching your client location to `us-central1` or `global`.
    2.  **Enable Llama Models**: For **Llama 3.3** and **Llama 4**, you **MUST** enable the model in Model Garden before use. Go to the [Model Garden](https://console.cloud.google.com/vertex-ai/model-garden), search for the model card (e.g., "Llama 3.3 API Service"), and click **Enable**. Only then can you make inference requests.
