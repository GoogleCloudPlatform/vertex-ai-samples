# ![Google Cloud](https://avatars.githubusercontent.com/u/2810941?s=60&v=4) Google Cloud Vertex AI Samples

This repository contains notebooks, code samples, sample apps, skills, and other resources that demonstrate how to use, develop and manage machine learning and generative AI workflows using Google Cloud Vertex AI.

## Overview

[Vertex AI](https://cloud.google.com/vertex-ai) is a fully-managed, unified AI development platform for building and using generative AI. This repository is designed to help you get started with Vertex AI. Whether you're new to Vertex AI or an experienced ML practitioner, you'll find valuable resources here.

⚠️ For more Vertex AI Generative AI notebook samples, please visit the Vertex AI [Generative AI](https://github.com/GoogleCloudPlatform/generative-ai) GitHub repository.

## Explore, learn and contribute

You can explore, learn, and contribute to this repository to unleash the full potential of machine learning on Vertex AI! 

### Explore and learn

Explore this repository, follow the links in the header section of each of the notebooks to -

![Colab](https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg)  Open and run the notebook in [Colab](https://colab.google/)\
![Colab Enterprise](https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN)  Open and run the notebook in [Colab Enterprise](https://cloud.google.com/colab/docs/introduction)\
![Workbench](https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg)  Open and run the notebook in [Vertex AI Workbench](https://cloud.google.com/vertex-ai/docs/workbench/introduction)\
![Github](https://raw.githubusercontent.com/primer/octicons/refs/heads/main/icons/mark-github-24.svg)  View the notebook on Github

### Contribute

See the [Contributing Guide](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/master/CONTRIBUTING.md).

## Get started

To get started using Vertex AI, you must have a Google Cloud project. 

- If you don't have a Google Cloud project, you can learn and build on GCP for free using [Free Trail](https://cloud.google.com/free).
- Once you have a Google Cloud project, you can learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).


## Repository structure

```text
├── notebooks
│   ├── official - Notebooks demonstrating use of each Vertex AI service
│   │   ├── automl
│   │   ├── custom
│   │   ├── ...
│   ├── community - Notebooks contributed by the community
│   │   ├── model_garden
│   │   ├── ...
├── community-content - Sample code and tutorials contributed by the community
├── docs - Deep-dive documentation and advanced setup guides
└── skills - Suite of AI Agent "Skills" for Vertex AI
    ├── README.md               # Developer guide for Vertex AI skills
    ├── vertex-ai/              # Primary router for Vertex AI tasks
    │   └── SKILL.md            # Entry point that routes across capabilities
    ├── genai-sdk/              # Gemini API usage with Gen AI SDK
    │   └── SKILL.md            # Guides for Python, JS/TS, Go, Java, C#
    ├── vertex-deploy/          # Deploying models to Endpoints
    │   └── SKILL.md            # Commands for open models & custom weights
    ├── vertex-inference/       # Inferencing with GenAI models
    │   └── SKILL.md            # Code samples for Gemini and OpenMaaS
    └── vertex-tuning/          # Secondary router for model fine-tuning
        ├── SKILL.md            # Router for tuning tasks
        ├── gemini/             # Fine-tuning first-party Gemini models
        │   └── SKILL.md
        └── open-model/         # Fine-tuning third-party open models
            └── SKILL.md
```
## Examples

<!-- markdownlint-disable MD033 -->
<table>

  <tr>
    <th style="text-align: center;">Category</th>
    <th style="text-align: center;">Product</th>
    <th style="text-align: center;">Description</th>
  </tr>
  <tr>
    <td>Model</td>
    <td>
      <a href="notebooks/community/model_garden"><code>Model Garden/</code></a>
    </td>
    <td>
      Curated collection of first-party, open-source, and third-party models available on Vertex AI including Gemini, Gemma, Llama 3, Claude 3 and many more.
    </td>
  </tr>
  <tr>
    <td>Data</td>
    <td>
      <a href="notebooks/official/feature_store"><code>Feature Store/</code></a>
    </td>
    <td>
      Set up and manage online serving using Vertex AI Feature Store.
    </td>
  </tr>
  <tr>
    <td></td>
    <td>
      <a href="notebooks/official/datasets"><code>datasets/</code></a>
    </td>
    <td>
      Use BigQuery and Data Labeling service with Vertex AI.
    </td>
  </tr>
  <tr>
    <td>Model development</td>
    <td>
      <a href="notebooks/official/automl"><code>automl/</code></a>
    </td>
    <td>
      Train and make predictions on AutoML models
    </td>
  </tr>
  <tr>
    <td></td>
    <td>
      <a href="notebooks/official/custom"><code>custom/</code></a>
    </td>
    <td>
      Create, deploy and serve custom  models on Vertex AI
    </td>
  </tr>
  <tr>
    <td></td>
    <td>
      <a href="notebooks/official/ray_on_vertex_ai"><code>ray_on_vertex_ai/</code></a>
    </td>
    <td>
      Use Colab Enterprise and Vertex AI SDK for Python to connect to the Ray Cluster.
    </td>
  </tr>
  <tr>
    <td>Deploy and use</td>
    <td>
      <a href="notebooks/official/prediction"><code>prediction/</code></a>
    </td>
    <td>
      Build, train and deploy models using prebuilt containers for custom training and prediction.
    </td>
  </tr>
  <tr>
    <td></td>
    <td>
      <a href="notebooks/official/model_registry"><code>model_registry/</code></a>
    </td>
    <td>
      Use Model Registry to create and register a model.
    </td>
  </tr>
  <tr>
    <td></td>
    <td>
      <a href="notebooks/official/explainable_ai"><code>Explainable AI/</code></a>
    </td>
    <td>
      Use Vertex Explainable AI's feature-based and example-based explanations to explain how or why a model produced a specific prediction.
    </td>
  </tr>
  <tr>
    <td></td>
    <td>
      <a href="notebooks/official/ml_metadata"><code>ml_metadata/</code></a>
    </td>
    <td>
      Record the metadata and artifacts and query that metadata to help analyze, debug, and audit the performance of your ML system.
    </td>
  </tr>
  <tr>
    <td>Tools</td>
    <td>
      <a href="notebooks/official/pipelines"><code>Pipelines/</code></a>
    </td>
    <td>
      Use `Vertex AI Pipelines` and `Google Cloud Pipeline Components` to build, tune, or deploy a custom model.
    </td>
  </tr>
</table>
<!-- markdownlint-enable MD033 -->


## Get help

Please use the [Issues page](https://github.com/GoogleCloudPlatform/vertex-ai-samples/issues) to provide feedback or submit a bug report.

## Disclaimer

This is not an officially supported Google product. The code in this repository is for demonstrative purposes only.


## References
- [Vertex AI Jupyter Notebook tutorials](https://cloud.google.com/vertex-ai/docs/tutorials/jupyter-notebooks)
- Vertex AI [Generative AI](https://github.com/GoogleCloudPlatform/generative-ai) GitHub repository
- [Vertex AI documentaton](https://cloud.google.com/vertex-ai/docs)
  
