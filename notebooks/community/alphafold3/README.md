# AlphaFold 3
[**Overview**](#overview) | [**Use cases**](#use-cases) | [**Documentation**](#documentation) | [**Prerequisites**](#prerequisites) | [**Quick start**](#quick-start)

## Overview
AlphaFold 3 is a revolutionary model developed by Google DeepMind and Isomorphic Labs that predicts the 3D structures and interactions of proteins, DNA, RNA, ligands, and chemical modifications.

By modeling these molecules and their interactions together in a unified diffusion-based architecture, AlphaFold 3 provides a comprehensive view of cellular machinery, enabling researchers to understand biological processes at atomic resolution.

AlphaFold 3 is available for commercial use on [Gemini Enterprise Agent Platform](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/open-models/alphafold-3).

## Use cases
* **Protein-Ligand Interaction Prediction**: Model the binding of small molecule ligands to proteins, enabling drug discovery and development.
* **Nucleic Acid Interaction Prediction**: Predict the complex structures of proteins interacting with DNA and RNA sequences.
* **Chemical Modifications**: Predict structures containing modified residues, ions, and covalent linkages.
* **Antibody-Antigen Modeling**: Map the 3D structures of antibody-antigen complexes to support therapeutic antibody design.

## Documentation
The examples provided here demonstrate how to deploy and use AlphaFold 3 on Gemini Enterprise Agent Platform.

### Links
* Read the [Nature journal paper](https://doi.org/10.1038/s41586-024-07487-w)
* Read the [Google DeepMind blog post](https://blog.google/technology/ai/google-deepmind-isomorphic-alphafold-3-ai-model/)
* Explore the [AlphaFold Server](https://alphafoldserver.com/welcome)
* View the open-source code and non-commercial weights on [GitHub](https://github.com/google-deepmind/alphafold3)

## Prerequisites
To deploy and use AlphaFold 3 on Vertex AI:
1. **Request Access**: Submit the [AlphaFold 3 Request Form](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/alphafold3-request) and work with your Google Cloud account team for commercial subscription allowlisting.
2. **Hardware Quota**: Deployments require an `a3-highgpu-1g` machine type (1x NVIDIA H100 80GB GPU) with 750 GB Local SSD provisioned for database caching.
3. **Endpoint Configuration**: Deploy the model to a Dedicated Endpoint and configure the inference timeout to 3,600 seconds.

## Quick start
| Notebook | Description | Links |
| :--- | :--- | :--- |
| [AlphaFold 3 Quickstart](cloudai_alphafold3_vai_quickstart.ipynb) | End-to-end protein-ligand docking prediction (KRAS G12C covalent complex with Sotorasib), output handling, and 3D visualization. | <a href="https://colab.research.google.com/github/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/alphafold3/cloudai_alphafold3_vai_quickstart.ipynb"><img src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Open in Colab" height="20"></a> |
