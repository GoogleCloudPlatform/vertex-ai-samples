![AlphaGenome header image](https://raw.githubusercontent.com/google-deepmind/alphagenome/refs/heads/main/docs/source/_static/header.png)

# AlphaGenome
[**Overview**](#overview) | [**Use Cases**](#use-cases) | [**Documentation**](#documentation) | [**Pricing**](#pricing) | [**Quick start inference**](#quick-start-inference) | 
[**Quick start finetune**](#quick-start-finetune)

## Overview
**Disclaimer:** *Experimental*.

*The AlphaGenome Private Preview is a "Pre-GA Offering" subject to the "Pre-GA
Offerings Terms" in the General Service Terms section of the Google Cloud
[Service Specific Terms](https://cloud.google.com/terms/service-terms). It is
also a “Generative AI Preview Product” as defined in and subject to the
[Additional Terms for Generative AI Preview Products](https://cloud.google.com/trustedtester/aitos?e=48754805&hl=en).
Pre-GA products are available "as is" and might have limited support. For more
information, see the [launch stage](https://cloud.google.com/products?e=48754805#product-launch-stages)
descriptions.*

Access to the AlphaGenome model capabilities requires application and approval.
Users must be added to an allowlist to use the service.
If you are interested in applying to the program, **Request Access** above.

&nbsp;

AlphaGenome is Google DeepMind’s unifying model for deciphering the regulatory
code within DNA sequences.

AlphaGenome offers multimodal predictions, encompassing diverse functional
outputs such as gene expression, splicing patterns, chromatin features, and
contact maps (see diagram below). The model analyzes DNA sequences of up to 1
million base pairs in length and can deliver predictions at single base-pair
resolution for most outputs.

Training data was sourced from large public consortia including
[ENCODE](http://encodeproject.org/), [GTEx](https://www.gtexportal.org/),
[4D Nucleome](https://4dnucleome.org/) and
[FANTOM5](https://fantom.gsc.riken.jp/5/), which experimentally measured these
properties covering important modalities of gene regulation across hundreds of
human and mouse cell types and tissues.

![Diagram showing an overview of the AlphaGenome model architecture and its inputs/outputs](https://www.alphagenomedocs.com/_images/model_overview.png)

## Use Cases
*   **Sequence-to-function predictions:** Predict multiple functional tracks (such as gene expression, splicing) from DNA sequences across a wide variety of tissues and cell types.

*   **Variant effect scoring:** Assess the impact of genetic variants by comparing predictions for the reference and alternative alleles and summarising the differences between them.

*   **Identify functional regions:** Use in silico mutagenesis (ISM) to identify functionally important regions in the DNA sequence.

*   **Human and mouse capability:** Generate predictions for both human and mouse genomes.

## Documentation
This API provides access to AlphaGenome, Google DeepMind's unifying model for
deciphering the regulatory code within DNA sequences. AlphaGenome offers
multimodal predictions, encompassing diverse functional outputs including gene
expression, splicing patterns, chromatin features, and contact maps (see diagram
below). The model analyzes up to 1 million base pairs of DNA sequence and can
deliver predictions at single base-pair resolution for most modalities.
AlphaGenome achieves state-of-the-art performance across a range of genomic
prediction benchmarks, including diverse variant effect prediction tasks.

The Google Cloud API for AlphaGenome provides a way for Google Cloud customers
to explore the AlphaGenome API for commercial use cases. This API is in private
preview (Request Access above). Once allowlisted, customers can access the API
directly or use the [colab](cloudai_alphagenome_vai_quickstart.ipynb).

### Acknowledgements

*Avsec, Ž., Latysheva, N., Cheng, J., Novati, G., Taylor, K. R., Ward, T., ... Kohli, P. (2025). AlphaGenome: advancing regulatory variant effect prediction with a unified DNA sequence model. bioRxiv.* [https://doi.org/10.1101/2025.06.25.661532](https://doi.org/10.1101/2025.06.25.661532)

### Contact
If you have any questions on using these models on Google Cloud please contact:
[alphagenome-cloud-external@google.com](mailto:alphagenome-cloud-external@google.com) or join the community [Discourse](https://www.alphagenomecommunity.com/) for more generic questions on AlphaGenome.

### Links

*   Read our [paper](https://doi.org/10.1101/2025.06.25.661532)
*   Read our [blog post](https://deepmind.google/discover/blog/alphagenome-ai-for-better-understanding-the-genome)
*   Join the [community](https://www.alphagenomecommunity.com/)
*   Check out the [AlphaGenome 101 Video](https://youtu.be/Xbvloe13nak)

## Pricing
Access to AlphaGenome on Vertex AI is currently restricted.
To utilize these models via this service:

*   You must **Request Access** using your Google contact.
*   Your application will be reviewed, and if approved, you will be **added to
    an allowlist**.
*   Only allowlisted users can access the API
*   **Pricing information** will be shared directly with users upon approval
    and placement on the allowlist.

## Quick start inference
The quickest way to get started with the AlphaGenome in Google Cloud Platform is to run [our example notebook](cloudai_alphagenome_vai_quickstart.ipynb) in [Google Colab](https://colab.research.google.com/).

## Quick start finetune
The quickest way to get started with the AlphaGenome fineutning in Google Cloud Platform is to run [our finetuning notebook](cloudai_alphagenome_finetune.ipynb) in [Google Cloud Platform Enterprise Colab](https://docs.cloud.google.com/colab/docs/introduction).
