## Before you begin

### Set up your Google Cloud project

**The following steps are required, regardless of your notebook environment.**

1. [Select or create a Google Cloud project](https://console.cloud.google.com/cloud-resource-manager). When you first create an account, you get a $300 free credit towards your compute/storage costs.

1. [Make sure that billing is enabled for your project](https://cloud.google.com/billing/docs/how-to/modify-project).

1. [Enable the Vertex AI, BigQuery, Compute Engine and Cloud Storage APIs](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com,bigquery,compute_component,storage_component).

1. If you are running this notebook locally, you need to install the [Cloud SDK](https://cloud.google.com/sdk).

1. Enter your project ID in the cell below. Then run the cell to make sure the
Cloud SDK uses the right project for all the commands in this notebook.

**Note**: Jupyter runs lines prefixed with `!` as shell commands, and it interpolates Python variables prefixed with `$` into these commands.

### Set up your local development environment

**If you are using Colab or Vertex AI Workbench Notebooks**, your environment already meets all the requirements to run this notebook. You can skip this step.

**Otherwise**, make sure your environment meets this notebook's requirements. You need the following:

- The Cloud Storage SDK
- Python 3
- virtualenv
- Jupyter notebook running in a virtual environment with Python 3

The Cloud Storage guide to [Setting up a Python development environment](https://cloud.google.com/python/setup) and the [Jupyter installation guide](https://jupyter.org/install) provide detailed instructions for meeting these requirements. The following steps provide a condensed set of instructions:

1. [Install and initialize the SDK](https://cloud.google.com/sdk/docs/).

2. [Install Python 3](https://cloud.google.com/python/setup#installing_python).

3. [Install virtualenv](https://cloud.google.com/python/setup#installing_and_using_virtualenv) and create a virtual environment that uses Python 3.  Activate the virtual environment.

4. To install Jupyter, run `pip3 install jupyter` on the command-line in a terminal shell.

5. To launch Jupyter, run `jupyter notebook` on the command-line in a terminal shell.

6. Open this notebook in the Jupyter Notebook Dashboard.