{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ur8xi4C7S06n"
      },
      "outputs": [],
      "source": [
        "# Copyright 2021 Google LLC\n",
        "#\n",
        "# Licensed under the Apache License, Version 2.0 (the \"License\");\n",
        "# you may not use this file except in compliance with the License.\n",
        "# You may obtain a copy of the License at\n",
        "#\n",
        "#     https://www.apache.org/licenses/LICENSE-2.0\n",
        "#\n",
        "# Unless required by applicable law or agreed to in writing, software\n",
        "# distributed under the License is distributed on an \"AS IS\" BASIS,\n",
        "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n",
        "# See the License for the specific language governing permissions and\n",
        "# limitations under the License."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "zFWqF3Y4Gilg"
      },
      "source": [
        "# Guide to Building End-to-End Reinforcement Learning Application Pipelines using Vertex AI"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "JAPoU8Sm5E6e"
      },
      "source": [
        "<table align=\"left\">\n",
        "\n",
        "  <td>\n",
        "    <a href=\"https://colab.research.google.com/github/GoogleCloudPlatform/vertex-ai-samples/tree/master/community-content/tf_agents_bandits_movie_recommendation_with_kfp_and_vertex_sdk/mlops_pipeline_tf_agents_bandits_movie_recommendation/mlops_pipeline_tf_agents_bandits_movie_recommendation.ipynb\">\n",
        "      <img src=\"https://cloud.google.com/ml-engine/images/colab-logo-32px.png\" alt=\"Colab logo\"> Run in Colab\n",
        "    </a>\n",
        "  </td>\n",
        "  <td>\n",
        "    <a href=\"https://github.com/GoogleCloudPlatform/vertex-ai-samples/tree/master/community-content/tf_agents_bandits_movie_recommendation_with_kfp_and_vertex_sdk/mlops_pipeline_tf_agents_bandits_movie_recommendation/mlops_pipeline_tf_agents_bandits_movie_recommendation.ipynb\">\n",
        "      <img src=\"https://cloud.google.com/ml-engine/images/github-logo-32px.png\" alt=\"GitHub logo\">\n",
        "      View on GitHub\n",
        "    </a>\n",
        "  </td>\n",
        "</table>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "tvgnzT1CKxrO"
      },
      "source": [
        "## Overview\n",
        "\n",
        "This demo showcases the use of [TF-Agents](https://www.tensorflow.org/agents), [Kubeflow Pipelines (KFP)](https://www.kubeflow.org/docs/components/pipelines/overview/pipelines-overview/) and [Vertex AI](https://cloud.google.com/vertex-ai), particularly [Vertex Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines), in building an end-to-end reinforcement learning (RL) pipeline of a movie recommendation system. The demo is intended for developers who want to create RL applications using TensorFlow, TF-Agents and Vertex AI services, and those who want to build end-to-end production pipelines using KFP and Vertex Pipelines. It is recommended for developers to have familiarity with RL and the contextual bandits formulation, and the TF-Agents interface.\n",
        "\n",
        "### Dataset\n",
        "\n",
        "This demo uses the [MovieLens 100K](https://www.kaggle.com/prajitdatta/movielens-100k-dataset) dataset to simulate an environment with users and their respective preferences. It is available at `gs://cloud-samples-data/vertex-ai/community-content/tf_agents_bandits_movie_recommendation_with_kfp_and_vertex_sdk/u.data`.\n",
        "\n",
        "### Objective\n",
        "\n",
        "In this notebook, you will learn how to build an end-to-end RL pipeline for a TF-Agents (particularly the bandits module) based movie recommendation system, using [KFP](https://www.kubeflow.org/docs/components/pipelines/overview/pipelines-overview/), [Vertex AI](https://cloud.google.com/vertex-ai) and particularly [Vertex Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines) which is fully managed and highly scalable.\n",
        "\n",
        "This Vertex Pipeline includes the following components:\n",
        "1. *Generator* to generate MovieLens simulation data\n",
        "2. *Ingester* to ingest data\n",
        "3. *Trainer* to train the RL policy\n",
        "4. *Deployer* to deploy the trained policy to a Vertex AI endpoint\n",
        "\n",
        "After pipeline construction, you (1) create the *Simulator* (which utilizes Cloud Functions, Cloud Scheduler and Pub/Sub) to send simulated MovieLens prediction requests, (2) create the *Logger* to asynchronously log prediction inputs and results (which utilizes Cloud Functions, Pub/Sub and a hook in the prediction code), and (3) create the *Trigger* to trigger recurrent re-training.\n",
        "\n",
        "A more general ML pipeline is demonstrated in [MLOps on Vertex AI](https://github.com/ksalama/ucaip-labs).\n",
        "\n",
        "### Costs\n",
        "\n",
        "This tutorial uses billable components of Google Cloud:\n",
        "\n",
        "* Vertex AI\n",
        "* BigQuery\n",
        "* Cloud Build\n",
        "* Cloud Functions\n",
        "* Cloud Scheduler\n",
        "* Cloud Storage\n",
        "* Pub/Sub\n",
        "\n",
        "Learn about [Vertex AI\n",
        "pricing](https://cloud.google.com/vertex-ai/pricing), [BigQuery pricing](https://cloud.google.com/bigquery/pricing), [Cloud Build](https://cloud.google.com/build/pricing), [Cloud Functions](https://cloud.google.com/functions/pricing), [Cloud Scheduler](https://cloud.google.com/scheduler/pricing), [Cloud Storage\n",
        "pricing](https://cloud.google.com/storage/pricing), and [Pub/Sub pricing](https://cloud.google.com/pubsub/pricing), and use the [Pricing\n",
        "Calculator](https://cloud.google.com/products/calculator/)\n",
        "to generate a cost estimate based on your projected usage."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ze4-nDLfK4pw"
      },
      "source": [
        "### Set up your local development environment\n",
        "\n",
        "**If you are using Colab or Google Cloud Notebooks**, your environment already meets\n",
        "all the requirements to run this notebook. You can skip this step."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "gCuSR8GkAgzl"
      },
      "source": [
        "**Otherwise**, make sure your environment meets this notebook's requirements.\n",
        "You need the following:\n",
        "\n",
        "* The Google Cloud SDK\n",
        "* Git\n",
        "* Python 3\n",
        "* virtualenv\n",
        "* Jupyter notebook running in a virtual environment with Python 3\n",
        "\n",
        "The Google Cloud guide to [Setting up a Python development\n",
        "environment](https://cloud.google.com/python/setup) and the [Jupyter\n",
        "installation guide](https://jupyter.org/install) provide detailed instructions\n",
        "for meeting these requirements. The following steps provide a condensed set of\n",
        "instructions:\n",
        "\n",
        "1. [Install and initialize the Cloud SDK.](https://cloud.google.com/sdk/docs/)\n",
        "\n",
        "1. [Install Python 3.](https://cloud.google.com/python/setup#installing_python)\n",
        "\n",
        "1. [Install\n",
        "   virtualenv](https://cloud.google.com/python/setup#installing_and_using_virtualenv)\n",
        "   and create a virtual environment that uses Python 3. Activate the virtual environment.\n",
        "\n",
        "1. To install Jupyter, run `pip3 install jupyter` on the\n",
        "command-line in a terminal shell.\n",
        "\n",
        "1. To launch Jupyter, run `jupyter notebook` on the command-line in a terminal shell.\n",
        "\n",
        "1. Open this notebook in the Jupyter Notebook Dashboard."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "i7EUnXsZhAGF"
      },
      "source": [
        "### Install additional packages\n",
        "\n",
        "Install additional package dependencies not installed in your notebook environment, such as the Kubeflow Pipelines (KFP) SDK."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "2b4ef9b72d43"
      },
      "outputs": [],
      "source": [
        "import os\n",
        "\n",
        "# The Google Cloud Notebook product has specific requirements\n",
        "IS_GOOGLE_CLOUD_NOTEBOOK = os.path.exists(\"/opt/deeplearning/metadata/env_version\")\n",
        "\n",
        "# Google Cloud Notebook requires dependencies to be installed with '--user'\n",
        "USER_FLAG = \"\"\n",
        "if IS_GOOGLE_CLOUD_NOTEBOOK:\n",
        "    USER_FLAG = \"--user\""
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "wyy5Lbnzg5fi"
      },
      "outputs": [],
      "source": [
        "! pip3 install {USER_FLAG} google-cloud-aiplatform\n",
        "! pip3 install {USER_FLAG} google-cloud-pipeline-components\n",
        "! pip3 install {USER_FLAG} --upgrade kfp\n",
        "! pip3 install {USER_FLAG} numpy\n",
        "! pip3 install {USER_FLAG} --upgrade tensorflow\n",
        "! pip3 install {USER_FLAG} --upgrade pillow\n",
        "! pip3 install {USER_FLAG} --upgrade tf-agents\n",
        "! pip3 install {USER_FLAG} --upgrade fastapi"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "hhq5zEbGg0XX"
      },
      "source": [
        "### Restart the kernel\n",
        "\n",
        "After you install the additional packages, you need to restart the notebook kernel so it can find the packages."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "EzrelQZ22IZj"
      },
      "outputs": [],
      "source": [
        "# Automatically restart kernel after installs\n",
        "import os\n",
        "\n",
        "if not os.getenv(\"IS_TESTING\"):\n",
        "    # Automatically restart kernel after installs\n",
        "    import IPython\n",
        "\n",
        "    app = IPython.Application.instance()\n",
        "    app.kernel.do_shutdown(True)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "lWEdiXsJg0XY"
      },
      "source": [
        "## Before you begin\n",
        "\n",
        "### Select a GPU runtime\n",
        "\n",
        "**Make sure you're running this notebook in a GPU runtime if you have that option. In Colab, select \"Runtime --> Change runtime type > GPU\"**"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "BF1j6f9HApxa"
      },
      "source": [
        "### Set up your Google Cloud project\n",
        "\n",
        "**The following steps are required, regardless of your notebook environment.**\n",
        "\n",
        "1. [Select or create a Google Cloud project](https://console.cloud.google.com/cloud-resource-manager). When you first create an account, you get a $300 free credit towards your compute/storage costs.\n",
        "\n",
        "1. [Make sure that billing is enabled for your project](https://cloud.google.com/billing/docs/how-to/modify-project).\n",
        "\n",
        "1. [Enable the Vertex AI API, BigQuery API, Cloud Build, Cloud Functions, Cloud Scheduler, Cloud Storage, and Pub/Sub API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com,bigquery.googleapis.com,build.googleapis.com,functions.googleapis.com,scheduler.googleapis.com,storage.googleapis.com,pubsub.googleapis.com).\n",
        "\n",
        "1. If you are running this notebook locally, you will need to install the [Cloud SDK](https://cloud.google.com/sdk).\n",
        "\n",
        "1. Enter your project ID in the cell below. Then run the cell to make sure the\n",
        "Cloud SDK uses the right project for all the commands in this notebook.\n",
        "\n",
        "**Note**: Jupyter runs lines prefixed with `!` as shell commands, and it interpolates Python variables prefixed with `$` into these commands."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "WReHDGG5g0XY"
      },
      "source": [
        "#### Set your project ID\n",
        "\n",
        "**If you don't know your project ID**, you may be able to get your project ID using `gcloud`."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "oM1iC_MfAts1"
      },
      "outputs": [],
      "source": [
        "import os\n",
        "\n",
        "PROJECT_ID = \"\"\n",
        "\n",
        "# Get your Google Cloud project ID from gcloud\n",
        "if not os.getenv(\"IS_TESTING\"):\n",
        "    shell_output = !gcloud config list --format 'value(core.project)' 2>/dev/null\n",
        "    PROJECT_ID = shell_output[0]\n",
        "    print(\"Project ID: \", PROJECT_ID)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "qJYoRfYng0XZ"
      },
      "source": [
        "Otherwise, set your project ID here."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "riG_qUokg0XZ"
      },
      "outputs": [],
      "source": [
        "if PROJECT_ID == \"\" or PROJECT_ID is None:\n",
        "    PROJECT_ID = \"[your-project-id]\"  # @param {type:\"string\"}"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "06571eb4063b"
      },
      "source": [
        "#### Timestamp\n",
        "\n",
        "If you are in a live tutorial session, you might be using a shared test account or project. To avoid name collisions between users on resources created, you create a timestamp for each instance session, and append it onto the name of resources you create in this tutorial."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "697568e92bd6"
      },
      "outputs": [],
      "source": [
        "from datetime import datetime\n",
        "\n",
        "TIMESTAMP = datetime.now().strftime(\"%Y%m%d%H%M%S\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "dr--iN2kAylZ"
      },
      "source": [
        "### Authenticate your Google Cloud account\n",
        "\n",
        "**If you are using Google Cloud Notebooks**, your environment is already\n",
        "authenticated. Skip this step."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "sBCra4QMA2wR"
      },
      "source": [
        "**If you are using Colab**, run the cell below and follow the instructions\n",
        "when prompted to authenticate your account via oAuth.\n",
        "\n",
        "**Otherwise**, follow these steps:\n",
        "\n",
        "1. In the Cloud Console, go to the [**Create service account key**\n",
        "   page](https://console.cloud.google.com/apis/credentials/serviceaccountkey).\n",
        "\n",
        "2. Click **Create service account**.\n",
        "\n",
        "3. In the **Service account name** field, enter a name, and\n",
        "   click **Create**.\n",
        "\n",
        "4. In the **Grant this service account access to project** section, click the **Role** drop-down list. Type \"Vertex AI\"\n",
        "into the filter box, and select\n",
        "   **Vertex AI Administrator**. Type \"Storage Object Admin\" into the filter box, and select **Storage Object Admin**.\n",
        "\n",
        "5. Click *Create*. A JSON file that contains your key downloads to your\n",
        "local environment.\n",
        "\n",
        "6. Enter the path to your service account key as the\n",
        "`GOOGLE_APPLICATION_CREDENTIALS` variable in the cell below and run the cell."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "PyQmSRbKA8r-"
      },
      "outputs": [],
      "source": [
        "import os\n",
        "import sys\n",
        "\n",
        "# If you are running this notebook in Colab, run this cell and follow the\n",
        "# instructions to authenticate your GCP account. This provides access to your\n",
        "# Cloud Storage bucket and lets you submit training jobs and prediction\n",
        "# requests.\n",
        "\n",
        "# The Google Cloud Notebook product has specific requirements\n",
        "IS_GOOGLE_CLOUD_NOTEBOOK = os.path.exists(\"/opt/deeplearning/metadata/env_version\")\n",
        "\n",
        "# If on Google Cloud Notebooks, then don't execute this code\n",
        "if not IS_GOOGLE_CLOUD_NOTEBOOK:\n",
        "    if \"google.colab\" in sys.modules:\n",
        "        from google.colab import auth as google_auth\n",
        "\n",
        "        google_auth.authenticate_user()\n",
        "\n",
        "    # If you are running this notebook locally, replace the string below with the\n",
        "    # path to your service account key and run this cell to authenticate your GCP\n",
        "    # account.\n",
        "    elif not os.getenv(\"IS_TESTING\"):\n",
        "        %env GOOGLE_APPLICATION_CREDENTIALS ''"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "zgPO1eR3CYjk"
      },
      "source": [
        "### Create a Cloud Storage bucket\n",
        "\n",
        "**The following steps are required, regardless of your notebook environment.**\n",
        "\n",
        "In this tutorial, a Cloud Storage bucket holds the MovieLens dataset files to be used for model training. Vertex AI also saves the trained model that results from your training job in the same bucket. Using this model artifact, you can then create Vertex AI model and endpoint resources in order to serve online predictions.\n",
        "\n",
        "Set the name of your Cloud Storage bucket below. It must be unique across all\n",
        "Cloud Storage buckets.\n",
        "\n",
        "You may also change the `REGION` variable, which is used for operations\n",
        "throughout the rest of this notebook. Make sure to [choose a region where Vertex AI services are\n",
        "available](https://cloud.google.com/vertex-ai/docs/general/locations#available_regions). You may\n",
        "not use a Multi-Regional Storage bucket for training with Vertex AI. Also note that Vertex\n",
        "Pipelines is currently only supported in select regions such as \"us-central1\" ([reference](https://cloud.google.com/vertex-ai/docs/general/locations))."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "MzGDU7TWdts_"
      },
      "outputs": [],
      "source": [
        "BUCKET_NAME = \"gs://[your-bucket-name]\"  # @param {type:\"string\"}\n",
        "REGION = \"[your-region]\"  # @param {type:\"string\"}"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "cf221059d072"
      },
      "outputs": [],
      "source": [
        "if BUCKET_NAME == \"\" or BUCKET_NAME is None or BUCKET_NAME == \"gs://[your-bucket-name]\":\n",
        "    BUCKET_NAME = \"gs://\" + PROJECT_ID + \"aip-\" + TIMESTAMP"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "-EcIXiGsCePi"
      },
      "source": [
        "**Only if your bucket doesn't already exist**: Run the following cell to create your Cloud Storage bucket."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "NIq7R4HZCfIc"
      },
      "outputs": [],
      "source": [
        "! gsutil mb -l $REGION $BUCKET_NAME"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ucvCsknMCims"
      },
      "source": [
        "Finally, validate access to your Cloud Storage bucket by examining its contents:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "vhOb7YnwClBb"
      },
      "outputs": [],
      "source": [
        "! gsutil ls -al $BUCKET_NAME"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "XoEqT2Y4DJmf"
      },
      "source": [
        "### Import libraries and define constants"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "pRUOFELefqf1"
      },
      "outputs": [],
      "source": [
        "import os\n",
        "import sys\n",
        "\n",
        "from google.cloud import aiplatform\n",
        "from google_cloud_pipeline_components import aiplatform as gcc_aip\n",
        "from kfp.v2 import compiler, dsl\n",
        "from kfp.v2.google.client import AIPlatformClient"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "eIMBwJpVVaU3"
      },
      "source": [
        "#### Fill out the following configurations"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "PhWoA2uCVaU3"
      },
      "outputs": [],
      "source": [
        "# BigQuery parameters (used for the Generator, Ingester, Logger)\n",
        "BIGQUERY_DATASET_ID = f\"{PROJECT_ID}.movielens_dataset\"  # @param {type:\"string\"} BigQuery dataset ID as `project_id.dataset_id`.\n",
        "BIGQUERY_LOCATION = \"us\"  # @param {type:\"string\"} BigQuery dataset region.\n",
        "BIGQUERY_TABLE_ID = f\"{BIGQUERY_DATASET_ID}.training_dataset\"  # @param {type:\"string\"} BigQuery table ID as `project_id.dataset_id.table_id`."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "fuN1uU27VaU3"
      },
      "source": [
        "#### Set additional configurations\n",
        "\n",
        "You may use the default values below as is."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "895ac243c125"
      },
      "outputs": [],
      "source": [
        "# Dataset parameters\n",
        "RAW_DATA_PATH = \"gs://[your-bucket-name]/raw_data/u.data\"   # @param {type:\"string\"}"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "62bfb9a820f6"
      },
      "outputs": [],
      "source": [
        "# Download the sample data into your RAW_DATA_PATH\n",
        "! gsutil cp \"gs://cloud-samples-data/vertex-ai/community-content/tf_agents_bandits_movie_recommendation_with_kfp_and_vertex_sdk/u.data\" $RAW_DATA_PATH"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "H3530hdGGilo"
      },
      "outputs": [],
      "source": [
        "# Pipeline parameters\n",
        "PIPELINE_NAME = \"movielens-pipeline\"  # Pipeline display name.\n",
        "ENABLE_CACHING = False  # Whether to enable execution caching for the pipeline.\n",
        "PIPELINE_ROOT = f\"{BUCKET_NAME}/pipeline\"  # Root directory for pipeline artifacts.\n",
        "PIPELINE_SPEC_PATH = \"metadata_pipeline.json\"  # Path to pipeline specification file.\n",
        "OUTPUT_COMPONENT_SPEC = \"output-component.yaml\"  # Output component specification file.\n",
        "\n",
        "# BigQuery parameters (used for the Generator, Ingester, Logger)\n",
        "BIGQUERY_TMP_FILE = (\n",
        "    \"tmp.json\"  # Temporary file for storing data to be loaded into BigQuery.\n",
        ")\n",
        "BIGQUERY_MAX_ROWS = 5  # Maximum number of rows of data in BigQuery to ingest.\n",
        "\n",
        "# Dataset parameters\n",
        "TFRECORD_FILE = (\n",
        "    f\"{BUCKET_NAME}/trainer_input_path/*\"  # TFRecord file to be used for training.\n",
        ")\n",
        "\n",
        "# Logger parameters (also used for the Logger hook in the prediction container)\n",
        "LOGGER_PUBSUB_TOPIC = \"logger-pubsub-topic\"  # Pub/Sub topic name for the Logger.\n",
        "LOGGER_CLOUD_FUNCTION = \"logger-cloud-function\"  # Cloud Functions name for the Logger."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "E6ppE7imft-y"
      },
      "source": [
        "## Create the RL pipeline components\n",
        "\n",
        "This section consists of the following steps:\n",
        "1. Create the *Generator* to generate MovieLens simulation data\n",
        "2. Create the *Ingester* to ingest data\n",
        "3. Create the *Trainer* to train the RL policy\n",
        "4. Create the *Deployer* to deploy the trained policy to a Vertex AI endpoint\n",
        "\n",
        "After pipeline construction, create the *Simulator* to send simulated MovieLens prediction requests, create the *Logger* to asynchronously log prediction inputs and results, and create the *Trigger* to trigger re-training.\n",
        "\n",
        "Here's the entire workflow:\n",
        "1. The startup pipeline has the following components: Generator --> Ingester --> Trainer --> Deployer. This pipeline only runs once.\n",
        "2. Then, the Simulator generates prediction requests (e.g. every 5 mins),  and the Logger gets invoked immediately at each prediction request and logs each prediction request asynchronously into BigQuery. The Trigger runs the re-training pipeline (e.g. every 30 mins) with the following components: Ingester --> Trainer --> Deploy.\n",
        "\n",
        "You can find the KFP SDK documentation [here](https://www.kubeflow.org/docs/components/pipelines/sdk/sdk-overview/)."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "dxTLuuWEGilo"
      },
      "source": [
        "### Create the *Generator* to generate MovieLens simulation data\n",
        "\n",
        "Create the Generator component to generate the initial set of training data using a MovieLens simulation environment and a random data-collecting policy. Store the generated data in BigQuery.\n",
        "\n",
        "The Generator source code is [`src/generator/generator_component.py`](src/generator/generator_component.py)."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ay1ztxwIGilo"
      },
      "source": [
        "#### Run unit tests on the Generator component\n",
        "\n",
        "Before running the command, you should update the `RAW_DATA_PATH` in [`src/generator/test_generator_component.py`](src/generator/test_generator_component.py)."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "R9FQacKtGilo"
      },
      "outputs": [],
      "source": [
        "! python3 -m unittest src.generator.test_generator_component"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "1gpYFPPBOWQP"
      },
      "source": [
        "### Create the *Ingester* to ingest data\n",
        "\n",
        "Create the Ingester component to ingest data from BigQuery, package them as `tf.train.Example` objects, and output TFRecord files.\n",
        "\n",
        "Read more about `tf.train.Example` and TFRecord [here](https://www.tensorflow.org/tutorials/load_data/tfrecord).\n",
        "\n",
        "The Ingester component source code is in [`src/ingester/ingester_component.py`](src/ingester/ingester_component.py)."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ZQkLU7wyOWQP"
      },
      "source": [
        "#### Run unit tests on the Ingester component"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Ej4rNnnEOWQP"
      },
      "outputs": [],
      "source": [
        "! python3 -m unittest src.ingester.test_ingester_component"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "KFdSpGAWWkL9"
      },
      "source": [
        "### Create the *Trainer* to train the RL policy\n",
        "\n",
        "Create the Trainer component to train a RL policy on the training dataset, and then submit a remote custom training job to Vertex AI. This component trains a policy using the TF-Agents LinUCB agent on the MovieLens simulation dataset, and saves the trained policy as a SavedModel.\n",
        "\n",
        "The Trainer component source code is in [`src/trainer/trainer_component.py`](src/trainer/trainer_component.py). You use additional Vertex AI platform code in pipeline construction to submit the training code defined in Trainer as a custom training job to Vertex AI. (The additional code is similar to what [`kfp.v2.google.experimental.run_as_aiplatform_custom_job`](https://github.com/kubeflow/pipelines/blob/master/sdk/python/kfp/v2/google/experimental/custom_job.py) does. You can find an example notebook [here](https://github.com/GoogleCloudPlatform/ai-platform-samples/blob/master/ai-platform-unified/notebooks/official/pipelines/google_cloud_pipeline_components_model_train_upload_deploy.ipynb) for how to use that first-party Trainer component.)\n",
        "\n",
        "The Trainer performs off-policy training, where you train a policy on a static set of pre-collected data records containing information including observation, action and reward. For a data record, the policy in training might not output the same action given the observation in that data record.\n",
        "\n",
        "If you're interested in pipeline metrics, read about [KFP Pipeline Metrics](https://www.kubeflow.org/docs/components/pipelines/sdk/pipelines-metrics/) here."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "fh5S-bcCcHn3"
      },
      "outputs": [],
      "source": [
        "# Trainer parameters\n",
        "TRAINING_ARTIFACTS_DIR = (\n",
        "    f\"{BUCKET_NAME}/artifacts\"  # Root directory for training artifacts.\n",
        ")\n",
        "TRAINING_REPLICA_COUNT = 1  # Number of replica to run the custom training job.\n",
        "TRAINING_MACHINE_TYPE = (\n",
        "    \"n1-standard-4\"  # Type of machine to run the custom training job.\n",
        ")\n",
        "TRAINING_ACCELERATOR_TYPE = \"ACCELERATOR_TYPE_UNSPECIFIED\"  # Type of accelerators to run the custom training job.\n",
        "TRAINING_ACCELERATOR_COUNT = 0  # Number of accelerators for the custom training job."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "MH3UOVU8WkL9"
      },
      "source": [
        "#### Run unit tests on the Trainer component"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "CEJ7_ymvWkL9"
      },
      "outputs": [],
      "source": [
        "! python3 -m unittest src.trainer.test_trainer_component"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "5cz7h6V4ibYb"
      },
      "source": [
        "### Create the *Deployer* to deploy the trained policy to a Vertex AI endpoint\n",
        "\n",
        "Use [`google_cloud_pipeline_components.aiplatform`](https://cloud.google.com/vertex-ai/docs/pipelines/build-pipeline#google-cloud-components) components during pipeline construction to:\n",
        "1. Upload the trained policy\n",
        "2. Create a Vertex AI endpoint\n",
        "3. Deploy the uploaded trained policy to the endpoint\n",
        "\n",
        "These 3 components formulate the Deployer. They support flexible configurations; for instance, if you want to set up traffic splitting for the endpoint to run A/B testing, you may pass in your configurations to [google_cloud_pipeline_components.aiplatform.ModelDeployOp](https://google-cloud-pipeline-components.readthedocs.io/en/google-cloud-pipeline-components-0.1.3/google_cloud_pipeline_components.aiplatform.html#google_cloud_pipeline_components.aiplatform.ModelDeployOp)."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "E7dbkbDMcR-m"
      },
      "outputs": [],
      "source": [
        "# Deployer parameters\n",
        "TRAINED_POLICY_DISPLAY_NAME = (\n",
        "    \"movielens-trained-policy\"  # Display name of the uploaded and deployed policy.\n",
        ")\n",
        "TRAFFIC_SPLIT = {\"0\": 100}\n",
        "ENDPOINT_DISPLAY_NAME = \"movielens-endpoint\"  # Display name of the prediction endpoint.\n",
        "ENDPOINT_MACHINE_TYPE = \"n1-standard-4\"  # Type of machine of the prediction endpoint.\n",
        "ENDPOINT_REPLICA_COUNT = 1  # Number of replicas of the prediction endpoint.\n",
        "ENDPOINT_ACCELERATOR_TYPE = \"ACCELERATOR_TYPE_UNSPECIFIED\"  # Type of accelerators to run the custom training job.\n",
        "ENDPOINT_ACCELERATOR_COUNT = 0  # Number of accelerators for the custom training job."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Ldr0yDs6ibYb"
      },
      "source": [
        "### Create a custom prediction container using Cloud Build\n",
        "\n",
        "Before setting up the Deployer, define and build a custom prediction container that serves predictions using the trained policy. The source code, Cloud Build YAML configuration file and Dockerfile are in `src/prediction_container`.\n",
        "\n",
        "This prediction container is the serving container for the deployed, trained policy. See a more detailed guide on building prediction custom containers [here](https://github.com/GoogleCloudPlatform/vertex-ai-samples/tree/master/community-content/tf_agents_bandits_movie_recommendation_with_kfp_and_vertex_sdk/step_by_step_sdk_tf_agents_bandits_movie_recommendation/step_by_step_sdk_tf_agents_bandits_movie_recommendation.ipynb)."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "cKZd3Q_Pcfzo"
      },
      "outputs": [],
      "source": [
        "# Prediction container parameters\n",
        "PREDICTION_CONTAINER = \"prediction-container\"  # Name of the container image.\n",
        "PREDICTION_CONTAINER_DIR = \"src/prediction_container\""
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "jkReLZUAibYb"
      },
      "source": [
        "#### Create a Cloud Build YAML file using Kaniko build\n",
        "\n",
        "Note: For this application, you are recommended to use E2_HIGHCPU_8 or other high resouce machine configurations instead of the standard machine type listed [here](https://cloud.google.com/build/docs/api/reference/rest/v1/projects.builds#Build.MachineType) to prevent out-of-memory errors."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "RUvEWUucibYc"
      },
      "outputs": [],
      "source": [
        "cloudbuild_yaml = \"\"\"steps:\n",
        "- name: \"gcr.io/kaniko-project/executor:latest\"\n",
        "  args: [\"--destination=gcr.io/{PROJECT_ID}/{PREDICTION_CONTAINER}:latest\",\n",
        "         \"--cache=true\",\n",
        "         \"--cache-ttl=99h\"]\n",
        "  env: [\"AIP_STORAGE_URI={ARTIFACTS_DIR}\",\n",
        "        \"PROJECT_ID={PROJECT_ID}\",\n",
        "        \"LOGGER_PUBSUB_TOPIC={LOGGER_PUBSUB_TOPIC}\"]\n",
        "options:\n",
        "  machineType: \"E2_HIGHCPU_8\"\n",
        "\"\"\".format(\n",
        "    PROJECT_ID=PROJECT_ID,\n",
        "    PREDICTION_CONTAINER=PREDICTION_CONTAINER,\n",
        "    ARTIFACTS_DIR=TRAINING_ARTIFACTS_DIR,\n",
        "    LOGGER_PUBSUB_TOPIC=LOGGER_PUBSUB_TOPIC,\n",
        ")\n",
        "\n",
        "with open(f\"{PREDICTION_CONTAINER_DIR}/cloudbuild.yaml\", \"w\") as fp:\n",
        "    fp.write(cloudbuild_yaml)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "kByjVm5yibYc"
      },
      "source": [
        "#### Run unit tests on the prediction code"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "tzLU1V6fibYc"
      },
      "outputs": [],
      "source": [
        "! python3 -m unittest src.prediction_container.test_main"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "RelRBSFvibYc"
      },
      "source": [
        "#### Build custom prediction container"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "9uHbODeXibYd"
      },
      "outputs": [],
      "source": [
        "! gcloud builds submit --config $PREDICTION_CONTAINER_DIR/cloudbuild.yaml $PREDICTION_CONTAINER_DIR"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "nW154IyqGilq"
      },
      "source": [
        "## Author and run the RL pipeline\n",
        "\n",
        "You author the pipeline using custom KFP components built from the previous section, and [create a pipeline run](https://cloud.google.com/vertex-ai/docs/pipelines/run-pipeline#kubeflow-pipelines-sdk) using Vertex Pipelines. You can read more about whether to enable execution caching [here](https://cloud.google.com/vertex-ai/docs/pipelines/build-pipeline#caching). You can also specifically configure the worker pool spec for training if for instance you want to train at scale and/or at a higher speed; you can adjust the replica count, machine type, accelerator type and count, and many other specifications.\n",
        "\n",
        "Here, you build a \"startup\" pipeline that generates randomly sampled training data (with the Generator) as the first step. This pipeline runs only once."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "cvXJzhSSGilq"
      },
      "outputs": [],
      "source": [
        "from google_cloud_pipeline_components.experimental.custom_job import utils\n",
        "from kfp.components import load_component_from_url\n",
        "\n",
        "generate_op = load_component_from_url(\n",
        "    \"https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/62a2a7611499490b4b04d731d48a7ba87c2d636f/community-content/tf_agents_bandits_movie_recommendation_with_kfp_and_vertex_sdk/mlops_pipeline_tf_agents_bandits_movie_recommendation/src/generator/component.yaml\"\n",
        ")\n",
        "ingest_op = load_component_from_url(\n",
        "    \"https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/62a2a7611499490b4b04d731d48a7ba87c2d636f/community-content/tf_agents_bandits_movie_recommendation_with_kfp_and_vertex_sdk/mlops_pipeline_tf_agents_bandits_movie_recommendation/src/ingester/component.yaml\"\n",
        ")\n",
        "train_op = load_component_from_url(\n",
        "    \"https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/62a2a7611499490b4b04d731d48a7ba87c2d636f/community-content/tf_agents_bandits_movie_recommendation_with_kfp_and_vertex_sdk/mlops_pipeline_tf_agents_bandits_movie_recommendation/src/trainer/component.yaml\"\n",
        ")\n",
        "\n",
        "\n",
        "@dsl.pipeline(pipeline_root=PIPELINE_ROOT, name=f\"{PIPELINE_NAME}-startup\")\n",
        "def pipeline(\n",
        "    # Pipeline configs\n",
        "    project_id: str,\n",
        "    raw_data_path: str,\n",
        "    training_artifacts_dir: str,\n",
        "    # BigQuery configs\n",
        "    bigquery_dataset_id: str,\n",
        "    bigquery_location: str,\n",
        "    bigquery_table_id: str,\n",
        "    bigquery_max_rows: int = 10000,\n",
        "    # TF-Agents RL configs\n",
        "    batch_size: int = 8,\n",
        "    rank_k: int = 20,\n",
        "    num_actions: int = 20,\n",
        "    driver_steps: int = 3,\n",
        "    num_epochs: int = 5,\n",
        "    tikhonov_weight: float = 0.01,\n",
        "    agent_alpha: float = 10,\n",
        ") -> None:\n",
        "    \"\"\"Authors a RL pipeline for MovieLens movie recommendation system.\n",
        "\n",
        "    Integrates the Generator, Ingester, Trainer and Deployer components. This\n",
        "    pipeline generates initial training data with a random policy and runs once\n",
        "    as the initiation of the system.\n",
        "\n",
        "    Args:\n",
        "      project_id: GCP project ID. This is required because otherwise the BigQuery\n",
        "        client will use the ID of the tenant GCP project created as a result of\n",
        "        KFP, which doesn't have proper access to BigQuery.\n",
        "      raw_data_path: Path to MovieLens 100K's \"u.data\" file.\n",
        "      training_artifacts_dir: Path to store the Trainer artifacts (trained policy).\n",
        "\n",
        "      bigquery_dataset: A string of the BigQuery dataset ID in the format of\n",
        "        \"project.dataset\".\n",
        "      bigquery_location: A string of the BigQuery dataset location.\n",
        "      bigquery_table_id: A string of the BigQuery table ID in the format of\n",
        "        \"project.dataset.table\".\n",
        "      bigquery_max_rows: Optional; maximum number of rows to ingest.\n",
        "\n",
        "      batch_size: Optional; batch size of environment generated quantities eg.\n",
        "        rewards.\n",
        "      rank_k: Optional; rank for matrix factorization in the MovieLens environment;\n",
        "        also the observation dimension.\n",
        "      num_actions: Optional; number of actions (movie items) to choose from.\n",
        "      driver_steps: Optional; number of steps to run per batch.\n",
        "      num_epochs: Optional; number of training epochs.\n",
        "      tikhonov_weight: Optional; LinUCB Tikhonov regularization weight of the\n",
        "        Trainer.\n",
        "      agent_alpha: Optional; LinUCB exploration parameter that multiplies the\n",
        "        confidence intervals of the Trainer.\n",
        "    \"\"\"\n",
        "    # Run the Generator component.\n",
        "    generate_task = generate_op(\n",
        "        project_id=project_id,\n",
        "        raw_data_path=raw_data_path,\n",
        "        batch_size=batch_size,\n",
        "        rank_k=rank_k,\n",
        "        num_actions=num_actions,\n",
        "        driver_steps=driver_steps,\n",
        "        bigquery_tmp_file=BIGQUERY_TMP_FILE,\n",
        "        bigquery_dataset_id=bigquery_dataset_id,\n",
        "        bigquery_location=bigquery_location,\n",
        "        bigquery_table_id=bigquery_table_id,\n",
        "    )\n",
        "    \n",
        "    # Run the Ingester component.\n",
        "    ingest_task = ingest_op(\n",
        "        project_id=project_id,\n",
        "        bigquery_table_id=generate_task.outputs[\"bigquery_table_id\"],\n",
        "        bigquery_max_rows=bigquery_max_rows,\n",
        "        tfrecord_file=TFRECORD_FILE,\n",
        "    )\n",
        "\n",
        "    # Run the Trainer component and submit custom job to Vertex AI.\n",
        "    # Convert the train_op component into a Vertex AI Custom Job pre-built component\n",
        "    custom_job_training_op = utils.create_custom_training_job_op_from_component(\n",
        "        component_spec=train_op,\n",
        "        replica_count=TRAINING_REPLICA_COUNT,\n",
        "        machine_type=TRAINING_MACHINE_TYPE,\n",
        "        accelerator_type=TRAINING_ACCELERATOR_TYPE,\n",
        "        accelerator_count=TRAINING_ACCELERATOR_COUNT,\n",
        "    )\n",
        "\n",
        "    train_task = custom_job_training_op(\n",
        "        training_artifacts_dir=training_artifacts_dir,\n",
        "        tfrecord_file=ingest_task.outputs[\"tfrecord_file\"],\n",
        "        num_epochs=num_epochs,\n",
        "        rank_k=rank_k,\n",
        "        num_actions=num_actions,\n",
        "        tikhonov_weight=tikhonov_weight,\n",
        "        agent_alpha=agent_alpha,\n",
        "        project=PROJECT_ID,\n",
        "        location=REGION,\n",
        "    )\n",
        "\n",
        "    # Run the Deployer components.\n",
        "    # Upload the trained policy as a model.\n",
        "    model_upload_op = gcc_aip.ModelUploadOp(\n",
        "        project=project_id,\n",
        "        display_name=TRAINED_POLICY_DISPLAY_NAME,\n",
        "        artifact_uri=train_task.outputs[\"training_artifacts_dir\"],\n",
        "        serving_container_image_uri=f\"gcr.io/{PROJECT_ID}/{PREDICTION_CONTAINER}:latest\",\n",
        "    )\n",
        "    # Create a Vertex AI endpoint. (This operation can occur in parallel with\n",
        "    # the Generator, Ingester, Trainer components.)\n",
        "    endpoint_create_op = gcc_aip.EndpointCreateOp(\n",
        "        project=project_id, display_name=ENDPOINT_DISPLAY_NAME\n",
        "    )\n",
        "    # Deploy the uploaded, trained policy to the created endpoint. (This operation\n",
        "    # has to occur after both model uploading and endpoint creation complete.)\n",
        "    gcc_aip.ModelDeployOp(\n",
        "        endpoint=endpoint_create_op.outputs[\"endpoint\"],\n",
        "        model=model_upload_op.outputs[\"model\"],\n",
        "        deployed_model_display_name=TRAINED_POLICY_DISPLAY_NAME,\n",
        "        traffic_split=TRAFFIC_SPLIT,\n",
        "        dedicated_resources_machine_type=ENDPOINT_MACHINE_TYPE,\n",
        "        dedicated_resources_accelerator_type=ENDPOINT_ACCELERATOR_TYPE,\n",
        "        dedicated_resources_accelerator_count=ENDPOINT_ACCELERATOR_COUNT,\n",
        "        dedicated_resources_min_replica_count=ENDPOINT_REPLICA_COUNT,\n",
        "    )"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "icYK0WoRGilr"
      },
      "outputs": [],
      "source": [
        "# Compile the authored pipeline.\n",
        "compiler.Compiler().compile(pipeline_func=pipeline, package_path=PIPELINE_SPEC_PATH)\n",
        "\n",
        "# Create a pipeline run job.\n",
        "job = aiplatform.PipelineJob(\n",
        "    display_name=f\"{PIPELINE_NAME}-startup\",\n",
        "    template_path=PIPELINE_SPEC_PATH,\n",
        "    pipeline_root=PIPELINE_ROOT,\n",
        "    parameter_values={\n",
        "        # Pipeline configs\n",
        "        \"project_id\": PROJECT_ID,\n",
        "        \"raw_data_path\": RAW_DATA_PATH,\n",
        "        \"training_artifacts_dir\": TRAINING_ARTIFACTS_DIR,\n",
        "        # BigQuery configs\n",
        "        \"bigquery_dataset_id\": BIGQUERY_DATASET_ID,\n",
        "        \"bigquery_location\": BIGQUERY_LOCATION,\n",
        "        \"bigquery_table_id\": BIGQUERY_TABLE_ID,\n",
        "    },\n",
        "    enable_caching=ENABLE_CACHING,\n",
        ")\n",
        "\n",
        "job.run()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "_YDDhAx5i1UL"
      },
      "source": [
        "## Create the *Simulator* to send simulated MovieLens prediction requests\n",
        "\n",
        "Create the Simulator to [obtain observations](https://github.com/tensorflow/agents/blob/v0.8.0/tf_agents/bandits/environments/movielens_py_environment.py#L118-L125) from the MovieLens simulation environment, formats them, and sends prediction requests to the Vertex AI endpoint.\n",
        "\n",
        "The workflow is: Cloud Scheduler --> Pub/Sub --> Cloud Functions --> Endpoint\n",
        "\n",
        "In production, this Simulator logic can be modified to that of gathering real-world input features as observations, getting prediction results from the endpoint and communicating those results to real-world users.\n",
        "\n",
        "The Simulator source code is [`src/simulator/main.py`](src/simulator/main.py)."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Sxz6T0yjcoX2"
      },
      "outputs": [],
      "source": [
        "# Simulator parameters\n",
        "SIMULATOR_PUBSUB_TOPIC = (\n",
        "    \"simulator-pubsub-topic\"  # Pub/Sub topic name for the Simulator.\n",
        ")\n",
        "SIMULATOR_CLOUD_FUNCTION = (\n",
        "    \"simulator-cloud-function\"  # Cloud Functions name for the Simulator.\n",
        ")\n",
        "SIMULATOR_SCHEDULER_JOB = (\n",
        "    \"simulator-scheduler-job\"  # Cloud Scheduler cron job name for the Simulator.\n",
        ")\n",
        "SIMULATOR_SCHEDULE = \"*/5 * * * *\"  # Cloud Scheduler cron job schedule for the Simulator. Eg. \"*/5 * * * *\" means every 5 mins.\n",
        "SIMULATOR_SCHEDULER_MESSAGE = (\n",
        "    \"simulator-message\"  # Cloud Scheduler message for the Simulator.\n",
        ")\n",
        "# TF-Agents RL configs\n",
        "BATCH_SIZE = 8\n",
        "RANK_K = 20\n",
        "NUM_ACTIONS = 20"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "6JkcpQmpi1UL"
      },
      "source": [
        "### Run unit tests on the Simulator"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "GoNH1VS_i1UL"
      },
      "outputs": [],
      "source": [
        "! python3 -m unittest src.simulator.test_main"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "0g8bl_pCi1UL"
      },
      "source": [
        "### Create a Pub/Sub topic\n",
        "\n",
        "- Read more about creating Pub/Sub topics [here](https://cloud.google.com/functions/docs/tutorials/pubsub)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "5apj2cEri1UL"
      },
      "outputs": [],
      "source": [
        "! gcloud pubsub topics create $SIMULATOR_PUBSUB_TOPIC"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "jA8gMvqXi1UM"
      },
      "source": [
        "### Set up a recurrent Cloud Scheduler job for the Pub/Sub topic\n",
        "\n",
        "- Read more about possible ways to create cron jobs [here](https://cloud.google.com/scheduler/docs/creating#gcloud).\n",
        "- Read about the cron job schedule format [here](https://man7.org/linux/man-pages/man5/crontab.5.html)."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "snTyjFkDi1UM"
      },
      "outputs": [],
      "source": [
        "scheduler_job_args = \" \".join(\n",
        "    [\n",
        "        SIMULATOR_SCHEDULER_JOB,\n",
        "        f\"--schedule='{SIMULATOR_SCHEDULE}'\",\n",
        "        f\"--topic={SIMULATOR_PUBSUB_TOPIC}\",\n",
        "        f\"--message-body={SIMULATOR_SCHEDULER_MESSAGE}\",\n",
        "    ]\n",
        ")\n",
        "\n",
        "! echo $scheduler_job_args"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "4pY_Gs_Di1UM"
      },
      "outputs": [],
      "source": [
        "! gcloud scheduler jobs create pubsub $scheduler_job_args"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "HjyV2Arei1UM"
      },
      "source": [
        "### Define the *Simulator* logic in a Cloud Function to be triggered periodically, and deploy this Function\n",
        "\n",
        "- Specify dependencies of the Function in [`src/simulator/requirements.txt`](src/simulator/requirements.txt).\n",
        "- Read more about the available configurable arguments for deploying a Function [here](https://cloud.google.com/sdk/gcloud/reference/functions/deploy). For instance, based on the complexity of your Function, you may want to adjust its memory and timeout.\n",
        "- Note that the environment variables in `ENV_VARS` should be comma-separated; there should not be additional spaces, or other characters in between. Read more about setting/updating/deleting environment variables [here](https://cloud.google.com/functions/docs/env-var).\n",
        "- Read more about sending predictions to Vertex endpoints [here](https://cloud.google.com/vertex-ai/docs/predictions/online-predictions-custom-models)."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "0LJ2C_pdibYg"
      },
      "outputs": [],
      "source": [
        "endpoints = ! gcloud ai endpoints list \\\n",
        "    --region=$REGION \\\n",
        "    --filter=display_name=$ENDPOINT_DISPLAY_NAME\n",
        "print(\"\\n\".join(endpoints), \"\\n\")\n",
        "\n",
        "ENDPOINT_ID = endpoints[2].split(\" \")[0]\n",
        "print(f\"ENDPOINT_ID={ENDPOINT_ID}\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "V4xwXBBgi1UM"
      },
      "outputs": [],
      "source": [
        "ENV_VARS = \",\".join(\n",
        "    [\n",
        "        f\"PROJECT_ID={PROJECT_ID}\",\n",
        "        f\"REGION={REGION}\",\n",
        "        f\"ENDPOINT_ID={ENDPOINT_ID}\",\n",
        "        f\"RAW_DATA_PATH={RAW_DATA_PATH}\",\n",
        "        f\"BATCH_SIZE={BATCH_SIZE}\",\n",
        "        f\"RANK_K={RANK_K}\",\n",
        "        f\"NUM_ACTIONS={NUM_ACTIONS}\",\n",
        "    ]\n",
        ")\n",
        "\n",
        "! echo $ENV_VARS"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "0yiHxUMBi1UM"
      },
      "outputs": [],
      "source": [
        "! gcloud functions deploy $SIMULATOR_CLOUD_FUNCTION \\\n",
        "    --region=$REGION \\\n",
        "    --trigger-topic=$SIMULATOR_PUBSUB_TOPIC \\\n",
        "    --runtime=python37 \\\n",
        "    --memory=512MB \\\n",
        "    --timeout=200s \\\n",
        "    --source=src/simulator \\\n",
        "    --entry-point=simulate \\\n",
        "    --stage-bucket=$BUCKET_NAME \\\n",
        "    --update-env-vars=$ENV_VARS"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "213JWEcLxAhN"
      },
      "source": [
        "## Create the *Logger* to asynchronously log prediction inputs and results\n",
        "\n",
        "Create the Logger to get environment feedback as rewards from the MovieLens simulation environment based on prediction observations and predicted actions, formulate trajectory data, and store said data back to BigQuery. The Logger closes the RL feedback loop from prediction to training data, and allows re-training of the policy on new training data.\n",
        "\n",
        "The Logger is triggered by a hook in the prediction code. At each prediction request, the prediction code messages a Pub/Sub topic, which triggers the Logger code.\n",
        "\n",
        "The workflow is: prediction container code (at prediction request) --> Pub/Sub --> Cloud Functions (logging predictions back to BigQuery)\n",
        "\n",
        "In production, this Logger logic can be modified to that of gathering real-world feedback (rewards) based on observations and predicted actions.\n",
        "\n",
        "The Logger source code is [`src/logger/main.py`](src/logger/main.py)."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "gIuPCKRjxAhN"
      },
      "source": [
        "### Run unit tests on the Logger"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "-eVdF88gxAhN"
      },
      "outputs": [],
      "source": [
        "! python3 -m unittest src.logger.test_main"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "hs56EW17xAhO"
      },
      "source": [
        "### Create a Pub/Sub topic\n",
        "\n",
        "- Read more about creating Pub/Sub topics [here](https://cloud.google.com/functions/docs/tutorials/pubsub)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ydoCTizJxAhO"
      },
      "outputs": [],
      "source": [
        "! gcloud pubsub topics create $LOGGER_PUBSUB_TOPIC"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "FnlsMxfjxAhO"
      },
      "source": [
        "### Define the *Logger* logic in a Cloud Function to be triggered by a Pub/Sub topic, which is triggered by the prediction code at each prediction request.\n",
        "\n",
        "- Specify dependencies of the Function in [`src/logger/requirements.txt`](src/logger/requirements.txt).\n",
        "- Read more about the available configurable arguments for deploying a Function [here](https://cloud.google.com/sdk/gcloud/reference/functions/deploy). For instance, based on the complexity of your Function, you may want to adjust its memory and timeout.\n",
        "- Note that the environment variables in `ENV_VARS` should be comma-separated; there should not be additional spaces, or other characters in between. Read more about setting/updating/deleting environment variables [here](https://cloud.google.com/functions/docs/env-var)."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "DwrukBPHxAhO"
      },
      "outputs": [],
      "source": [
        "ENV_VARS = \",\".join(\n",
        "    [\n",
        "        f\"PROJECT_ID={PROJECT_ID}\",\n",
        "        f\"RAW_DATA_PATH={RAW_DATA_PATH}\",\n",
        "        f\"BATCH_SIZE={BATCH_SIZE}\",\n",
        "        f\"RANK_K={RANK_K}\",\n",
        "        f\"NUM_ACTIONS={NUM_ACTIONS}\",\n",
        "        f\"BIGQUERY_TMP_FILE={BIGQUERY_TMP_FILE}\",\n",
        "        f\"BIGQUERY_DATASET_ID={BIGQUERY_DATASET_ID}\",\n",
        "        f\"BIGQUERY_LOCATION={BIGQUERY_LOCATION}\",\n",
        "        f\"BIGQUERY_TABLE_ID={BIGQUERY_TABLE_ID}\",\n",
        "    ]\n",
        ")\n",
        "\n",
        "! echo $ENV_VARS"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "OykKRkScxAhO"
      },
      "outputs": [],
      "source": [
        "! gcloud functions deploy $LOGGER_CLOUD_FUNCTION \\\n",
        "    --region=$REGION \\\n",
        "    --trigger-topic=$LOGGER_PUBSUB_TOPIC \\\n",
        "    --runtime=python37 \\\n",
        "    --memory=512MB \\\n",
        "    --timeout=200s \\\n",
        "    --source=src/logger \\\n",
        "    --entry-point=log \\\n",
        "    --stage-bucket=$BUCKET_NAME \\\n",
        "    --update-env-vars=$ENV_VARS"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "n0YSz3xtJcci"
      },
      "source": [
        "## Create the *Trigger* to trigger re-training\n",
        "\n",
        "Create the Trigger to recurrently re-run the pipeline to re-train the policy on new training data, using `kfp.v2.google.client.AIPlatformClient.create_schedule_from_job_spec`. You create a pipeline for orchestration on Vertex Pipelines, and a Cloud Scheduler job that recurrently triggers the pipeline. The method also automatically creates a Cloud Function that acts as an intermediary between the Scheduler and Pipelines. You can find the source code [here](https://github.com/kubeflow/pipelines/blob/v1.7.0-alpha.3/sdk/python/kfp/v2/google/client/client.py#L347-L391).\n",
        "\n",
        "When the Simulator sends prediction requests to the endpoint, the Logger is triggered by the hook in the prediction code to log prediction results to BigQuery, as new training data. As this pipeline has a recurrent schedule, it utlizes the new training data in training a new policy, therefore closing the feedback loop. Theoretically speaking, if you set the pipeline scheduler to be infinitely frequent, then you would be approaching real-time, continuous training."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "-YmLQ-ykJcci"
      },
      "outputs": [],
      "source": [
        "TRIGGER_SCHEDULE = \"*/30 * * * *\"  # Schedule to trigger the pipeline. Eg. \"*/30 * * * *\" means every 30 mins."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ay1x-rgIJcci"
      },
      "outputs": [],
      "source": [
        "ingest_op = load_component_from_url(\n",
        "    \"https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/62a2a7611499490b4b04d731d48a7ba87c2d636f/community-content/tf_agents_bandits_movie_recommendation_with_kfp_and_vertex_sdk/mlops_pipeline_tf_agents_bandits_movie_recommendation/src/ingester/component.yaml\"\n",
        ")\n",
        "train_op = load_component_from_url(\n",
        "    \"https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/62a2a7611499490b4b04d731d48a7ba87c2d636f/community-content/tf_agents_bandits_movie_recommendation_with_kfp_and_vertex_sdk/mlops_pipeline_tf_agents_bandits_movie_recommendation/src/trainer/component.yaml\"\n",
        ")\n",
        "\n",
        "\n",
        "@dsl.pipeline(pipeline_root=PIPELINE_ROOT, name=f\"{PIPELINE_NAME}-retraining\")\n",
        "def pipeline(\n",
        "    # Pipeline configs\n",
        "    project_id: str,\n",
        "    training_artifacts_dir: str,\n",
        "    # BigQuery configs\n",
        "    bigquery_table_id: str,\n",
        "    bigquery_max_rows: int = 10000,\n",
        "    # TF-Agents RL configs\n",
        "    rank_k: int = 20,\n",
        "    num_actions: int = 20,\n",
        "    num_epochs: int = 5,\n",
        "    tikhonov_weight: float = 0.01,\n",
        "    agent_alpha: float = 10,\n",
        ") -> None:\n",
        "    \"\"\"Authors a re-training pipeline for MovieLens movie recommendation system.\n",
        "\n",
        "    Integrates the Ingester, Trainer and Deployer components.\n",
        "\n",
        "    Args:\n",
        "      project_id: GCP project ID. This is required because otherwise the BigQuery\n",
        "        client will use the ID of the tenant GCP project created as a result of\n",
        "        KFP, which doesn't have proper access to BigQuery.\n",
        "      training_artifacts_dir: Path to store the Trainer artifacts (trained policy).\n",
        "\n",
        "      bigquery_table_id: A string of the BigQuery table ID in the format of\n",
        "        \"project.dataset.table\".\n",
        "      bigquery_max_rows: Optional; maximum number of rows to ingest.\n",
        "\n",
        "      rank_k: Optional; rank for matrix factorization in the MovieLens environment;\n",
        "        also the observation dimension.\n",
        "      num_actions: Optional; number of actions (movie items) to choose from.\n",
        "      num_epochs: Optional; number of training epochs.\n",
        "      tikhonov_weight: Optional; LinUCB Tikhonov regularization weight of the\n",
        "        Trainer.\n",
        "      agent_alpha: Optional; LinUCB exploration parameter that multiplies the\n",
        "        confidence intervals of the Trainer.\n",
        "    \"\"\"\n",
        "    # Run the Ingester component.\n",
        "    ingest_task = ingest_op(\n",
        "        project_id=project_id,\n",
        "        bigquery_table_id=bigquery_table_id,\n",
        "        bigquery_max_rows=bigquery_max_rows,\n",
        "        tfrecord_file=TFRECORD_FILE,\n",
        "    )\n",
        "\n",
        "    # Run the Trainer component and submit custom job to Vertex AI.\n",
        "    # Convert the train_op component into a Vertex AI Custom Job pre-built component\n",
        "    custom_job_training_op = utils.create_custom_training_job_op_from_component(\n",
        "        component_spec=train_op,\n",
        "        replica_count=TRAINING_REPLICA_COUNT,\n",
        "        machine_type=TRAINING_MACHINE_TYPE,\n",
        "        accelerator_type=TRAINING_ACCELERATOR_TYPE,\n",
        "        accelerator_count=TRAINING_ACCELERATOR_COUNT,\n",
        "    )\n",
        "\n",
        "    train_task = custom_job_training_op(\n",
        "        training_artifacts_dir=training_artifacts_dir,\n",
        "        tfrecord_file=ingest_task.outputs[\"tfrecord_file\"],\n",
        "        num_epochs=num_epochs,\n",
        "        rank_k=rank_k,\n",
        "        num_actions=num_actions,\n",
        "        tikhonov_weight=tikhonov_weight,\n",
        "        agent_alpha=agent_alpha,\n",
        "        project=PROJECT_ID,\n",
        "        location=REGION,\n",
        "    )\n",
        "\n",
        "    # Run the Deployer components.\n",
        "    # Upload the trained policy as a model.\n",
        "    model_upload_op = gcc_aip.ModelUploadOp(\n",
        "        project=project_id,\n",
        "        display_name=TRAINED_POLICY_DISPLAY_NAME,\n",
        "        artifact_uri=train_task.outputs[\"training_artifacts_dir\"],\n",
        "        serving_container_image_uri=f\"gcr.io/{PROJECT_ID}/{PREDICTION_CONTAINER}:latest\",\n",
        "    )\n",
        "    # Create a Vertex AI endpoint. (This operation can occur in parallel with\n",
        "    # the Generator, Ingester, Trainer components.)\n",
        "    endpoint_create_op = gcc_aip.EndpointCreateOp(\n",
        "        project=project_id, display_name=ENDPOINT_DISPLAY_NAME\n",
        "    )\n",
        "    # Deploy the uploaded, trained policy to the created endpoint. (This operation\n",
        "    # has to occur after both model uploading and endpoint creation complete.)\n",
        "    gcc_aip.ModelDeployOp(\n",
        "        endpoint=endpoint_create_op.outputs[\"endpoint\"],\n",
        "        model=model_upload_op.outputs[\"model\"],\n",
        "        deployed_model_display_name=TRAINED_POLICY_DISPLAY_NAME,\n",
        "        dedicated_resources_machine_type=ENDPOINT_MACHINE_TYPE,\n",
        "        dedicated_resources_accelerator_type=ENDPOINT_ACCELERATOR_TYPE,\n",
        "        dedicated_resources_accelerator_count=ENDPOINT_ACCELERATOR_COUNT,\n",
        "        dedicated_resources_min_replica_count=ENDPOINT_REPLICA_COUNT,\n",
        "    )"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "9yPjcm75Jcci"
      },
      "outputs": [],
      "source": [
        "# Compile the authored pipeline.\n",
        "compiler.Compiler().compile(pipeline_func=pipeline, package_path=PIPELINE_SPEC_PATH)\n",
        "\n",
        "# Createa Vertex AI client.\n",
        "api_client = AIPlatformClient(project_id=PROJECT_ID, region=REGION)\n",
        "\n",
        "# Schedule a recurring pipeline.\n",
        "response = api_client.create_schedule_from_job_spec(\n",
        "    job_spec_path=PIPELINE_SPEC_PATH,\n",
        "    schedule=TRIGGER_SCHEDULE,\n",
        "    parameter_values={\n",
        "        # Pipeline configs\n",
        "        \"project_id\": PROJECT_ID,\n",
        "        \"training_artifacts_dir\": TRAINING_ARTIFACTS_DIR,\n",
        "        # BigQuery config\n",
        "        \"bigquery_table_id\": BIGQUERY_TABLE_ID,\n",
        "    },\n",
        ")\n",
        "response[\"name\"]"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "TpV-iwP9qw9c"
      },
      "source": [
        "## Cleaning up\n",
        "\n",
        "To clean up all Google Cloud resources used in this project, you can [delete the Google Cloud\n",
        "project](https://cloud.google.com/resource-manager/docs/creating-managing-projects#shutting_down_projects) you used for the tutorial.\n",
        "\n",
        "Otherwise, you can delete the individual resources you created in this tutorial (you also need to clean up other resources that are difficult to delete here, such as the all/partial of data in BigQuery, the recurring pipeline and its Scheduler job, the uploaded policy/model, etc.):"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "sx_vKniMq9ZX"
      },
      "outputs": [],
      "source": [
        "# Delete endpoint resource.\n",
        "! gcloud ai endpoints delete $ENDPOINT_ID --quiet --region $REGION\n",
        "\n",
        "# Delete Pub/Sub topics.\n",
        "! gcloud pubsub topics delete $SIMULATOR_PUBSUB_TOPIC --quiet\n",
        "! gcloud pubsub topics delete $LOGGER_PUBSUB_TOPIC --quiet\n",
        "\n",
        "# Delete Cloud Functions.\n",
        "! gcloud functions delete $SIMULATOR_CLOUD_FUNCTION --quiet\n",
        "! gcloud functions delete $LOGGER_CLOUD_FUNCTION --quiet\n",
        "\n",
        "# Delete Scheduler job.\n",
        "! gcloud scheduler jobs delete $SIMULATOR_SCHEDULER_JOB --quiet\n",
        "\n",
        "# Delete Cloud Storage objects that were created.\n",
        "! gsutil -m rm -r $PIPELINE_ROOT\n",
        "! gsutil -m rm -r $TRAINING_ARTIFACTS_DIR"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "collapsed_sections": [],
      "name": "mlops_pipeline_tf_agents_bandits_movie_recommendation.ipynb",
      "toc_visible": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
