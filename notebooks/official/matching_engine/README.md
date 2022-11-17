## Vertex-AI: Matching Engine Notebooks
  - [Create Vertex AI Matching Engine index](#sdk_matching_engine_for_indexing)
  - [Introduction to builtin Swivel embedding algorithm](#intro-swivel)
  - [Introduction to builtin Two-towers embedding algorithm](#two-tower-model-introduction)
---
---


<a id="sdk_matching_engine_for_indexing"></a>[Create Vertex AI Matching Engine index](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/matching_engine/sdk_matching_engine_for_indexing.ipynb)

Learn how to create Approximate Nearest Neighbor (ANN) Index, query against indexes, and validate the performance of the index.

The steps performed include:

* Create ANN Index and Brute Force Index
* Create an IndexEndpoint with VPC Network
* Deploy ANN Index and Brute Force Index
* Perform online query
* Compute recall

    <details>
    <summary>Example code snippet from the Notebook:</summary>
    
    * Create an IndexEndpoint with VPC Network
        ```python
        # [START aiplatform_sdk_matching_engine_for_indexing]
        VPC_NETWORK = "[your-network-name]"
        VPC_NETWORK_FULL = "projects/{}/global/networks/{}".format(PROJECT_NUMBER, VPC_NETWORK)
        my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
            display_name="index_endpoint_for_demo",
            description="index endpoint description",
            network=VPC_NETWORK_FULL,
        )
        # [END aiplatform_sdk_matching_engine_for_indexing]
        ```
        [:notebook: sdk_matching_engine_for_indexing.ipynb](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/matching_engine/sdk_matching_engine_for_indexing.ipynb)
    </details>
---


<a id="intro-swivel"></a>[Introduction to builtin Swivel embedding algorithm](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/matching_engine/intro-swivel.ipynb)

Learn how to train custom embeddings using Vertex AI Pipelines and deploy the model for serving.

The steps performed include:

1. **Setup**: Importing the required libraries and setting your global variables.
2. **Configure parameters**: Setting the appropriate parameter values for the pipeline job.
3. **Train on Vertex AI Pipelines**: Create a Swivel job to Vertex Pipelines using pipeline template.
4. **Deploy on Vertex AI Prediction**: Importing and deploying the trained model to a callable endpoint.
5. **Predict**: Calling the deployed endpoint using online prediction.
6. **Cleaning up**: Deleting resources created by this tutorial.

    <details>
    <summary>Example code snippet from the Notebook:</summary>
    
    * Deploy the embedding model for online serving
        ```python
        # [START aiplatform_intro-swivel]
        ENDPOINT_NAME = "swivel_embedding"  # <---CHANGE THIS (OPTIONAL)
        MODEL_VERSION_NAME = "movie-tf2-cpu-2.4"  # <---CHANGE THIS (OPTIONAL)

        aiplatform.init(project=PROJECT_ID, location=REGION)

        # Create a model endpoint
        endpoint = aiplatform.Endpoint.create(display_name=ENDPOINT_NAME)

        # Upload the trained model to Model resource
        model = aiplatform.Model.upload(
            display_name=MODEL_VERSION_NAME,
            artifact_uri=SAVEDMODEL_DIR,
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-4:latest",
        )

        # Deploy the Model to the Endpoint
        model.deploy(
            endpoint=endpoint,
            machine_type="n1-standard-2",
        )
        # [END aiplatform_intro-swivel]
        ```
        [:notebook: intro-swivel.ipynb](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/matching_engine/intro-swivel.ipynb)
    </details>
---

<a id="two-tower-model-introduction"></a>[Introduction to builtin Two-towers embedding algorithm](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/matching_engine/two-tower-model-introduction.ipynb)

Learn how to run the two-tower model.

The steps performed include:
1. **Setup**: Importing the required libraries and setting your global variables.
2. **Configure parameters**: Setting the appropriate parameter values for the training job.
3. **Train on Vertex AI Training**: Submitting a training job.
4. **Deploy on Vertex AI Prediction**: Importing and deploying the trained model to a callable endpoint.
5. **Predict**: Calling the deployed endpoint using online or batch prediction.
6. **Hyperparameter tuning**: Running a hyperparameter tuning job.
7. **Cleaning up**: Deleting resources created by this tutorial.

    <details>
    <summary>Example code snippet from the Notebook:</summary>
    
    * Deploy the model
        ```python
        # [START aiplatform_two-tower-model-introduction]

        # Create a model endpoint
        endpoint = aiplatform.Endpoint.create(display_name=DATASET_NAME)

        # Deploy model to the endpoint
        model.deploy(
            endpoint=endpoint,
            machine_type="n1-standard-4",
            traffic_split={"0": 100},
            deployed_model_display_name=DISPLAY_NAME,
        )

        # [END aiplatform_two-tower-model-introduction]
        ```
        [:notebook: two-tower-model-introduction.ipynb](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/matching_engine/two-tower-model-introduction.ipynb)
    </details>
