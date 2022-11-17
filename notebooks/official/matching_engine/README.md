## Vertex-AI: Matching Engine Notebook

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
