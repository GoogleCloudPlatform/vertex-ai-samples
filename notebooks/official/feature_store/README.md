
[Streaming ingestion SDK](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/feature_store/feature_store_streaming_ingestion_sdk.ipynb)

```
Learn how to ingest features from a `Pandas DataFrame` into your Vertex AI Feature Store using `write_feature_values` method from the Vertex AI SDK.

The steps performed include:

- Create `Feature Store`
- Create new `Entity Type` for your `Feature Store`
- Ingest feature values from `Pandas DataFrame` into `Feature Store`'s `Entity Types`.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Feature Store](https://cloud.google.com/vertex-ai/docs/featurestore).


[Online feature serving and fetching of BigQuery data with Vertex AI Feature Store](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/feature_store/online_feature_serving_and_fetching_bigquery_data_with_feature_store.ipynb)

```
Learn how to create and use an online feature store instance to host and serve data in `BigQuery` with `Vertex AI Feature Store` in an end to end workflow of feature values serving and fetching user journey.

The steps performed include:

- Provision an online feature store instance to host and serve data.
- Register a `BigQuery` view with the online feature store instance and set up the sync job.
- Use the online server to fetch feature values for online prediction.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Feature Store](https://cloud.google.com/vertex-ai/docs/featurestore/overview).


[Online feature serving and vector retrieval of BigQuery data with Vertex AI Feature Store](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/feature_store/online_feature_serving_and_vector_retrieval_bigquery_data_with_feature_store.ipynb)

```
Learn how to create and use an online feature store instance to host and serve data in `BigQuery` with `Vertex AI Feature Store` in an end to end workflow of features serving and vector retrieval user journey.

The steps performed include:

- Provision an online feature store instance to host and serve data.
- Create an online feature store instance to serve a `BigQuery` table.
- Use the online server to search nearest neighbors.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Feature Store](https://cloud.google.com/vertex-ai/docs/featurestore/overview).


[Using Vertex AI Feature Store with Pandas Dataframe](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/feature_store/sdk-feature-store-pandas.ipynb)

```
Learn how to use `Vertex AI Feature Store` with pandas Dataframe.

The steps performed include:

- Create Featurestore, entity types and features.
- Ingest feature values from Pandas DataFrame into Feature Store's Entity types.
- Read Entity feature values from Online Feature Store into Pandas DataFrame.
- Batch serve feature values from your Feature Store into Pandas DataFrame.


- Online serving with updated feature values.
- Point-in-time correctness to fetch feature values for training.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Feature Store](https://cloud.google.com/vertex-ai/docs/featurestore).


[Online and Batch predictions using Vertex AI Feature Store](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/feature_store/sdk-feature-store.ipynb)

```
Learn how to use `Vertex AI Feature Store` to import feature data, and to access the feature data for both online serving and offline tasks, such as training.

The steps performed include:

- Create featurestore, entity type, and feature resources.
- Import feature data into `Vertex AI Feature Store` resource.
- Serve online prediction requests using the imported features.
- Access imported features in offline jobs, such as training jobs.
- Use streaming ingestion to ingest small amount of data.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Feature Store](https://cloud.google.com/vertex-ai/docs/featurestore).

