
[Unstructured data analytics with BigQuery ML and Vertex AI pre-trained models](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/bigquery_ml_inference/bq_ml_with_vision_translation_nlp.ipynb)

```
Learn how to analyze unstructured data within BigQuery using BigQuery's inference engine. You will use BigQuery ML to connect to three pretrained Vertex AI APIs - Vision API, Translation API and Natural Language Processing API.

The steps performed include:

- Define pre-trained models for Vision AI, Translation AI and NLP AI in BigQuery ML
- Cal the Vision API (`ML.ANNOTATE_IMAGE`) to detect text in images stored in Cloud Storage
  You will need to create an object table in BigQuery to do this
- Call the Translation API (`ML.TRANSLATE`) to detect the language of text, and translate non-English movie titles to English
- Call the Natural Language API (`ML.UNDERSTAND_TEXT`) to run sentiment analysis over movie reviews stored in BigQuery

```

&nbsp;&nbsp;&nbsp;Learn more about [BigQuery ML inference engine](https://cloud.google.com/bigquery/docs/reference/standard-sql/inference-overview).

