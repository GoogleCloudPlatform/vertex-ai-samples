
[Custom model batch prediction with feature filtering](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/prediction/custom_batch_prediction_feature_filter.ipynb)

```
Learn how to create a custom-trained model from a Python script in a Docker container using the Vertex AI SDK for Python, and then run a batch prediction job by including or excluding a list of features.

The steps performed include:

- Create a Vertex AI custom `TrainingPipeline` for training a model.
- Train a TensorFlow model.
- Send batch prediction job.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Batch Prediction](https://cloud.google.com/vertex-ai/docs/tabular-data/classification-regression/get-batch-predictions).


[Get started with Custom Prediction Routine (CPR)](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/prediction/get_started_with_cpr.ipynb)

```
Learn how to use Custom Prediction Routine  for `Vertex AI Predictions`.

The steps performed include:

- Write a custom data preprocessor.
- Train the model.
- Build a custom scikit-learn serving container with custom data preprocessing using the Custom Prediction Routine model server.
    - Test the model serving container locally.
    - Upload and deploy the model serving container to Vertex AI Endpoint.
    - Make a prediction request.
- Build a custom scikit-learn serving container with custom predictor (post-processing) using the Custom Prediction Routine model server.
    - Implement custom predictor.
    - Test the model serving container locally.
    - Upload and deploy the model serving container to Vertex AI Endpoint.
    - Make a prediction request.
- Build a custom scikit-learn serving container with custom predictor and HTTP request handler using the Custom Prediction Routine model server.
    - Implement a custom handler.
    - Test the model serving container locally.
    - Upload and deploy the model serving container to Vertex AI Endpoint.
    - Make a prediction request.
- Customize the Dockerfile for a custom scikit-learn serving container with custom predictor and HTTP request handler using the Custom Prediction Routine model server.
    - Implement a custom Dockerfile.
    - Test the model serving container locally.
    - Upload and deploy the model serving container to Vertex AI Endpoint.
    - Make a prediction request.

```

&nbsp;&nbsp;&nbsp;Learn more about [Custom prediction routines](https://cloud.google.com/vertex-ai/docs/predictions/custom-prediction-routines).


[Vertex AI LLM and streaming prediction](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/prediction/llm_streaming_prediction.ipynb)

```
Learn how to use Vertex AI LLM to download pretrained LLM model, make predictions and finetuning the model.

The steps performed include:

- Load a pretrained text generation model.
- Make a non-streaming prediction
- Load a pretrained text generation model, which supports streaming.
- Make a streaming prediction
- Load a pretrained chat model.
- Do a local interactive chat session.
- Do a batch prediction with a text generation model.
- Do a batch prediction with a text embedding model.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Language Models](https://cloud.google.com/python/docs/reference/aiplatform/latest/vertexai.language_models.TextGenerationModel#vertexai_language_models_TextGenerationModel_predict_streaming).


[Serving PyTorch image models with prebuilt containers on Vertex AI](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/prediction/pytorch_image_classification_with_prebuilt_serving_containers.ipynb)

```
Learn how to package and deploy a PyTorch image classification model using a prebuilt Vertex AI container with TorchServe for serving online and batch predictions.

The steps performed include:

- Download a pretrained image model from PyTorch
- Create a custom model handler
- Package model artifacts in a model archive file
- Upload model for deployment
- Deploy model for prediction
- Make online predictions
- Make batch predictions

```

&nbsp;&nbsp;&nbsp;Learn more about [Pre-built containers for prediction](https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers).


[Train and deploy PyTorch models with prebuilt containers on Vertex AI](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/prediction/pytorch_train_deploy_models_with_prebuilt_containers.ipynb)

```
Learn how to build, train and deploy a PyTorch image classification model using prebuilt containers for custom training and prediction.

The steps performed include:

- Package training application into a Python source distribution
- Configure and run training job in a prebuilt container
- Package model artifacts in a model archive file
- Upload model for deployment
- Deploy model using a prebuilt container for prediction
- Make online predictions

```


[Vertex AI SDK 2.0 Vertex AI Remote Prediction](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/prediction/sdk2_remote_prediction.ipynb)

```
Learn to use `Vertex AI SDK 2.

The steps performed include:

- Download and split the dataset
- Perform transformations as a Vertex AI remote training.
- For scikit-learn, PyTorch, TensorFlow, PyTorch Lightning
    - Train the model remotely.
    - Uptrain the pretrained model remotely.
    - Evaluate both the pretrained and uptrained model.
    - Make a prediction remotely

```

