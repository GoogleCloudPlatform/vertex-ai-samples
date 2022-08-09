# Stage 6: Serving

## Purpose

Process prediction requests and return corresponding predictions in a timely manner consistent with the business requirement, whether online, on-demand or batch predictions

## Recommendations  

The sixth stage in MLOps is serving predictions from the blessed model deployed to production. The serving methods, depending on business requirements may be one or more of the following:

- Batch predictions – prediction requests that are queued and handled offline. This is done entirely with Google Cloud core infrastructure.

- Online predictions - prediction requests that are received externally over the Internet and processed in (near) real-time. The serving of the requests/responses is done entirely with Google Cloud core infrastructure, the requesting web application/clients may originate anywhere on the Internet. If the request originates outside of the Google Cloud core infrastructure, a proxy is needed to traverse through the firewall.

- On-demand predictions - prediction requests that are received internally with Google Cloud core infrastructure, or direct via an edge device. The prediction response to the requestor must be near instantaneous. The serving of the requests/responses may be either within Google Cloud core infrastructure, or externally on an edge device. An example of the former is an emergency sensor and on the later a medical sensor.

This stage may be done entirely by MLOps. We recommend:

- Use Google Cloud core infrastructure for online serving and batch serving, and on-demand serving where it meets the speed requirements for how the responses are utilized.
- Use IAM role settings for access control in cross-project when the application and the serving binaries are entirely within Google Cloud core infrastructure, but in different projects.
- Deploy serving binaries within regions that are the closest to where the requests originate. Deploy in multiple regions, when requests span regional boundaries.
- Use Cloud Functions as a proxy when prediction requests originate externally to Google Cloud core infrastructure, or must otherwise cross firewall boundaries that cannot not otherwise be handled by IAM role settings.
- Features that dynamically change per example (e.g., bank balance) are stored in Vertex Feature Store.


<img src='stage6v2.png'>

## Notebooks

### Get Started


[Get started with TensorFlow serving functions with Vertex AI Prediction](get_started_with_tf_serving_function.ipynb)

```
The steps performed include:
- Download a pretrained image classification model from TensorFlow Hub.
- Create a serving function to receive compressed image data, and output decomopressed preprocessed data for the model input.
- Upload the TensorFlow Hub model and serving function as a `Vertex AI Model` resource.
- Creating an `Endpoint` resource.
- Deploying the `Model` resource to an `Endpoint` resource.
- Make an online prediction to the `Model` resource instance deployed to the `Endpoint` resource.
```

[Get started with FastAPI with Vertex AI Prediction](get_started_with_fastapi.ipynb)

```
The steps performed include:
- Download a pretrained image classification model from TensorFlow Hub.
- Create a serving function to receive compressed image data, and output decomopressed preprocessed data for the model input.
- Upload the TensorFlow Hub model and serving function as a `Vertex AI Model` resource.
- Creating an `Endpoint` resource.
- Deploying the `Model` resource to an `Endpoint` resource with `FastAPI` custom serving binary.
- Make an online prediction to the `Model` resource instance deployed to the `Endpoint` resource.
```

[Get started with Nvidia Triton server](get_started_with_nvidia_triton_serving.ipynb)

```
The steps performed in this tutorial include:
- Download the model artifacts from TensorFlow Hub.
- Create Triton serving configuration file for the model.
- Construct a custom container, with Triton serving image, for model deployment.
- Upload the model as a `Vertex AI Model` resource.
- Deploy the `Vertex AI Model` resource to a `Vertex AI Endpoint` resource.
- Make a prediction request
- Undeploy the `Model` resource and delete the `Endpoint`

```

[Get started with Custom Prediction Routine (CPR)](get_started_with_cpr.ipynb)

```
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

[Get started with re-importing AutoML tabular models](get_started_automl_tabular_exported_deploy.ipynb)

```
The steps performed include:
- Importing a pretrained AutoML tabular exported model artifacts, as a `Model` resource.
- Create an `Endpoint` resource.
- Deploy the `Model` resource to the `Endpoint` resource.
- Make a prediction.
```

[Get started with Vertex Explainable AI using custom deployment container](get_started_with_xai_and_custom_server.ipynb)

```
The steps performed include:
- Locally train a Pytorch tabular classifier.
- Locally test the trained model.
- Build a HTTP server using FastAPI.
- Create a custom serving container with the trained model and FastAPI server.
- Locally test the custom serving container.
- Push the custom serving container to the Artifact Registry.
- Upload the custom serving container as a `Model` resource.
- Deploy the `Model` resource to an `Endpoint` resource.
- Make a prediction request to the deployed custom serving container.
- Make an explanation request to the deployed custom serving container.

```

[Get started with TensorFlow serving functions with Vertex AI Raw Prediction](get_started_with_raw_predict.ipynb)

```
The steps performed include:
- Download a pretrained tabular classification model artifacts for a TensorFlow 1.x estimator.
- Upload the TensorFlow estimator model as a `Vertex AI Model` resource.
- Creating an `Endpoint` resource.
- Deploying the `Model` resource to an `Endpoint` resource.
- Make an online raw prediction to the `Model` resource instance deployed to the `Endpoint` resource.
```

[Get started with Vertex AI Matching Engine and Two Towers builtin algorithm](get_started_with_matching_engine_twotowers.ipynb)

```
The steps performed include:
1. Train the `Two-Tower` algorithm to generate embeddings (encoder) for the dataset.
2. Hyperparameter tune the trained `Two-Tower` encoder.
3. Make example predictions (embeddings) from then trained encoder.
4. Generate embeddings using the trained `Two-Tower` builtin algorithm.
5. Store embeddings to format supported by `Matching Engine`.
6. Create a `Matching Engine Index` for the embeddings.
7. Deploy the `Matching Engine Index` to a `Index Endpoint`.
8. Make a matching engine prediction request.

```

[Get started with Optimized TensorFlow Enterprise container with Vertex AI Prediction / text models](get_started_with_optimized_tfe_bert.ipynb)

```
The steps performed include:
- Download a pretrained BERT model from TensorFlow Hub.
- Fine-tune (transfer learning) the BERT model as a binary classifier.
- Upload the TensorFlow Hub model as a `Vertex AI Model` resource, with standard TensorFlow serving container.
- Upload the TensorFlow Hub model as a `Vertex AI Model` resource, with TensorFlow Enterprise Optimized container
- Create two `Endpoint` resources.
- Deploying both `Model` resources to separate `Endpoint` resources.
- Make the same online prediction requests to both `Model` resource instances deployed to the `Endpoint` resources.
- Compare the prediction accuracy between the two deployed `Model` resources.
- Configuring container settings for fine-tune control of optimizations.
- Create a `Private Endpoint` resource.
- Deploy the `Model` resoure with then `TensorFlow Enterprise Optimized` to the `Private Endpoint` resource.
- Make an online prediction request to the `Private Endpoint` resource.

```

[Get started with Vertex AI Matching Engine](get_started_with_matching_engine.ipynb)

```
The steps performed include:
- Create ANN Index.
- Create an IndexEndpoint with VPC Network
- Deploy ANN Index
- Perform online query
- Deploy brute force Index.
- Perform calibration between ANN and brute force index.

```

[Get started with Vertex AI Matching Engine and Swivel builtin algorithm](get_started_with_matching_engine_swivel.ipynb)

```
The steps performed include:
1. Train the `Swivel` algorithm to generate embeddings (encoder) for the dataset.
2. Make example predictions (embeddings) from then trained encoder.
3. Generate embeddings using the trained `Swivel` builtin algorithm.
4. Store embeddings to format supported by `Matching Engine`.
5. Create a `Matching Engine Index` for the embeddings.
6. Deploy the `Matching Engine Index` to a `Index Endpoint`.
7. Make a matching engine prediction request.

```

[Get started with TensorFlow serving with Vertex AI Prediction](get_started_with_tf_serving.ipynb)

```
The steps performed include:
- Download a pretrained image classification model from TensorFlow Hub.
- Create a serving function to receive compressed image data, and output decomopressed preprocessed data for the model input.
- Upload the TensorFlow Hub model and serving function as a `Vertex AI Model` resource.
- Creating an `Endpoint` resource.
- Deploying the `Model` resource to an `Endpoint` resource with `TensorFlow Serving` serving binary.
- Make an online prediction to the `Model` resource instance deployed to the `Endpoint` resource.
```