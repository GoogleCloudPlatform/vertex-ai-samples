
[AutoML training tabular binary classification model for batch explanation](official/explainable_ai/sdk_automl_tabular_binary_classification_batch_explain.ipynb)

Learn to use `AutoML` to create a tabular binary classification model from a Python script, and then learn to use `Vertex AI Batch Prediction` to make predictions with explanations.

The steps performed include:

- Create a `Vertex Dataset` resource.
- Train an `AutoML` tabular binary classification model.
- View the model evaluation metrics for the trained model.
- Make a batch prediction request with explainability.


* Prediction Service: Does an on-demand prediction for the entire set of instances (i.e., one or more data items) and returns the results in real-time.

* Batch Prediction Service: Does a queued (batch) prediction for the entire set of instances in the background and stores the results in a Cloud Storage bucket when ready.

[Custom training image classification model for batch prediction with explainabilty](official/explainable_ai/sdk_custom_image_classification_batch_explain.ipynb)

Learn to use `Vertex AI Training and Explainable AI` to create a custom image classification model with explanations, and then you learn to use `Vertex AI Batch Prediction` to make a batch prediction request with explanations.

The steps performed include:

- Create a `Vertex AI` custom job for training a TensorFlow model.
- View the model evaluation for the trained model.
- Set explanation parameters for when the model is deployed.
- Upload the trained model artifacts and explanation parameters as a `Model` resource.
- Make a batch prediction with explanations.
