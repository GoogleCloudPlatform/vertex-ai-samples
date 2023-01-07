
[Sentiment Analysis using AutoML Natural Language and Vertex AI](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/workbench/sentiment_analysis/Sentiment_Analysis.ipynb)

```
Learn how to train and deploy an AutoML sentiment analysis model, and make predictions.

The steps performed are:

- Loading the required data. 
- Preprocessing the data.
- Selecting the required data for the model.
- Loading the dataset into Vertex AI managed datasets.
- Training a sentiment model using AutoML Text training.
- Evaluating the model.
- Deploying the model on Vertex AI.
- Getting predictions.
- Clean up.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Workbench](https://cloud.google.com/vertex-ai/docs/workbench/introduction).

&nbsp;&nbsp;&nbsp;Learn more about [AutoML Text](https://cloud.google.com/vertex-ai/docs/tutorials/text-classification-automl/training).


[Interactive exploratory analysis of BigQuery data in a notebook](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/workbench/exploratory_data_analysis/explore_data_in_bigquery_with_workbench.ipynb)

```
Learn about various ways to explore and gain insights from BigQuery data in a Jupyter notebook environment.

The steps performed include:

- Using Python & SQL to query public data in BigQuery
- Exploring the dataset using BigQuery INFORMATION_SCHEMA
- Creating interactive elements to help explore interesting parts of the data
- Doing some exploratory correlation and time series analysis
- Creating static and interactive outputs (data tables and plots) in the notebook
- Saving some outputs to Cloud Storage

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Workbench](https://cloud.google.com/vertex-ai/docs/workbench/introduction).

&nbsp;&nbsp;&nbsp;Learn more about [BigQuery](https://cloud.google.com/bigquery).


[Forecasting retail demand with Vertex AI and BigQuery ML](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/workbench/demand_forecasting/forecasting-retail-demand.ipynb)

```
Learn how to build ARIMA (Autoregressive integrated moving average) model from BigQuery ML on retail data

The steps performed include:

* Explore data
* Model with BigQuery and the ARIMA model
* Evaluate the model
* Evaluate the model results using BigQuery ML (on training data)
* Evalute the model results - MAE, MAPE, MSE, RMSE (on test data)
* Use the executor feature

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Workbench](https://cloud.google.com/vertex-ai/docs/workbench/introduction).

&nbsp;&nbsp;&nbsp;Learn more about [BigQuery ML](https://cloud.google.com/bigquery-ml/docs/managing-models-vertex).

[Predictive Maintenance using Vertex AI](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/workbench/predictive_maintainance/predictive_maintenance_usecase.ipynb)

```

The steps performed are:

- Loading the required dataset from a Cloud Storage bucket.
- Analyzing the fields present in the dataset.
- Selecting the required data for the predictive maintenance model.
- Training an XGBoost regression model for predicting the remaining useful life.
- Evaluating the model.
- Running the notebook end-to-end as a training job using Executor.
- Deploying the model on Vertex AI.
- Clean up.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Workbench](https://cloud.google.com/vertex-ai/docs/workbench/introduction).

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Training](https://cloud.google.com/vertex-ai/docs/training/custom-training).


[Predictive Maintenance using Vertex AI](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/workbench/predictive_maintainance/predictive_maintenance_usecase.ipynb)

```
Learn how to the executor feature of Vertex AI Workbench to automate a workflow to train and deploy a model.

```
The steps performed are:

- Loading the required dataset from a Cloud Storage bucket.
- Analyzing the fields present in the dataset.
- Selecting the required data for the predictive maintenance model.
- Training an XGBoost regression model for predicting the remaining useful life.
- Evaluating the model.
- Running the notebook end-to-end as a training job using Executor.
- Deploying the model on Vertex AI.
- Clean up.
```


[Telecom subscriber churn prediction on Vertex AI](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/workbench/subscriber_churn_prediction/telecom-subscriber-churn-prediction.ipynb)

```
This tutorial shows you how to do exploratory data analysis, preprocess data, train, deploy and get predictions from a churn prediction model on a tabular churn dataset.

The steps performed include:

- Load data from a Cloud Storage path
- Perform exploratory data analysis (EDA)
- Preprocess the data
- Train a scikit-learn model
- Evaluate the scikit-learn model
- Save the model to a Cloud Storage path
- Create a model and an endpoint in Vertex AI
- Deploy the trained model to an endpoint
- Generate predictions and explanations on test data from the hosted model
- Undeploy the model resource

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Workbench](https://cloud.google.com/vertex-ai/docs/workbench/introduction).

&nbsp;&nbsp;&nbsp;Learn more about [Vertex Explainable AI](https://cloud.google.com/vertex-ai/docs/explainable-ai/overview).


[SparkML with Dataproc and BigQuery](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/workbench/spark/spark_ml.ipynb)

```
This tutorial runs an Apache SparkML job that fetches data from the BigQuery dataset, performs exploratory data analysis, cleans the data, executes feature engineering, trains the model, evaluates the model, outputs results, and saves the model to a Cloud Storage bucket.

The steps performed are:

- Sets up a Google Cloud project and Dataproc cluster.
- Creates a Cloud Storage bucket and a BigQuery dataset.
- Configures the spark-bigquery-connector.
- Ingests BigQuery data into a Spark DataFrame.
- Performa Exploratory Data Analysis (EDA).
- Visualizes the data with samples.
- Cleans the data.
- Selects features.
- Trains the model.
- Outputs results.
- Saves the model to a Cloud Storage bucket.
- Deletes the resources created for the tutorial.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Workbench](https://cloud.google.com/vertex-ai/docs/workbench/introduction).

&nbsp;&nbsp;&nbsp;Learn more about [Dataproc](https://cloud.google.com/vertex-ai/docs/pipelines/dataproc-component).


[Digest and analyze data from BigQuery with Dataproc](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/workbench/spark/spark_bigquery.ipynb)

```
This notebook tutorial runs an Apache Spark job that fetches data from the BigQuery "GitHub Activity Data" dataset, queries the data, and then writes the results back to BigQuery.

The steps performed are:

- Setting up a Google Cloud project and Dataproc cluster.
- Configuring the spark-bigquery-connector.
- Ingesting data from BigQuery into a Spark DataFrame.
- Preprocessing ingested data.
- Querying the most frequently used programming language in monoglot repos.
- Querying the average size (MB) of code in each language stored in monoglot repos.
- Querying the languages files most frequently found together in polyglot repos.
- Writing the query results back into BigQuery.
- Deleting the resources created for this notebook tutorial.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Workbench](https://cloud.google.com/vertex-ai/docs/workbench/introduction).

&nbsp;&nbsp;&nbsp;Learn more about [Dataproc](https://cloud.google.com/vertex-ai/docs/pipelines/dataproc-component).


[Train a multi-class classification model for ads-targeting](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/workbench/ads_targetting/training-multi-class-classification-model-for-ads-targeting-usecase.ipynb)

```
Learn how to collect data from BigQuery, preprocess it, and train a multi-class classification model on an e-commerce dataset.

The steps performed include:

- Fetch the required data from BigQuery
- Preprocess the data
- Train a TensorFlow (>=2.4) classification model
- Evaluate the loss for the trained model
- Automate the notebook execution using the executor feature
- Save the model to a Cloud Storage path
- Clean up the created resources

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Workbench](https://cloud.google.com/vertex-ai/docs/workbench/introduction).

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Training](https://cloud.google.com/vertex-ai/docs/training/custom-training).


[Inventory prediction on ecommerce data using Vertex AI](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/workbench/inventory-prediction/inventory_prediction.ipynb)

```
This tutorial shows you how to do exploratory data analysis, preprocess data, train model, evaluate model, deploy model, configure What-If Tool.

The steps performed include:

* Load the dataset from BigQuery using the "BigQuery in Notebooks" integration.
* Analyze the dataset.
* Preprocess the features in the dataset.
* Build a random forest classifier model that predicts whether a product will get sold in the next 60 days.
* Evaluate the model.
* Deploy the model using Vertex AI.
* Configure and test with the What-If Tool.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Workbench](https://cloud.google.com/vertex-ai/docs/workbench/introduction).

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Training](https://cloud.google.com/vertex-ai/docs/training/custom-training).


[Build a fraud detection model on Vertex AI](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/workbench/fraud_detection/fraud-detection-model.ipynb)

```
This tutorial demonstrates data analysis and model-building using a synthetic financial dataset.

The steps performed include:

- Installation of required libraries
- Reading the dataset from a Cloud Storage bucket
- Performing exploratory analysis on the dataset
- Preprocessing the dataset
- Training a random forest model using scikit-learn
- Saving the model to a Cloud Storage bucket
- Creating a Vertex AI model resource and deploying to an endpoint
- Running the What-If Tool on test data
- Un-deploying the model and cleaning up the model resources

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Workbench](https://cloud.google.com/vertex-ai/docs/workbench/introduction).

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Training](https://cloud.google.com/vertex-ai/docs/training/custom-training).


[Taxi fare prediction using the Chicago Taxi Trips dataset](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/workbench/chicago_taxi_fare_prediction/chicago_taxi_fare_prediction.ipynb)

```
The goal of this notebook is to provide an overview on the latest Vertex AI features like **Explainable AI** and **BigQuery in Notebooks** by trying to solve a taxi fare prediction problem.

The steps performed include:

- Loading the dataset using "BigQuery in Notebooks".
- Performing exploratory data analysis on the dataset.
- Feature selection and preprocessing.
- Building a linear regression model using scikit-learn.
- Configuring the model for Vertex Explainable AI.
- Deploying the model to Vertex AI.
- Testing the deployed model.
- Clean up.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Workbench](https://cloud.google.com/vertex-ai/docs/workbench/introduction).

&nbsp;&nbsp;&nbsp;Learn more about [Vertex Explainable AI](https://cloud.google.com/vertex-ai/docs/explainable-ai/overview).


[Churn prediction for game developers using Google Analytics 4 and BigQuery ML](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/workbench/gaming_churn_prediction/churn_prediction_for_game_developers.ipynb)

```
Learn how to train, evaluate a propensity model in BigQuery ML.

The steps performed include:

* Explore an export of Google Analytics 4 data on BigQuery.
* Prepare the training data using demographic, behavioral data, and labels (churn/not-churn).
* Train an XGBoost model using BigQuery ML.
* Evaluate a model using BigQuery ML.
* Make predictions on which users will churn using BigQuery ML.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Workbench](https://cloud.google.com/vertex-ai/docs/workbench/introduction).

&nbsp;&nbsp;&nbsp;Learn more about [BigQuery ML](https://cloud.google.com/bigquery-ml/docs/managing-models-vertex).


[Analysis of pricing optimization on CDM Pricing Data](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/workbench/pricing_optimization/pricing-optimization.ipynb)

```
The objective of this notebook is to build a pricing optimization model using BigQuery ML.

The steps performed include:

- Load the required dataset from a Cloud Storage bucket.
- Analyze the fields present in the dataset.
- Process the data to build a model.
- Build a BigQuery ML forecast model on the processed data.
- Get forecasted values from the BigQuery ML model.
- Interpret the forecasts to identify the best prices.
- Clean up.

```

&nbsp;&nbsp;&nbsp;Learn more about [Vertex AI Workbench](https://cloud.google.com/vertex-ai/docs/workbench/introduction).

&nbsp;&nbsp;&nbsp;Learn more about [BigQuery ML](https://cloud.google.com/bigquery-ml/docs/managing-models-vertex).

