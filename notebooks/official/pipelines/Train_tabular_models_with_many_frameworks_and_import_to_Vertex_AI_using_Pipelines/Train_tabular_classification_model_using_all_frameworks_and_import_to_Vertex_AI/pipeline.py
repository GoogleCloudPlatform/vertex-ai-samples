# python3 -m pip install "kfp<2.0.0" "google-cloud-aiplatform>=1.16.0" --upgrade --quiet
from kfp import components

# %% Loading components
download_from_gcs_op = components.load_component_from_url("https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/399405402d95f4a011e2d2e967c96f8508ba5688/community-content/pipeline_components/google-cloud/storage/download/component.yaml")
select_columns_using_Pandas_on_CSV_data_op = components.load_component_from_url("https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/399405402d95f4a011e2d2e967c96f8508ba5688/community-content/pipeline_components/pandas/Select_columns/in_CSV_format/component.yaml")
fill_all_missing_values_using_Pandas_on_CSV_data_op = components.load_component_from_url("https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/399405402d95f4a011e2d2e967c96f8508ba5688/community-content/pipeline_components/pandas/Fill_all_missing_values/in_CSV_format/component.yaml")
binarize_column_using_Pandas_on_CSV_data_op = components.load_component_from_url("https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/399405402d95f4a011e2d2e967c96f8508ba5688/community-content/pipeline_components/pandas/Binarize_column/in_CSV_format/component.yaml")
split_rows_into_subsets_op = components.load_component_from_url("https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/399405402d95f4a011e2d2e967c96f8508ba5688/community-content/pipeline_components/dataset_manipulation/Split_rows_into_subsets/in_CSV/component.yaml")

# TensorFlow
create_fully_connected_tensorflow_network_op = components.load_component_from_url("https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/399405402d95f4a011e2d2e967c96f8508ba5688/community-content/pipeline_components/tensorflow/Create_fully_connected_network/component.yaml")
train_model_using_Keras_on_CSV_op = components.load_component_from_url("https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/399405402d95f4a011e2d2e967c96f8508ba5688/community-content/pipeline_components/tensorflow/Train_model_using_Keras/on_CSV/component.yaml")
predict_with_TensorFlow_model_on_CSV_data_op = components.load_component_from_url("https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/399405402d95f4a011e2d2e967c96f8508ba5688/community-content/pipeline_components/tensorflow/Predict/on_CSV/component.yaml")
upload_Tensorflow_model_to_Google_Cloud_Vertex_AI_op = components.load_component_from_url("https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/399405402d95f4a011e2d2e967c96f8508ba5688/community-content/pipeline_components/google-cloud/Vertex_AI/Models/Upload_Tensorflow_model/component.yaml")

# PyTorch
create_fully_connected_pytorch_network_op = components.load_component_from_url("https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/399405402d95f4a011e2d2e967c96f8508ba5688/community-content/pipeline_components/PyTorch/Create_fully_connected_network/component.yaml")
train_pytorch_model_from_csv_op = components.load_component_from_url("https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/399405402d95f4a011e2d2e967c96f8508ba5688/community-content/pipeline_components/PyTorch/Train_PyTorch_model/from_CSV/component.yaml")
create_pytorch_model_archive_with_base_handler_op = components.load_component_from_url("https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/399405402d95f4a011e2d2e967c96f8508ba5688/community-content/pipeline_components/PyTorch/Create_PyTorch_Model_Archive/with_base_handler/component.yaml")
upload_PyTorch_model_archive_to_Google_Cloud_Vertex_AI_op = components.load_component_from_url("https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/399405402d95f4a011e2d2e967c96f8508ba5688/community-content/pipeline_components/google-cloud/Vertex_AI/Models/Upload_PyTorch_model_archive/component.yaml")

# XGBoost
train_XGBoost_model_on_CSV_op = components.load_component_from_url("https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/399405402d95f4a011e2d2e967c96f8508ba5688/community-content/pipeline_components/XGBoost/Train/component.yaml")
xgboost_predict_on_CSV_op = components.load_component_from_url("https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/399405402d95f4a011e2d2e967c96f8508ba5688/community-content/pipeline_components/XGBoost/Predict/component.yaml")
upload_XGBoost_model_to_Google_Cloud_Vertex_AI_op = components.load_component_from_url("https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/d5c9918850a6cc70004c4269dae066cfe2e664eb/community-content/pipeline_components/google-cloud/Vertex_AI/Models/Upload_XGBoost_model/component.yaml")

# Scikit-learn
#train_linear_regression_model_using_scikit_learn_from_CSV_op = components.load_component_from_url("https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/1f5cf6e06409b704064b2086c0a705e4e6b4fcde/community-content/pipeline_components/ML_frameworks/Scikit_learn/Train_linear_regression_model/from_CSV/component.yaml")
train_logistic_regression_model_using_scikit_learn_from_CSV_op = components.load_component_from_url("https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/1f5cf6e06409b704064b2086c0a705e4e6b4fcde/community-content/pipeline_components/ML_frameworks/Scikit_learn/Train_logistic_regression_model/from_CSV/component.yaml")
upload_Scikit_learn_pickle_model_to_Google_Cloud_Vertex_AI_op = components.load_component_from_url("https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/399405402d95f4a011e2d2e967c96f8508ba5688/community-content/pipeline_components/google-cloud/Vertex_AI/Models/Upload_Scikit-learn_pickle_model/component.yaml")

# Vertex AI
deploy_model_to_endpoint_op = components.load_component_from_url("https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/399405402d95f4a011e2d2e967c96f8508ba5688/community-content/pipeline_components/google-cloud/Vertex_AI/Models/Deploy_to_endpoint/component.yaml")

# %% Pipeline definition
def train_tabular_classification_model_using_all_frameworks_pipeline():
    dataset_gcs_uri = "gs://ml-pipeline-dataset/Chicago_taxi_trips/chicago_taxi_trips_2019-01-01_-_2019-02-01_limit=10000.csv"
    feature_columns = ["trip_seconds", "trip_miles", "pickup_community_area", "dropoff_community_area", "fare", "tolls", "extras"]  # Excluded "trip_total"
    label_column = "tips"
    training_set_fraction = 0.8
    # Deploying the model might incur additional costs over time
    deploy_model = False

    classification_label_column = "class"
    all_columns = [label_column] + feature_columns

    dataset = download_from_gcs_op(
        gcs_path=dataset_gcs_uri
    ).outputs["Data"]

    dataset = select_columns_using_Pandas_on_CSV_data_op(
        table=dataset,
        column_names=all_columns,
    ).outputs["transformed_table"]

    dataset = fill_all_missing_values_using_Pandas_on_CSV_data_op(
        table=dataset,
        replacement_value="0",
        # # Optional:
        # column_names=None,  # =[...]
    ).outputs["transformed_table"]

    classification_dataset = binarize_column_using_Pandas_on_CSV_data_op(
        table=dataset,
        column_name=label_column,
        predicate=" > 0",
        new_column_name=classification_label_column,
    ).outputs["transformed_table"]

    split_task = split_rows_into_subsets_op(
        table=classification_dataset,
        fraction_1=training_set_fraction,
    )
    classification_training_data = split_task.outputs["split_1"]
    classification_testing_data = split_task.outputs["split_2"]

    # TensorFlow
    tensorflow_network = create_fully_connected_tensorflow_network_op(
        input_size=len(feature_columns),
        # Optional:
        hidden_layer_sizes=[10],
        activation_name="elu",
        output_activation_name="sigmoid",
        # output_size=1,
    ).outputs["model"]

    tensorflow_model = train_model_using_Keras_on_CSV_op(
        training_data=classification_training_data,
        model=tensorflow_network,
        label_column_name=classification_label_column,
        # Optional:
        loss_function_name="binary_crossentropy",
        number_of_epochs=10,
        #learning_rate=0.1,
        #optimizer_name="Adadelta",
        #optimizer_parameters={},
        #batch_size=32,
        #metric_names=["mean_absolute_error"],
        #random_seed=0,
    ).outputs["trained_model"]

    tensorflow_predictions = predict_with_TensorFlow_model_on_CSV_data_op(
        dataset=classification_testing_data,
        model=tensorflow_model,
        # label_column_name needs to be set when doing prediction on a dataset that has labels
        label_column_name=classification_label_column,
        # Optional:
        # batch_size=1000,
    ).outputs["predictions"]

    tensorflow_vertex_model_name = upload_Tensorflow_model_to_Google_Cloud_Vertex_AI_op(
        model=tensorflow_model,
    ).outputs["model_name"]

    # Deploying the model might incur additional costs over time
    if deploy_model:
        tensorflow_vertex_endpoint_name = deploy_model_to_endpoint_op(
            model_name=tensorflow_vertex_model_name,
        ).outputs["endpoint_name"]

    # PyTorch
    pytorch_network = create_fully_connected_pytorch_network_op(
        input_size=len(feature_columns),
        # Optional:
        hidden_layer_sizes=[10],
        activation_name="elu",
        output_activation_name="sigmoid",
        # output_size=1,
    ).outputs["model"]

    pytorch_model = train_pytorch_model_from_csv_op(
        model=pytorch_network,
        training_data=classification_training_data,
        label_column_name=classification_label_column,
        loss_function_name="binary_cross_entropy",
        # Optional:
        #number_of_epochs=1,
        #learning_rate=0.1,
        #optimizer_name="Adadelta",
        #optimizer_parameters={},
        #batch_size=32,
        #batch_log_interval=100,
        #random_seed=0,
    ).outputs["trained_model"]

    pytorch_model_archive = create_pytorch_model_archive_with_base_handler_op(
        model=pytorch_model,
        # Optional:
        # model_name="model",
        # model_version="1.0",
    ).outputs["Model archive"]

    pytorch_vertex_model_name = upload_PyTorch_model_archive_to_Google_Cloud_Vertex_AI_op(
        model_archive=pytorch_model_archive,
    ).outputs["model_name"]

    # Deploying the model might incur additional costs over time
    if deploy_model:
        pytorch_vertex_endpoint_name = deploy_model_to_endpoint_op(
            model_name=pytorch_vertex_model_name,
        ).outputs["endpoint_name"]

    # XGBoost
    xgboost_model = train_XGBoost_model_on_CSV_op(
        training_data=classification_training_data,
        label_column_name=classification_label_column,
        objective="binary:logistic",
        # Optional:
        #starting_model=None,
        #num_iterations=10,
        #booster_params={},
        #booster="gbtree",
        #learning_rate=0.3,
        #min_split_loss=0,
        #max_depth=6,
    ).outputs["model"]

    # Predicting on the testing data
    xgboost_predictions = xgboost_predict_on_CSV_op(
        data=classification_testing_data,
        model=xgboost_model,
        # label_column needs to be set when doing prediction on a dataset that has labels
        label_column_name=classification_label_column,
    ).outputs["predictions"]

    xgboost_vertex_model_name = upload_XGBoost_model_to_Google_Cloud_Vertex_AI_op(
        model=xgboost_model,
    ).outputs["model_name"]

    # Deploying the model might incur additional costs over time
    if deploy_model:
        xgboost_vertex_endpoint_name = deploy_model_to_endpoint_op(
            model_name=xgboost_vertex_model_name,
        ).outputs["endpoint_name"]

    # Scikit-learn
    sklearn_model = train_logistic_regression_model_using_scikit_learn_from_CSV_op(
        dataset=classification_training_data,
        label_column_name=classification_label_column,
        # Optional:
        #penalty="l2",
        #solver="lbfgs",
        #max_iterations=100,
        #multi_class_mode="auto",
        #random_seed=0,
    ).outputs["model"]

    sklearn_vertex_model_name = upload_Scikit_learn_pickle_model_to_Google_Cloud_Vertex_AI_op(
        model=sklearn_model,
    ).outputs["model_name"]

    # Deploying the model might incur additional costs over time
    if deploy_model:
        sklearn_vertex_endpoint_name = deploy_model_to_endpoint_op(
            model_name=sklearn_vertex_model_name,
        ).outputs["endpoint_name"]

pipeline_func=train_tabular_classification_model_using_all_frameworks_pipeline

# %% Pipeline submission
if __name__ == '__main__':
    from google.cloud import aiplatform
    aiplatform.PipelineJob.from_pipeline_func(pipeline_func=pipeline_func).submit()
