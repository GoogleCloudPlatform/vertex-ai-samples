# python3 -m pip install "kfp<2.0.0" "google-cloud-aiplatform>=1.16.0" --upgrade --quiet
from kfp import components

# %% Loading components
download_from_gcs_op = components.load_component_from_url("https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/399405402d95f4a011e2d2e967c96f8508ba5688/community-content/pipeline_components/google-cloud/storage/download/component.yaml")
select_columns_using_Pandas_on_CSV_data_op = components.load_component_from_url("https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/399405402d95f4a011e2d2e967c96f8508ba5688/community-content/pipeline_components/pandas/Select_columns/in_CSV_format/component.yaml")
fill_all_missing_values_using_Pandas_on_CSV_data_op = components.load_component_from_url("https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/399405402d95f4a011e2d2e967c96f8508ba5688/community-content/pipeline_components/pandas/Fill_all_missing_values/in_CSV_format/component.yaml")
split_rows_into_subsets_op = components.load_component_from_url("https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/399405402d95f4a011e2d2e967c96f8508ba5688/community-content/pipeline_components/dataset_manipulation/Split_rows_into_subsets/in_CSV/component.yaml")
create_fully_connected_tensorflow_network_op = components.load_component_from_url("https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/399405402d95f4a011e2d2e967c96f8508ba5688/community-content/pipeline_components/tensorflow/Create_fully_connected_network/component.yaml")
train_model_using_Keras_on_CSV_op = components.load_component_from_url("https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/399405402d95f4a011e2d2e967c96f8508ba5688/community-content/pipeline_components/tensorflow/Train_model_using_Keras/on_CSV/component.yaml")
predict_with_TensorFlow_model_on_CSV_data_op = components.load_component_from_url("https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/399405402d95f4a011e2d2e967c96f8508ba5688/community-content/pipeline_components/tensorflow/Predict/on_CSV/component.yaml")
upload_Tensorflow_model_to_Google_Cloud_Vertex_AI_op = components.load_component_from_url("https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/399405402d95f4a011e2d2e967c96f8508ba5688/community-content/pipeline_components/google-cloud/Vertex_AI/Models/Upload_Tensorflow_model/component.yaml")
deploy_model_to_endpoint_op = components.load_component_from_url("https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/399405402d95f4a011e2d2e967c96f8508ba5688/community-content/pipeline_components/google-cloud/Vertex_AI/Models/Deploy_to_endpoint/component.yaml")

# %% Pipeline definition
def train_tabular_regression_model_using_Tensorflow_pipeline():
    dataset_gcs_uri = "gs://ml-pipeline-dataset/Chicago_taxi_trips/chicago_taxi_trips_2019-01-01_-_2019-02-01_limit=10000.csv"
    feature_columns = ["trip_seconds", "trip_miles", "pickup_community_area", "dropoff_community_area", "fare", "tolls", "extras"]  # Excluded "trip_total"
    label_column = "tips"
    training_set_fraction = 0.8
    # Deploying the model might incur additional costs over time
    deploy_model = False

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

    split_task = split_rows_into_subsets_op(
        table=dataset,
        fraction_1=training_set_fraction,
    )
    training_data = split_task.outputs["split_1"]
    testing_data = split_task.outputs["split_2"]

    network = create_fully_connected_tensorflow_network_op(
        input_size=len(feature_columns),
        # Optional:
        hidden_layer_sizes=[10],
        activation_name="elu",
        # output_activation_name=None,
        # output_size=1,
    ).outputs["model"]

    model = train_model_using_Keras_on_CSV_op(
        training_data=training_data,
        model=network,
        label_column_name=label_column,
        # Optional:
        #loss_function_name="mean_squared_error",
        number_of_epochs=10,
        #learning_rate=0.1,
        #optimizer_name="Adadelta",
        #optimizer_parameters={},
        #batch_size=32,
        metric_names=["mean_absolute_error"],
        #random_seed=0,
    ).outputs["trained_model"]

    predictions = predict_with_TensorFlow_model_on_CSV_data_op(
        dataset=testing_data,
        model=model,
        # label_column_name needs to be set when doing prediction on a dataset that has labels
        label_column_name=label_column,
        # Optional:
        # batch_size=1000,
    ).outputs["predictions"]

    vertex_model_name = upload_Tensorflow_model_to_Google_Cloud_Vertex_AI_op(
        model=model,
    ).outputs["model_name"]

    # Deploying the model might incur additional costs over time
    if deploy_model:
        vertex_endpoint_name = deploy_model_to_endpoint_op(
            model_name=vertex_model_name,
        ).outputs["endpoint_name"]

pipeline_func=train_tabular_regression_model_using_Tensorflow_pipeline

# %% Pipeline submission
if __name__ == '__main__':
    from google.cloud import aiplatform
    aiplatform.PipelineJob.from_pipeline_func(pipeline_func=pipeline_func).submit()
