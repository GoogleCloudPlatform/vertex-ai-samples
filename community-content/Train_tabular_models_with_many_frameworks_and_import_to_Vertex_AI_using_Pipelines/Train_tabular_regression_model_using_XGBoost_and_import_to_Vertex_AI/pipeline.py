# python3 -m pip install "kfp<2.0.0" "google-cloud-aiplatform>=1.16.0" --upgrade --quiet
from kfp import components

# %% Loading components
download_from_gcs_op = components.load_component_from_url("https://raw.githubusercontent.com/Ark-kun/pipeline_components/27a5ea25e849c9e8c0cb6ed65518bc3ece259aaf/components/google-cloud/storage/download/workaround_for_buggy_KFPv2_compiler/component.yaml")
select_columns_using_Pandas_on_CSV_data_op = components.load_component_from_url("https://raw.githubusercontent.com/Ark-kun/pipeline_components/8c78aae096806cff3bc331a40566f42f5c3e9d4b/components/pandas/Select_columns/in_CSV_format/component.yaml")
fill_all_missing_values_using_Pandas_on_CSV_data_op = components.load_component_from_url("https://raw.githubusercontent.com/Ark-kun/pipeline_components/23405971f5f16a41b16c343129b893c52e4d1d48/components/pandas/Fill_all_missing_values/in_CSV_format/component.yaml")
split_rows_into_subsets_op = components.load_component_from_url("https://raw.githubusercontent.com/Ark-kun/pipeline_components/daae5a4abaa35e44501818b1534ed7827d7da073/components/dataset_manipulation/Split_rows_into_subsets/in_CSV/component.yaml")
train_XGBoost_model_on_CSV_op = components.load_component_from_url("https://raw.githubusercontent.com/Ark-kun/pipeline_components/58d3a47f904f32a64af8403330ba7e2134cae46d/components/XGBoost/Train/component.yaml")
xgboost_predict_on_CSV_op = components.load_component_from_url("https://raw.githubusercontent.com/Ark-kun/pipeline_components/4694ec97baccf59284c2a1db4aa2250c22291eab/components/XGBoost/Predict/component.yaml")
upload_XGBoost_model_to_Google_Cloud_Vertex_AI_op = components.load_component_from_url("https://raw.githubusercontent.com/Ark-kun/pipeline_components/c6a8b67d1ada2cc17665c99ff6b410df588bee28/components/google-cloud/Vertex_AI/Models/Upload_XGBoost_model/workaround_for_buggy_KFPv2_compiler/component.yaml")
deploy_model_to_endpoint_op = components.load_component_from_url("https://raw.githubusercontent.com/Ark-kun/pipeline_components/27a5ea25e849c9e8c0cb6ed65518bc3ece259aaf/components/google-cloud/Vertex_AI/Models/Deploy_to_endpoint/workaround_for_buggy_KFPv2_compiler/component.yaml")

# %% Pipeline definition
def train_tabular_regression_model_using_XGBoost_pipeline():
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

    model = train_XGBoost_model_on_CSV_op(
        training_data=training_data,
        label_column_name=label_column,
        # Optional:
        #starting_model=None,
        #num_iterations=10,
        #booster_params={},
        #objective="reg:squarederror",
        #booster="gbtree",
        #learning_rate=0.3,
        #min_split_loss=0,
        #max_depth=6,
    ).outputs["model"]

    # Predicting on the testing data
    predictions = xgboost_predict_on_CSV_op(
        data=testing_data,
        model=model,
        # label_column needs to be set when doing prediction on a dataset that has labels
        label_column_name=label_column,
    ).outputs["predictions"]

    vertex_model_name = upload_XGBoost_model_to_Google_Cloud_Vertex_AI_op(
        model=model,
    ).outputs["model_name"]

    # Deploying the model might incur additional costs over time
    if deploy_model:
        vertex_endpoint_name = deploy_model_to_endpoint_op(
            model_name=vertex_model_name,
        ).outputs["endpoint_name"]

pipeline_func = train_tabular_regression_model_using_XGBoost_pipeline

# %% Pipeline submission
if __name__ == '__main__':
    from google.cloud import aiplatform
    aiplatform.PipelineJob.from_pipeline_func(pipeline_func=pipeline_func).submit()
