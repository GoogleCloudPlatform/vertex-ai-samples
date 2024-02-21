#!/bin/bash

# Initialize variables.
training_job_name=""
gcs_experiment_path=""
gin_config_file="configs/360.gin"
factor=4
max_training_steps=25000

# Parse named arguments.
while [[ $# -gt 0 ]]; do
  case $1 in
    -training_job_name)
      training_job_name="$2"
      shift # past argument
      shift # past value
      ;;
    -gcs_experiment_path)
      gcs_experiment_path="$2"
      shift # past argument
      shift # past value
      ;;
    -gin_config_file)
      gin_config_file="$2"
      shift # past argument
      shift # past value
      ;;
    -factor)
      factor="$2"
      if ! [[ $factor =~ ^[0-9]+$ ]]; then
        echo "Error: -factor must be an integer."
        exit 1
      fi
      shift # past argument
      shift # past value
      ;;
    -max_training_steps)
      max_training_steps="$2"
      if ! [[ $max_training_steps =~ ^[0-9]+$ ]]; then
        echo "Error: -max_training_steps must be an integer."
        exit 1
      fi
      shift # past argument
      shift # past value
      ;;
    *)    # unknown option
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

# Function to create a directory if it doesn't exist.
create_dir_if_not_exists() {
  local dir_path=$1
  if [[ ! -d "$dir_path" ]]; then
    echo "Creating folder: $dir_path"
    mkdir "$dir_path"
  else
    echo "Folder $dir_path already exists."
  fi
}

# Extract folder names and paths.
scene_folder_name=$(basename "${gcs_experiment_path}")
local_dataset_path="local_dataset"
local_experiment_path="exp"
DATASET_PATH="$local_experiment_path/$scene_folder_name/data"
EXPERIMENT=$scene_folder_name

# Create necessary directories.
create_dir_if_not_exists "$local_dataset_path"
create_dir_if_not_exists "$local_experiment_path"
create_dir_if_not_exists "$local_experiment_path/$scene_folder_name"

# Copy experiment from GCS bucket to local.
gsutil -m cp -r "${gcs_experiment_path}/data" "$local_experiment_path/$scene_folder_name" || exit 1

echo "GCS Experiment: $gcs_experiment_path"
echo "Gin Config File: $gin_config_file"
echo "Factor: $factor"
echo "Scene: $scene_folder_name"
echo "Local Dataset: $DATASET_PATH"
echo "Local Experiment: $EXPERIMENT"

accelerate launch train.py --gin_configs="$gin_config_file" \
  --gin_bindings="Config.data_dir = '${DATASET_PATH}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
  --gin_bindings="Config.factor = ${factor}" \
  --gin_bindings="Config.max_steps = ${max_training_steps}"

gsutil -m rm -r "${gcs_experiment_path}/checkpoints/${training_job_name}"
gsutil -m cp -r "$local_experiment_path/$scene_folder_name/config.gin" "${gcs_experiment_path}/${training_job_name}_config.gin"
gsutil -m cp -r "$local_experiment_path/$scene_folder_name/checkpoints/*/*" "${gcs_experiment_path}/checkpoints/${training_job_name}"