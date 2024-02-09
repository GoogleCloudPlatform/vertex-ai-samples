#!/bin/bash
# This script runs rendering for ZipNeRF given an experiment folder
# from a GCS bucket with colmap dataset.

# Initialize associative array for arguments.
declare -A args

# vv-docker:google3-begin(internal)
# TODO(b/311468174): Pass gin config file from gcs bucket.
# vv-docker:google3-end
# Function to parse named arguments.
parse_args() {
  while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
      -gcs_experiment_path|-gin_config_file|-gcs_keyframes_file)
        args[$key]="$2"
        shift # past argument
        shift # past value
        ;;
      -training_job_name)
        training_job_name="$2"
        shift # past argument
        shift # past value
        ;;
      -rendering_job_name)
        rendering_job_name="$2"
        shift # past argument
        shift # past value
        ;;
      -render_path_frames|-factor|-render_video_fps)
        args[$key]="$2"
        if ! [[ ${args[$key]} =~ ^[0-9]+$ ]]; then
          echo "Error: $key must be an integer."
          exit 1
        fi
        shift # past argument
        shift # past value
        ;;
      *)
        echo "Unknown option: $1" >&2
        exit 1
        ;;
    esac
  done
}

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

# Function to launch rendering.
launch_rendering() {
  local keyframes_file=$1
  local render_bindings=(
    "--gin_configs=${args[-gin_config_file]}"
    "--gin_bindings=Config.data_dir='${DATASET_PATH}'"
    "--gin_bindings=Config.exp_name='${EXPERIMENT}'"
    "--gin_bindings=Config.render_path=True"
    "--gin_bindings=Config.render_path_frames=${args[-render_path_frames]}"
    "--gin_bindings=Config.render_video_fps=${args[-render_video_fps]}"
    "--gin_bindings=Config.factor=${args[-factor]}"
  )

  if [[ -n $keyframes_file ]]; then
    render_bindings+=("--gin_bindings=Config.render_spline_keyframes='${keyframes_file}'")
  fi

  accelerate launch render.py "${render_bindings[@]}"
}

# Parse arguments.
parse_args "$@"

# Extract folder names and paths.
scene_folder_name=$(basename "${args[-gcs_experiment_path]}")
local_dataset_path="local_dataset"
local_experiment_path="exp"
exp_folder_name=$(basename "${args[-gcs_experiment_path]}")
DATASET_PATH="$local_experiment_path/$exp_folder_name/data"
CHECKPOINTS_PATH="$local_experiment_path/$exp_folder_name/checkpoints"
OUTPUT_RENDER_PATH="$local_experiment_path/$scene_folder_name/render"
EXPERIMENT=$exp_folder_name

# Create necessary directories.
create_dir_if_not_exists "$local_dataset_path"
create_dir_if_not_exists "$local_experiment_path"
create_dir_if_not_exists "$local_experiment_path/$exp_folder_name"
create_dir_if_not_exists "$CHECKPOINTS_PATH"

# Create the file log_render.txt in the exp folder.
touch "$local_experiment_path/$exp_folder_name/log_render.txt"

# Copy experiment from GCS bucket to local
gsutil -m cp -r "${args[-gcs_experiment_path]}/data" "$local_experiment_path/$exp_folder_name" || exit 1
gsutil -m cp -r "${args[-gcs_experiment_path]}/checkpoints/${training_job_name}/*" "$CHECKPOINTS_PATH" || exit 1

# Check and copy keyframes file.
if [[ -n ${args[-gcs_keyframes_file]} ]]; then
  keyframes_file_basename=$(basename "${args[-gcs_keyframes_file]}")
  local_keyframes_file="$local_dataset_path/$keyframes_file_basename"
  gsutil cp "${args[-gcs_keyframes_file]}" "$local_keyframes_file" || exit 1
  echo "Local keyframe file: $local_keyframes_file"
  launch_rendering "$local_keyframes_file"
else
  launch_rendering ""
fi

# Copy rendered data back to GCS.
gsutil -m cp -r "$OUTPUT_RENDER_PATH" "${args[-gcs_experiment_path]}/render/${rendering_job_name}"