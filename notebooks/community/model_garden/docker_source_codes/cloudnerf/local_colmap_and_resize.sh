#!/bin/bash
# This script runs colmap for scale invariant feature (SIFT) extraction and
# matching to map camera extrinsics and intrinsics values for ZipNeRF,
# given a folder of images and videos
# from a GCS bucket. It uses ffmepg to extract an image from a video at
# 1fps. The folder can contain images or videos. If both images and videos
# are present, the extracted frames from the videos is added to the images
# to create the final combined image dataset.
# vv-docker:google3-begin(internal)
# TODO(b/314042136): Specify cloudnerf colmap fps.
# vv-docker:google3-end

# Initialize variables.
use_gpu=1  # Default to 1 (assuming the docker is run on a machine with GPU)
gcs_dataset_path=""
gcs_experiment_path=""
camera=""

# This loop processes command-line arguments for configuring the container.
# It supports arguments for GPU usage, dataset and experiment paths,
# and camera type.
while [[ $# -gt 0 ]]; do
  case $1 in
    -use_gpu)
      use_gpu="$2"
      if ! [[ $use_gpu =~ ^[0-9]+$ ]]; then
        echo "Error: -use_gpu must be an integer."
        exit 1
      fi
      shift # past argument
      shift # past value
      ;;
    -gcs_dataset_path)
      gcs_dataset_path="$2"
      shift # past argument
      shift # past value
      ;;
    -gcs_experiment_path)
      gcs_experiment_path="$2"
      shift # past argument
      shift # past value
      ;;
    -camera)
      camera="$2"
      if [[ $camera != "OPENCV" && $camera != "OPENCV_FISHEYE" ]]; then
        echo "Error: -camera must be either 'OPENCV' or 'OPENCV_FISHEYE'."
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

local_folder="dataset_content"
images_folder="dataset_images"
images_subfolder="images"
output_folder="$images_folder/$images_subfolder"

# Create the local folder if it doesn't exist
mkdir -p "$local_folder"
mkdir -p "$output_folder"

# Download the content from the GCS URI
gsutil -m cp -r "$gcs_dataset_path"/* "$local_folder/"

# Process files in the local folder
for file in "$local_folder"/*; do
  if [[ -f "$file" ]]; then
    # Check if the file is an image (e.g., jpg, png, etc.)
    if file --mime-type "$file" | grep -q "image"; then
      # Copy the image to the "images" subfolder within the "dataset_images" folder
      cp "$file" "$output_folder/$(basename "$file")"
    elif file --mime-type "$file" | grep -q "video"; then
      # Use FFmpeg to extract an image every 30 frames from the video
      ffmpeg -i "$file" -vf "select='not(mod(n,30))'" "$output_folder/$(basename "$file" ."${file##*.}")_%03d.jpg"
    else
      echo "Skipping unsupported file: $file"
    fi
  fi
done

# Run COLMAP Feature extraction
colmap feature_extractor \
    --database_path "$local_folder"/database.db \
    --image_path "$output_folder" \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model "$camera" \
    --SiftExtraction.use_gpu "$use_gpu"

# Run COLMAP Feature matching
colmap exhaustive_matcher \
    --database_path "$local_folder"/database.db \
    --SiftMatching.use_gpu "$use_gpu"

# Bundle adjustment. The default Mapper tolerance is unnecessarily large,
# decreasing it speeds up bundle adjustment steps.
mkdir -p "$local_folder"/sparse
colmap mapper \
    --database_path "$local_folder"/database.db \
    --image_path "$output_folder" \
    --output_path "$local_folder"/sparse \
    --Mapper.ba_global_function_tolerance=0.000001

# Downsample images at 1/2, 1/4, 1/8 scales. Save feature matching to
# sqlite database.
# All input and output images:
#   $gcs_dataset_path
#   $gcs_experiment_path/data/images
# Downsampled output images:
#   $gcs_experiment_path/data/images_2/
#   $gcs_experiment_path/data/images_4/
#   $gcs_experiment_path/data/images_8/
# COLMAP sparse reconstruction files: project.ini, images.bin,
# cameras.bin, points3D.bin
#   $gcs_experiment_path/data/sparse/0/
cp -r "$output_folder" "$images_folder"/images_2
pushd "$images_folder"/images_2
ls | xargs -P 8 -I {} mogrify -resize 50% {}
popd
gsutil -m cp -r "$images_folder"/images_2/*  "$gcs_experiment_path"/data/images_2

cp -r "$output_folder" "$images_folder"/images_4
pushd "$images_folder"/images_4
ls | xargs -P 8 -I {} mogrify -resize 25% {}
popd
gsutil -m cp -r "$images_folder"/images_4/*  "$gcs_experiment_path"/data/images_4

cp -r "$output_folder" "$images_folder"/images_8
pushd "$images_folder"/images_8
ls | xargs -P 8 -I {} mogrify -resize 12.5% {}
popd
gsutil -m cp "$images_folder"/images_8/*  "$gcs_experiment_path"/data/images_8

# Copy images and sparse reconstruction files to gcs experiment folder.
gsutil -m cp "$images_folder"/images/*  "$gcs_experiment_path"/data/images
gsutil -m cp -r "$local_folder"/sparse  "$gcs_experiment_path"/data
gsutil -m cp "$local_folder"/database.db  "$gcs_experiment_path"/data

echo "Processing complete."