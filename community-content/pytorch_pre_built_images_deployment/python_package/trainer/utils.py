# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.\n",
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import datetime

from google.cloud import storage

from transformers import AutoTokenizer
from datasets import load_dataset, load_metric, ReadInstruction
from trainer import metadata


def preprocess_function(examples):
    tokenizer = AutoTokenizer.from_pretrained(
        metadata.PRETRAINED_MODEL_NAME,
        use_fast=True,
    )

    # Tokenize the texts
    tokenizer_args = (
        (examples['text'],) 
    )
    result = tokenizer(*tokenizer_args, 
                       padding='max_length', 
                       max_length=metadata.MAX_SEQ_LENGTH, 
                       truncation=True)

    # TEMP: We can extract this automatically but Unique method of the dataset
    # is not reporting the label -1 which shows up in the pre-processing
    # Hence the additional -1 term in the dictionary
    label_to_id = metadata.TARGET_LABELS
    
    # Map labels to IDs (not necessary for GLUE tasks)
    if label_to_id is not None and "label" in examples:
        result["label"] = [label_to_id[l] for l in examples["label"]]

    return result


def load_data(args):
    """Loads the data into two different data loaders. (Train, Test)

        Args:
            args: arguments passed to the python script
    """
    # Dataset loading repeated here to make this cell idempotent
    # Since we are over-writing datasets variable
    dataset = load_dataset(metadata.DATASET_NAME)

    dataset = dataset.map(preprocess_function, 
                          batched=True, 
                          load_from_cache_file=True)

    train_dataset, test_dataset = dataset["train"], dataset["test"]

    return train_dataset, test_dataset


def save_model(args):
    """Saves the model to Google Cloud Storage or local file system

    Args:
      args: contains name for saved model.
    """
    scheme = 'gs://'
    if args.job_dir.startswith(scheme):
        job_dir = args.job_dir.split("/")
        bucket_name = job_dir[2]
        object_prefix = "/".join(job_dir[3:]).rstrip("/")

        if object_prefix:
            model_path = '{}/{}'.format(object_prefix, args.model_name)
        else:
            model_path = '{}'.format(args.model_name)

        bucket = storage.Client().bucket(bucket_name)    
        local_path = os.path.join("/tmp", args.model_name)
        files = [f for f in os.listdir(local_path) if os.path.isfile(os.path.join(local_path, f))]
        for file in files:
            local_file = os.path.join(local_path, file)
            blob = bucket.blob("/".join([model_path, file]))
            blob.upload_from_filename(local_file)
        print(f"Saved model files in gs://{bucket_name}/{model_path}")
    else:
        print(f"Saved model files at {os.path.join('/tmp', args.model_name)}")
        print(f"To save model files in GCS bucket, please specify job_dir starting with gs://")
        
