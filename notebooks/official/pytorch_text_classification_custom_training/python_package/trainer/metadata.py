#!/usr/bin/env python
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Task type can be either 'classification', 'regression', or 'custom'.
# This is based on the target feature in the dataset.
TASK_TYPE = 'classification'

# Dataset paths
    
TRAIN_DATA = "gs://cloud-samples-data/ai-platform-unified/datasets/text/happydb/happydb_train.csv"
TEST_DATA = "gs://cloud-samples-data/ai-platform-unified/datasets/text/happydb/happydb_test.csv"

# pre-trained model name
PRETRAINED_MODEL_NAME = 'bert-base-cased'

# List of the class values (labels) in a classification dataset.
TARGET_LABELS = {"leisure": 0, "exercise":1, "enjoy_the_moment":2, "affection":3,"achievement":4, "nature":5, "bonding":6}


# maximum sequence length
MAX_SEQ_LENGTH = 128