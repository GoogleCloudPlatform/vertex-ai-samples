# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_preprocessing.py is the module for

  - ingest data
  - do simple preprocessing tasks
  - upload processed data to gcs
"""

# Libraries --------------------------------------------------------------------------------
import logging
import argparse
from pathlib import Path
import sys

try:
    from pyspark import SparkContext, SparkConf
    from pyspark.sql import SparkSession
except ImportError as error:
    print('WARN: Something wrong with pyspark library. Please check configuration settings!')
    print(error)

from pyspark.sql.types import StructType, DoubleType, StringType

# Variables --------------------------------------------------------------------------------
DATA_SCHEMA = (StructType()
               .add("label", StringType(), True)
               .add("loan_amount", StringType(), True)
               .add("loan_term", StringType(), True)
               .add("property_area", StringType(), True)
               .add("timestamp", StringType(), True)
               .add("entity_type_customer_id", StringType(), True)
               .add("feature_7", DoubleType(), True)
               .add("feature_3", DoubleType(), True)
               .add("feature_1", DoubleType(), True)
               .add("feature_9", DoubleType(), True)
               .add("feature_5", DoubleType(), True)
               .add("feature_0", DoubleType(), True)
               .add("feature_8", DoubleType(), True)
               .add("feature_4", DoubleType(), True)
               .add("feature_2", DoubleType(), True)
               .add("feature_6", DoubleType(), True)
               )

ENTITY_CUSTOMER_ID = 'entity_type_customer_id'
FEATURE_STORE_IDS = ['timestamp', 'entity_type_customer_id']
CATEGORICAL_VARIABLES = ['loan_term', 'property_area']
IDX_CATEGORICAL_FEATURES = [f'{col}_idx' for col in CATEGORICAL_VARIABLES]
TARGET = 'label'


# Helpers ----------------------------------------------------------------------------------

def set_logger():
    """
    Set logger for the module
    Returns:
        logger: logger object
    """
    fmt_pattern = "%(asctime)s — %(name)s — %(levelname)s —" "%(funcName)s:%(lineno)d — %(message)s"
    main_logger = logging.getLogger(__name__)
    main_logger.setLevel(logging.INFO)
    main_logger.propagate = False
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt_pattern)
    stream_handler.setFormatter(formatter)
    main_logger.addHandler(stream_handler)
    return main_logger


def get_args():
    """
    Get arguments from command line
    Returns:
        args: arguments from command line
    """
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        '--train-data-path',
        help='The GCS path of training sample',
        type=str,
        required=True)
    args_parser.add_argument(
        '--out-process-path',
        help='''
        The path to load processed data. 
        Format: 
        - locally: /path/to/dir
        - cloud: gs://bucket/path
        ''',
        type=str,
        required=True)
    return args_parser.parse_args()


# Main -------------------------------------------------------------------------------------

def main(logger, args):
    """
    Main function
    Args:
        logger: logger object
        args: arguments from command line
    Returns:
        None
    """
    # variables
    train_data_path = args.train_data_path
    output_data_path = args.out_process_path

    logger.info('initializing data preprocessing.')
    logger.info('start spark session.')

    spark = (SparkSession.builder
             .master("local[*]")
             .appName("spark go live")
             .config('spark.ui.port', '4050')
             .getOrCreate())
    try:
        logger.info(f'spark version: {spark.sparkContext.version}')
        logger.info('start ingesting data.')

        training_data_raw_df = (spark.read.option("header", True)
                                .option("delimiter", ',')
                                .schema(DATA_SCHEMA)
                                .csv(train_data_path)
                                .drop(*FEATURE_STORE_IDS))

        training_data_raw_df = training_data_raw_df.withColumn("label",
                                                               training_data_raw_df.label.cast('double'))
        training_data_raw_df = training_data_raw_df.withColumn("loan_amount",
                                                               training_data_raw_df.loan_amount.cast('double'))
        training_data_raw_df.show(truncate=False)

        logger.info(f'load prepared data to {output_data_path}.')
        if output_data_path.startswith('gs://'):
            training_data_raw_df.write.mode('overwrite').csv(str(output_data_path), header=True)
        else:
            output_file_path = Path(output_data_path)
            output_file_path.mkdir(parents=True, exist_ok=True)
            training_data_raw_df.write.mode('overwrite').csv(str(output_file_path), header=True)
    except RuntimeError as main_error:
        logger.error(main_error)
    else:
        logger.info('data preprocessing successfully completed!')
        return 0


if __name__ == "__main__":
    runtime_args = get_args()
    runtime_logger = set_logger()
    main(runtime_logger, runtime_args)