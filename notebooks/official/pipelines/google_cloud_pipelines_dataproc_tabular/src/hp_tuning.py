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
hp_model_tuning.py is the module for hypertune the spark pipeline
"""

# Libraries --------------------------------------------------------------------------------
import logging
import sys
import argparse
from os import environ
from datetime import datetime
from pathlib import Path as path
import tempfile
from urllib.parse import urlparse, urljoin
import json

try:
    from pyspark import SparkContext, SparkConf
    from pyspark.sql import SparkSession
except ImportError as e:
    print('WARN: Something wrong with pyspark library. Please check configuration settings!')
    print(e)

from pyspark.sql.types import StructType, DoubleType, StringType
from pyspark.sql.functions import col, udf
from pyspark.sql.functions import round as spark_round
from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline

from google.cloud import storage

# Variables --------------------------------------------------------------------------------

# Data schema
DATA_SCHEMA = (StructType()
               .add("label", DoubleType(), True)
               .add("loan_amount", DoubleType(), True)
               .add("loan_term", StringType(), True)
               .add("property_area", StringType(), True)
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

# Training
TARGET = 'label'
CATEGORICAL_VARIABLES = ['loan_term', 'property_area']
IDX_CATEGORICAL_FEATURES = [f'{col}_idx' for col in CATEGORICAL_VARIABLES]
REAL_TIME_FEATURES_VECTOR = 'real_time_features_vector'
REAL_TIME_FEATURES = 'real_time_features'
FEATURES_SELECTED = ['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5',
                     'feature_6', 'feature_7', 'feature_8', 'feature_9', 'real_time_features']
FEATURES = 'features'
RANDOM_SEED = 8
RANDOM_QUOTAS = [0.8, 0.2]
MAX_DEPTH = [5, 10, 15]
MAX_BINS = [24, 32, 40]
N_TREES = [25, 30, 35]
N_FOLDS = 5


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
        '--train-path',
        help='''
        The GCS path of training data'
        Format: 
        - locally: /path/to/dir
        - cloud: gs://bucket/path
        ''',
        type=str,
        required=False)
    args_parser.add_argument(
        '--model-path',
        help='''
        The GCS path to store the trained model. 
        Format: 
        - locally: /path/to/dir
        - cloud: gs://bucket/path
        ''',
        type=str,
        required=False)
    args_parser.add_argument(
        '--metrics-path',
        help='''
        The GCS path to store the metrics of model. 
        Format: 
        - locally: /path/to/dir
        - cloud: gs://bucket/path
        ''',
        type=str,
        required=True)
    return args_parser.parse_args()


def build_preprocessing_components():
    """
    Build preprocessing components
    Returns:
        preprocessing_components: preprocessing components
    """
    loan_term_indexer = StringIndexer(inputCol=CATEGORICAL_VARIABLES[0], outputCol=IDX_CATEGORICAL_FEATURES[0],
                                      stringOrderType='frequencyDesc', handleInvalid='keep')
    property_area_indexer = StringIndexer(inputCol=CATEGORICAL_VARIABLES[1], outputCol=IDX_CATEGORICAL_FEATURES[1],
                                          stringOrderType='frequencyDesc', handleInvalid='keep')
    data_preprocessing_stages = [loan_term_indexer, property_area_indexer]
    return data_preprocessing_stages


def build_feature_engineering_components():
    """
    Build feature engineering components
    Returns:
        feature_engineering_components: feature engineering components
    """
    feature_engineering_stages = []
    realtime_vector_assembler = VectorAssembler(inputCols=IDX_CATEGORICAL_FEATURES, outputCol=REAL_TIME_FEATURES_VECTOR)
    realtime_scaler = StandardScaler(inputCol=REAL_TIME_FEATURES_VECTOR, outputCol=REAL_TIME_FEATURES)
    features_vector_assembler = VectorAssembler(inputCols=FEATURES_SELECTED, outputCol=FEATURES)
    feature_engineering_stages.extend((realtime_vector_assembler,
                                       realtime_scaler,
                                       features_vector_assembler))
    return feature_engineering_stages


def build_training_model_component():
    """
    Build training model component
    Returns:
        training_model_component: training model component
    """
    model_training_stage = []
    rfor = RandomForestClassifier(featuresCol=FEATURES, labelCol=TARGET, seed=RANDOM_SEED)
    model_training_stage.append(rfor)
    return model_training_stage


def build_hp_pipeline(data_preprocessing_stages, feature_engineering_stages, model_training_stage):
    """
    Build hyperparameter pipeline
    Args:
        data_preprocessing_stages: preprocessing components
        feature_engineering_stages: feature engineering components
        model_training_stage: training model component
    Returns:
        hp_pipeline: hyperparameter pipeline
    """
    pipeline = Pipeline(stages=data_preprocessing_stages + feature_engineering_stages + model_training_stage)
    params_grid = (ParamGridBuilder()
                   .addGrid(model_training_stage[0].maxDepth, MAX_DEPTH)
                   .addGrid(model_training_stage[0].maxBins, MAX_BINS)
                   .addGrid(model_training_stage[0].numTrees, N_TREES)
                   .build())
    evaluator = BinaryClassificationEvaluator(labelCol=TARGET)
    cross_validator = CrossValidator(estimator=pipeline,
                                     estimatorParamMaps=params_grid,
                                     evaluator=evaluator,
                                     numFolds=N_FOLDS)
    return cross_validator


def get_true_score_prediction(predictions, target):
    """
    Get true score and prediction
    Args:
        predictions: predictions
        target: target column
    Returns:
        roc_dict: a dict of roc values for each class
    """

    split1_udf = udf(lambda value: value[1].item(), DoubleType())
    roc_dataset = predictions.select(col(target).alias('true'),
                                     spark_round(split1_udf('probability'), 5).alias('score'),
                                     'prediction')
    roc_df = roc_dataset.toPandas()
    roc_dict = roc_df.to_dict(orient='list')
    return roc_dict


def get_metrics(predictions, target, mode):
    """
    Get metrics
    Args:
        predictions: predictions
        target: target column
        mode: train or test
    Returns:
        metrics: metrics
    """
    metric_labels = ['area_roc', 'area_prc', 'accuracy', 'f1', 'precision', 'recall']
    metric_cols = ['true', 'score', 'prediction']
    metric_keys = [f'{mode}_{ml}' for ml in metric_labels] + metric_cols

    bc_evaluator = BinaryClassificationEvaluator(labelCol=target)
    mc_evaluator = MulticlassClassificationEvaluator(labelCol=target)

    # areas, acc, f1, prec, rec
    metric_values = []
    area_roc = round(bc_evaluator.evaluate(predictions, {bc_evaluator.metricName: 'areaUnderROC'}), 5)
    area_prc = round(bc_evaluator.evaluate(predictions, {bc_evaluator.metricName: 'areaUnderPR'}), 5)
    acc = round(mc_evaluator.evaluate(predictions, {mc_evaluator.metricName: "accuracy"}), 5)
    f1 = round(mc_evaluator.evaluate(predictions, {mc_evaluator.metricName: "f1"}), 5)
    prec = round(mc_evaluator.evaluate(predictions, {mc_evaluator.metricName: "weightedPrecision"}), 5)
    rec = round(mc_evaluator.evaluate(predictions, {mc_evaluator.metricName: "weightedRecall"}), 5)

    # true, score, prediction
    roc_dict = get_true_score_prediction(predictions, target)
    true = roc_dict['true']
    score = roc_dict['score']
    pred = roc_dict['prediction']

    metric_values.extend((area_roc, area_prc, acc, f1, prec, rec, true, score, pred))
    metrics = dict(zip(metric_keys, metric_values))

    return metrics


def upload_file(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)


def write_metrics(bucket_name, metrics, destination, dir='/tmp'):
    temp_dir = tempfile.TemporaryDirectory(dir=dir)
    temp_metrics_file_path = str(path(temp_dir.name) / path(destination).name)
    with open(temp_metrics_file_path, 'w') as temp_file:
        json.dump(metrics, temp_file)
    upload_file(bucket_name, temp_metrics_file_path, destination)
    temp_dir.cleanup()


# Main -------------------------------------------------------------------------------------

def main(logger, args):
    """
    Main function
    Args:
        logger: logger
        args: args
    Returns:
        None
    """
    train_path = args.train_path
    model_path = args.model_path
    metrics_path = args.metrics_path

    try:
        logger.info('initializing pipeline training.')
        logger.info('start spark session.')
        spark = (SparkSession.builder
                 .master("local[*]")
                 .appName("spark go live")
                 .config('spark.ui.port', '4050')
                 .config('spark.jars.packages', 'ml.combust.mleap:mleap-runtime_2.12:0.19.0')
                 .config('spark.jars.packages', 'ml.combust.mleap:mleap-base_2.12:0.19.0')
                 .config('spark.jars.packages', 'ml.combust.mleap:mleap-spark_2.12:0.19.0')
                 .config('spark.jars.packages', 'ml.combust.mleap:mleap-spark-extension_2.12:0.19.0')
                 .getOrCreate())
        logger.info(f'spark version: {spark.sparkContext.version}')
        logger.info('start building pipeline.')
        preprocessing_stages = build_preprocessing_components()
        feature_engineering_stages = build_feature_engineering_components()
        model_training_stage = build_training_model_component()
        pipeline_cross_validator = build_hp_pipeline(preprocessing_stages, feature_engineering_stages,
                                                     model_training_stage)
        logger.info(f'load train data from {train_path}.')
        if train_path.startswith('bq://'):
            raw_data = spark.read.format('bigquery') \
                .option('table', train_path.replace('bq://', '')) \
                .load()
        else:
            raw_data = (spark.read.format('csv')
                        .option("header", "true")
                        .schema(DATA_SCHEMA)
                        .load(train_path))
        logger.info(f'fit model pipeline.')
        train, test = raw_data.randomSplit(RANDOM_QUOTAS, seed=RANDOM_SEED)
        pipeline_model = pipeline_cross_validator.fit(train)
        predictions = pipeline_model.transform(test)
        metrics = get_metrics(predictions, TARGET, 'test')
        for m, v in metrics.items():
            print(f'{m}: {v}')

        logger.info(f'load model pipeline in {model_path}.')
        if model_path.startswith('gs://'):
            pipeline_model.write().overwrite().save(model_path)
        else:
            path(model_path).mkdir(parents=True, exist_ok=True)
            pipeline_model.write().overwrite().save(model_path)

        logger.info(f'Upload metrics under {metrics_path}.')
        if metrics_path.startswith('gs://'):
            bucket = urlparse(model_path).netloc
            metrics_file_path = urlparse(metrics_path).path.strip('/')
            write_metrics(bucket, metrics, metrics_file_path)
        else:
            metrics_version_path = path(metrics_path).parents[0]
            metrics_version_path.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, 'w') as json_file:
                json.dump(metrics, json_file)
            json_file.close()
    except RuntimeError as main_error:
        logger.error(main_error)
    else:
        logger.info('model pipeline training successfully completed!')
        return 0


if __name__ == "__main__":
    runtime_args = get_args()
    runtime_logger = set_logger()
    main(runtime_logger, runtime_args)