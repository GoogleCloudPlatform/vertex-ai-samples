import argparse
import logging
import os
import pickle
import zipfile
from typing import List, Tuple

import pandas as pd
import wget
from google.cloud import storage
from google.cloud.logging import Client as LogClient
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def download_dataset_from_url(url: str) -> pd.DataFrame:
    """Downloads and unzips the dataset from `url` and reads it with pandas.

    Args:
        url (str, optional): URL to the dataset.
    """

    zip_filepath = wget.download(url, out=".")

    with zipfile.ZipFile(zip_filepath, "r") as zf:
        zf.extract(path=".", member="newsCorpora.csv")

    COLUMN_NAMES = ["id", "title", "url", "publisher",
                    "category", "story", "hostname", "timestamp"]

    return pd.read_csv(
        "newsCorpora.csv", delimiter="\t", names=COLUMN_NAMES, index_col=0
    )


def get_train_test_data(dataframe: pd.DataFrame, test_size: float = 0.2
) -> Tuple[List, List, List, List]:
    """Splits the news dataset into train and test features and labels.

    Args:
        news (pd.DataFrame): The dataset as pandas DataFrame.
        test_size (float): The size in percent of the test data.

    Returns:
        Tuple[List, List, List, List]: Tuple with train and test data
    """

    train, test = train_test_split(dataframe, test_size=test_size)

    x_train, y_train = train["title"].values, train["category"].values
    x_test, y_test = test["title"].values, test["category"].values

    return x_train, y_train, x_test, y_test


def export_model_to_gcs(fitted_pipeline: Pipeline, gcs_uri: str) -> str:
    """Exports trained pipeline to GCS

    Parameters:
            fitted_pipeline (sklearn.pipelines.Pipeline): the Pipeline object
                with data already fitted (trained pipeline object).
            gcs_uri (str): GCS path to store the trained pipeline
                i.e gs://example_bucket/training-job.
    Returns:
            export_path (str): Model GCS location
    """

    artifact_filename = 'model.pkl'

    # Save model artifact to local filesystem (doesn't persist)
    local_path = artifact_filename
    with open(local_path, 'wb') as model_file:
        pickle.dump(fitted_pipeline, model_file)

    # Upload model artifact to Cloud Storage
    storage_path = os.path.join(gcs_uri, artifact_filename)
    blob = storage.blob.Blob.from_string(storage_path, client=storage.Client())
    blob.upload_from_filename(local_path)


def export_evaluation_report_to_gcs(report: str, gcs_uri: str) -> None:
    """
    Exports training job report to GCS

    Parameters:
        report (str): Full report in text to sent to GCS
        gcs_uri (str): GCS path to store the report
            i.e gs://example_bucket/training-job
    """

    artifact_filename = 'report.txt'

    # Upload model artifact to Cloud Storage
    storage_path = os.path.join(gcs_uri, artifact_filename)
    blob = storage.blob.Blob.from_string(storage_path, client=storage.Client())
    blob.upload_from_string(report)


def train_and_score(X_train: List, y_train: List, X_test: List, y_test: List
) -> Tuple[Pipeline, float]:
    """Trains and cross-validates a text classifier pipeline.

    Args:
        X_train (List): Train features as list of strings.
        y_train (List): Train labels as list of strings.
        X_test (List): Test labels as list of strings.
        y_test (List): Test labels as list of strings.

    Returns:
        Tuple[Pipeline, float]: Fitted pipeline and mean accuracy.
    """

    pipeline = Pipeline([
                ("vectorizer", CountVectorizer()),
                ("tfidf", TfidfTransformer()),
                ("naivebayes", MultinomialNB()),
        ])

    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)

    return pipeline, score


# Define all the command line arguments your model can accept for training
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_url",
        help="Download url for the training data.",
        type=str
    )

    parser.add_argument(
        "--project_id",
        help="GCP project id for cloud logging.",
        type=str
    )

    args = parser.parse_args()
    arguments = args.__dict__

    # set up the GCP logger
    client = LogClient(project=arguments["project_id"])
    client.setup_logging(log_level=logging.INFO)
    logging.info("Starting custom training job.")

    # download the data from url
    logging.info("Downloading training data from: {}".format(arguments["dataset_url"]))
    dataframe = download_dataset_from_url(arguments["dataset_url"])
    train_test_data = get_train_test_data(dataframe)

    # train and cross validate
    logging.info("Training started ...")
    model, score = train_and_score(*train_test_data)
    logging.info(f"Training completed with model score: {score}")

    # export model to gcs
    _gcs_uri = os.environ["AIP_MODEL_DIR"]
    logging.info("Exporting model artifacts ...")
    export_model_to_gcs(model, _gcs_uri)
    export_evaluation_report_to_gcs(str(score), _gcs_uri)
    logging.info(f"Exported model artifacts to GCS bucket: {_gcs_uri}")
