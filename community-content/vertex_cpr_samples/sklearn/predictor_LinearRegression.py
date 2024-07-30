import numpy as np
import os
import pickle

from google.cloud.aiplatform.constants import prediction
from google.cloud.aiplatform.utils import prediction_utils
from google.cloud.aiplatform.prediction.predictor import Predictor
from sklearn.datasets import make_blobs
from sklearn.linear_model import LinearRegression 

class LinearRegressionPredictor(Predictor):

    def __init__(self):
        return

    def load(self, artifacts_uri: str) -> None:
        prediction_utils.download_model_artifacts(artifacts_uri)
        if os.path.exists(prediction.MODEL_FILENAME_PKL):
            self._model = pickle.load(open(prediction.MODEL_FILENAME_PKL, "rb"))
        else:
            self._model = LogisticRegression()
        X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
        self._model.fit(X, y)

    def preprocess(self, prediction_input: dict) -> np.ndarray:
        instances = prediction_input["instances"]
        return np.asarray(instances)

    def predict(self, instances: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(instances)

    def postprocess(self, prediction_results: np.ndarray) -> dict:
        return {"predictions": prediction_results.tolist()}
