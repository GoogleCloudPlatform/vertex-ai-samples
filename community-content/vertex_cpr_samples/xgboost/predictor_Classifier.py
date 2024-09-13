import os
import numpy as np
import pickle
import xgboost as xgb

from google.cloud.aiplatform.constants import prediction
from google.cloud.aiplatform.utils import prediction_utils
from google.cloud.aiplatform.prediction.predictor import Predictor
from sklearn.datasets import make_blobs
from xgboost import XGBClassifier


class ClassifierPredictor(Predictor):

    def __init__(self):
        return

    def load(self, artifacts_uri: str) -> None:
        prediction_utils.download_model_artifacts(artifacts_uri)
        if os.path.exists(prediction.MODEL_FILENAME_PKL):
            booster = pickle.load(open(prediction.MODEL_FILENAME_PKL, "rb"))
        else:
            X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
            model = XGBClassifier()
            model.fit(X, y)
            booster = model.get_booster()
        self._booster = booster

    def preprocess(self, prediction_input: dict) -> xgb.DMatrix:
        instances = prediction_input["instances"]
        return xgb.DMatrix(instances)

    def predict(self, instances: xgb.DMatrix) -> np.ndarray:
        return self._booster.predict(instances)

    def postprocess(self, prediction_results: np.ndarray) -> dict:
        return {"predictions": prediction_results.tolist()}
