import os
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb

from google.cloud.aiplatform.constants import prediction
from google.cloud.aiplatform.utils import prediction_utils
from google.cloud.aiplatform.prediction.predictor import Predictor

class XGBRankerPredictor(Predictor):

    def __init__(self):
        return

    def load(self, artifacts_uri: str) -> None:
        prediction_utils.download_model_artifacts(artifacts_uri)
        if os.path.exists(prediction.MODEL_FILENAME_PKL):
            booster = pickle.load(open(prediction.MODEL_FILENAME_PKL, "rb"))
            self._booster = booster
        else:
            N = 500
            dates = pd.date_range(start='2023-01-01', end='2023-01-12', periods=N)
            X = pd.DataFrame(np.random.randn(N, 5), columns=list('ABCDE'), index=dates)
            y = pd.Series(np.random.randint(0, 10, size=N), index=dates, name='label')
            group = X.groupby(dates + pd.offsets.MonthEnd(0)).size()
            sample_weight = pd.Series(np.arange(len(group)), index=group.index)
            model = xgb.XGBRanker(objective='rank:pairwise', max_depth=3, learning_rate=0.1, booster='gbtree', tree_method='hist', n_jobs=4, n_estimators=50, enable_categorical=False, random_state=42)
            model.fit(X=X, y=y, group=group, sample_weight=sample_weight, verbose=True)
            booster = model.get_booster()
            self._booster = booster

    def preprocess(self, prediction_input: dict) -> xgb.DMatrix:
        instances = prediction_input["instances"]
        return xgb.DMatrix(instances)

    def predict(self, instances: xgb.DMatrix) -> np.ndarray:
        return self._booster.predict(instances, output_margin=False, ntree_limit=0)

    def postprocess(self, prediction_results: np.ndarray) -> dict:
        return {"predictions": prediction_results.tolist()}