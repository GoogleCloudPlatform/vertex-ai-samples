import os
import torch

from google.cloud.aiplatform.utils import prediction_utils
from google.cloud.aiplatform.prediction.predictor import Predictor
from torchvision.models import detection, resnet50, ResNet50_Weights
from typing import Dict, List

class ResNetPredictor(Predictor):

    def __init__(self):
        return

    def load(self, artifacts_uri: str) -> None:
        prediction_utils.download_model_artifacts(artifacts_uri)
        if os.path.exists("model.pth.tar"):
            self.model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
            stat_dic = torch.load("model.pth.tar")
            self.model.load_state_dict(stat_dic['state_dict'])
        else:
            weights = ResNet50_Weights.DEFAULT
            self.model = resnet50(weights=weights)
        self.model.eval()

    def preprocess(self, prediction_input: dict) -> torch.Tensor:
        instances = prediction_input["instances"]
        return torch.Tensor(instances)

    @torch.inference_mode()
    def predict(self, instances: torch.Tensor) -> List[str]:
        return self._model(instances)

    def postprocess(self, prediction_results: List[str]) -> Dict:
        return {"predictions": prediction_results}
