import ast
import json
import os
import pickle
import torch

from google.cloud.aiplatform.utils import prediction_utils
from google.cloud.aiplatform.prediction.predictor import Predictor
from transformers import AutoModelForQuestionAnswering
from typing import Dict, List

class TorchTransformersPredictor(Predictor):

    def __init__(self):
        return

    def load(self, artifacts_uri: str) -> None:
        prediction_utils.download_model_artifacts(artifacts_uri)
        
        if os.path.isfile("setup_config.json"):
            with open("setup_config.json") as setup_config_file:
                self.setup_config = json.load(setup_config_file)

        if os.path.exists("model.pt"):
            self.model = AutoModelForQuestionAnswering.from_pretrained("model.pt")
            self.model.eval()
        else:
            raise ValueError("One of the following model files must be provided: model.pt.")

    def preprocess(self, prediction_input: dict) -> torch.Tensor:
        max_length = self.setup_config["max_length"]
        instances = prediction_input["instances"]
        question_context = ast.literal_eval(instances)
        question = question_context["question"]
        context = question_context["context"]
        inputs = self.tokenizer.encode_plus(
            question,
            context,
            max_length=int(max_length),
            pad_to_max_length=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        return torch.Tensor(input_ids, attention_mask)
    
    @torch.inference_mode()
    def predict(self, instances: torch.Tensor) -> List[str]:
        input_ids, attention_mask = instances
        outputs = self._model(input_ids, attention_mask)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits
        
        num_rows, num_cols = answer_start_scores.shape
        inferences = []
        for i in range(num_rows):
            answer_start_scores_one_seq = answer_start_scores[i].unsqueeze(0)
            answer_start = torch.argmax(answer_start_scores_one_seq)
            answer_end_scores_one_seq = answer_end_scores[i].unsqueeze(0)
            answer_end = torch.argmax(answer_end_scores_one_seq) + 1
            prediction = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(
                    input_ids[i].tolist()[answer_start:answer_end]
                )
            )
            inferences.append(prediction)
        return inferences

    def postprocess(self, prediction_results: List[str]) -> Dict:
        return {"predictions": prediction_results}
     

