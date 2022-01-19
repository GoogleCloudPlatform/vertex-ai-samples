
import os
import json
import logging

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class TransformersClassifierHandler(BaseHandler):
    """
    The handler takes an input string and returns the classification text 
    based on the serialized transformers checkpoint.
    """
    def __init__(self):
        super(TransformersClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        """ Loads the model.pt file and initialized the model object.
        Instantiates Tokenizer for preprocessor to use
        Loads labels to name mapping file for post-processing inference response
        """
        self.manifest = ctx.manifest

        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt or pytorch_model.bin file")
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        logger.debug('Transformer model from path {0} loaded successfully'.format(model_dir))
        
        # Ensure to use the same tokenizer used during training
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

        # Read the mapping file, index to object name
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")

        if os.path.isfile(mapping_file_path):
            with open(mapping_file_path) as f:
                self.mapping = json.load(f)
        else:
            logger.warning('Missing the index_to_name.json file. Inference output will default.')
            self.mapping = {"0": "Negative",  "1": "Positive"}

        self.initialized = True

    def preprocess(self, data):
        """ Preprocessing input request by tokenizing
            Extend with your own preprocessing steps as needed
        """
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        sentences = text.decode('utf-8')
        logger.info("Received text: '%s'", sentences)

        # Tokenize the texts
        tokenizer_args = ((sentences,))
        inputs = self.tokenizer(*tokenizer_args,
                                padding='max_length',
                                max_length=128,
                                truncation=True,
                                return_tensors = "pt")
        return inputs

    def inference(self, inputs):
        """ Predict the class of a text using a trained transformer model.
        """
        prediction = self.model(inputs['input_ids'].to(self.device))[0].argmax().item()

        if self.mapping:
            prediction = self.mapping[str(prediction)]

        logger.info("Model predicted: '%s'", prediction)
        return [prediction]

    def postprocess(self, inference_output):
        return inference_output
