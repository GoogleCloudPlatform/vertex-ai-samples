"""Test the timm_serving predictor."""
import base64
import json
import logging
import os
import pickle
from typing import List, Dict

from absl import flags
from absl import logging
from absl.testing import absltest
from config import CPRConfig
import fastapi
from google.cloud import aiplatform
from google.cloud.aiplatform import prediction as cpr
import PIL
from timm_serving import predictor
import torch

VIT_SMALL_PARAMS = 22878952


def b64_encode_file(path: str) -> str:
    """Encode a file's contents as base64.

    Args:
      path: Path to the file.

    Returns:
      Base64-encoded contents of the file.
    """
    with open(path, "rb") as f:
        return str(base64.b64encode(f.read()), encoding="utf-8")


def make_instance_dict(
    image_paths: List[str], base64_encodings: List[str]
) -> Dict[str, List[str]]:
    """Generate a dictionary similar to a parsed prediction server request.

    Args:
      image_paths: Paths to image files to include.
      base64_encodings: Pre-encoded base64 strings.

    Returns:
      Dictionary of instances in the format accepted by the preprocessor.
    """
    instances = [s for s in base64_encodings]
    for path in image_paths:
        instances.append(b64_encode_file(path))
    return {"instances": instances}


def count_parameters(model: torch.nn.Module):
    """Count the parameters in a Pytorch model.

    Args:
      model: Pytorch model (nn.Module).

    Returns:
      Number of parameters in the model.

    """
    return sum(p.numel() for p in model.parameters())


class PredictorUnitTests(absltest.TestCase):
    """Unit tests for timm_serving.predictor."""

    def setUp(self):
        super().setUp()
        self.config = CPRConfig()
        try:
            self.config.load()
        except FileNotFoundError:
            logging.info("No saved config file found, using default values.")
        self.predictor = predictor.TimmPredictor()

    def test_load_from_saved_state_dict_ok(self):
        self.predictor.load(self.config.artifact_local_dir)
        self.assertEqual(count_parameters(self.predictor._model), VIT_SMALL_PARAMS)

    def test_load_bad_path(self):
        with self.assertRaises(FileNotFoundError):
            self.predictor.load("testdata/")
        with self.assertRaisesRegex(ValueError, "not a directory"):
            self.predictor.load("blah")

    def test_load_bad_data(self):
        with self.assertRaises(pickle.UnpicklingError):
            self.predictor.load("testdata/bad_model_1")
        with self.assertRaisesRegex(RuntimeError, "Invalid magic number"):
            self.predictor.load("testdata/bad_model_2")

    def test_preprocess_ok(self):
        self.predictor.load(self.config.artifact_local_dir)
        instance_dict = make_instance_dict(
            base64_encodings=[],
            image_paths=[
                "testdata/airplane.jpg",
                "testdata/mandrill.tiff",
                "testdata/mandrill.tiff",
                "testdata/cat_alpha.png",
            ],
        )
        result = self.predictor.preprocess(instance_dict)
        self.assertEqual(result.size(), torch.Size([4, 3, 224, 224]))
        self.assertEqual(result.dtype, torch.float32)

    def test_preprocess_no_instances(self):
        self.predictor.load(self.config.artifact_local_dir)
        with self.assertRaises(fastapi.HTTPException) as ctx:
            self.predictor.preprocess({})
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertRegex(ctx.exception.detail, 'must contain "instances"')

    def test_preprocess_wrong_shape_instances(self):
        self.predictor.load(self.config.artifact_local_dir)
        instance_dict = {"instances": [[b64_encode_file("testdata/mandrill.tiff")]]}
        with self.assertRaises(fastapi.HTTPException) as ctx:
            self.predictor.preprocess(instance_dict)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertRegex(ctx.exception.detail, "not 'list'")

    def test_preprocess_bad_base64(self):
        self.predictor.load(self.config.artifact_local_dir)
        instance_dict = make_instance_dict(base64_encodings=["!@#$"], image_paths=[])
        with self.assertRaises(fastapi.HTTPException) as ctx:
            self.predictor.preprocess(instance_dict)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertRegex(ctx.exception.detail, "[Bb]ase64")

    def test_preprocess_not_image_data(self):
        self.predictor.load(self.config.artifact_local_dir)
        instance_dict = make_instance_dict(
            base64_encodings=[], image_paths=["testdata/bad.jpg"]
        )
        with self.assertRaises(fastapi.HTTPException) as ctx:
            self.predictor.preprocess(instance_dict)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertRegex(ctx.exception.detail, "image file")

    def test_predict_ok(self):
        self.predictor.load(self.config.artifact_local_dir)
        inputs = torch.zeros(size=[2, 3, 224, 224], dtype=torch.float32)
        if torch.cuda.device_count() > 0:
            inputs = inputs.cuda()
        result = self.predictor.predict(inputs)
        self.assertEqual(result.size(), torch.Size([2, 1000]))
        self.assertEqual(result.dtype, torch.float32)

    def test_postprocess_ok(self):
        class_probs = torch.zeros(size=[2, 1000])
        class_probs[0, 0] = 1
        class_probs[1, 123] = 1
        result = self.predictor.postprocess(class_probs)
        predictions = result["predictions"]
        self.assertLen(predictions[0]["class_names"], 5)
        self.assertLen(predictions[0]["indices"], 5)
        self.assertLen(predictions[0]["probabilities"], 5)
        self.assertLen(predictions[1]["class_names"], 5)
        self.assertLen(predictions[1]["indices"], 5)
        self.assertLen(predictions[1]["probabilities"], 5)
        self.assertContainsSubsequence(predictions[0]["class_names"][0], "tench")
        self.assertContainsSubsequence(
            predictions[1]["class_names"][0], "spiny lobster"
        )


class ServerEndToEndTests(absltest.TestCase):
    """End-to-end tests for the model server, using LocalEndpoint."""

    def setUp(self):
        super().setUp()
        self.config = CPRConfig()
        try:
            self.config.load()
        except FileNotFoundError:
            logging.info("No saved config file found, using default values.")
        self.local_model = cpr.LocalModel(
            serving_container_spec=aiplatform.gapic.ModelContainerSpec(
                image_uri=self.config.image
            )
        )

        self.local_endpoint = self.local_model.deploy_to_local_endpoint(
            artifact_uri=self.config.artifact_local_dir or os.getcwd()
        )
        self.local_endpoint.serve()

    def tearDown(self):
        self.local_endpoint.stop()
        super().tearDown()

    def test_e2e_healthcheck_ok(self):
        health_check_response = self.local_endpoint.run_health_check()
        self.assertEqual(health_check_response.status_code, 200)
        self.assertEqual(health_check_response.content, b"{}")

    def test_e2e_predict_ok(self):
        predict_request = json.dumps(
            make_instance_dict(
                base64_encodings=[],
                image_paths=[
                    "testdata/mandrill.tiff",
                ],
            )
        )
        response = self.local_endpoint.predict(
            request=predict_request, headers={"Content-Type": "application/json"}
        )
        logging.info(response.content)
        self.assertEqual(response.status_code, 200)
        predictions = response.json()["predictions"]
        self.assertContainsSubsequence(predictions[0]["class_names"][0], "baboon")

    def test_e2e_predict_bad_json_returns_400(self):
        predict_request = "blah"
        response = self.local_endpoint.predict(
            request=predict_request, headers={"Content-Type": "application/json"}
        )
        logging.info(response.content)
        self.assertEqual(response.status_code, 400)

    def test_e2e_predict_no_instances_returns_400(self):
        predict_request = json.dumps({})
        response = self.local_endpoint.predict(
            request=predict_request, headers={"Content-Type": "application/json"}
        )
        logging.info(response.content)
        self.assertEqual(response.status_code, 400)

    def test_e2e_predict_bad_base64_returns_400(self):
        predict_request = json.dumps(
            make_instance_dict(base64_encodings=["blah"], image_paths=[])
        )
        response = self.local_endpoint.predict(
            request=predict_request, headers={"Content-Type": "application/json"}
        )
        logging.info(response.content)
        self.assertEqual(response.status_code, 400)

    def test_e2e_predict_bad_image_returns_400(self):
        predict_request = json.dumps(
            make_instance_dict(base64_encodings=[], image_paths=["testdata/bad.jpg"])
        )
        response = self.local_endpoint.predict(
            request=predict_request, headers={"Content-Type": "application/json"}
        )
        logging.info(response.content)
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    absltest.main()
