# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build the model server container."""
import json
import logging
import os
import pathlib
from typing import Sequence

from absl import app
from absl import logging
from config import CPRConfig
from google.cloud import aiplatform
from google.cloud.aiplatform import prediction as cpr
import smart_open
import timm
from timm_serving import predictor
import torch


def build_container(config: CPRConfig, tag: str) -> cpr.LocalModel:
    """Build the model server container.

    Args:
      tag: Output image tag.

    Returns:
      LocalModel exposing the built model server.
    """
    return cpr.LocalModel.build_cpr_model(
        src_dir=os.path.join(os.getcwd()),
        output_image_uri=tag,
        base_image=config.base_image,
        predictor=predictor.TimmPredictor,
        requirements_path=os.path.join(os.getcwd(), "requirements.txt"),
    )


def save_model_artifact(destination: str) -> None:
    """Save a copy of the model state dict."""
    model = timm.create_model(predictor.TimmPredictor.TIMM_MODEL_NAME, pretrained=True)
    dest_file = os.path.join(destination, predictor.TimmPredictor.WEIGHTS_FILE)
    with smart_open.open(dest_file, "wb") as f:
        torch.save(model, f)
    logging.info("Saved model to %s", dest_file)
    logging.info("%s parameters", sum(p.numel() for p in model.parameters()))


def upload_model(config: CPRConfig) -> aiplatform.Model:
    """Tag and upload the model server."""
    ar_tag = (
        f"{config.region}-docker.pkg.dev/{config.project_id}"
        f"/{config.repository}/{config.image}"
    )
    local_model = build_container(config, tag=ar_tag)
    aiplatform.init(project=config.project_id, location=config.region)
    local_model.push_image()
    aip_model = aiplatform.Model.upload(
        local_model=local_model,
        display_name=predictor.TimmPredictor.TIMM_MODEL_NAME,
        artifact_uri=config.artifact_gcs_dir,
    )
    config.model_name = aip_model.resource_name
    config.save()
    return aip_model


def deploy_model(config: CPRConfig) -> aiplatform.Endpoint:
    """Deploy the model server to a Vertex Prediction endpoint."""
    aiplatform.init(project=config.project_id, location=config.region)
    aip_model = aiplatform.Model(model_name=config.model_name)
    endpoint = aip_model.deploy(machine_type=config.machine_type)
    config.endpoint_name = endpoint.resource_name
    config.save()
    return endpoint


def probe_prediction(config: CPRConfig, request_path: str) -> None:
    """Send a sample prediction request to the Vertex Prediction endpoint."""
    aiplatform.init(project=config.project_id, location=config.region)
    aip_endpoint = aiplatform.Endpoint(endpoint_name=config.endpoint_name)
    with open(request_path) as f:
        logging.info(aip_endpoint.predict(**json.load(f)))


def main(argv: Sequence[str]):
    config = CPRConfig()
    if pathlib.Path(config.config_file).exists():
        config.load()

    actions = set(argv[1:])
    if "build" in actions:
        build_container(config, config.image)
        save_model_artifact(config.artifact_local_dir)
    if "upload" in actions:
        save_model_artifact(config.artifact_gcs_dir)
        upload_model(config)
    if "deploy" in actions:
        deploy_model(config)
    if "probe" in actions:
        probe_prediction(config, request_path="sample_request.json")


if __name__ == "__main__":
    app.run(main)
