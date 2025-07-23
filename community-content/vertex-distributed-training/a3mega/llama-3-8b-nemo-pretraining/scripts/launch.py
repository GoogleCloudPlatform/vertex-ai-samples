"""Launch script for Vertex distributed training"""

# Copy the sample_job_config.json file to job_config.json
# to define the job parameters.
#
# Run like this:
#
#   python3 vertex_dist_train/launch.py --config_file=job_config.json
#

import datetime
import json
import os
import pprint
from collections.abc import Sequence
from typing import Any, List

from absl import app, flags
from google.cloud import aiplatform
from google.cloud.aiplatform_v1.types.custom_job import Scheduling
from pytz import timezone

FLAGS = flags.FLAGS
flags.DEFINE_string("config_file", None, "Path to JSON config file")
flags.DEFINE_boolean(
    "debug", False, "Debug mode: just print the command, don't run it."
)


def launch_job(
    job_name: str,
    project: str,
    region: str,
    gcs_bucket: str,
    image_uri: str,
    entrypoint_cmd: List[str],
    trainer_args: List[Any],
    num_nodes: int,
    machine_type: str,
    num_gpus_per_node: int,
    gpu_type: str,
    strategy: str,
    reservation_name: str = "",
):
    assert strategy in ("dws", "spot", "reservation")
    aiplatform.init(
        project=project, location=region, staging_bucket=gcs_bucket
    )

    train_job = aiplatform.CustomContainerTrainingJob(
        display_name=job_name,
        container_uri=image_uri,
        command=entrypoint_cmd,
    )

    job_args = dict(
        args=trainer_args,
        enable_web_access=True,
        replica_count=num_nodes,
        machine_type=machine_type,
        accelerator_type=gpu_type,
        accelerator_count=num_gpus_per_node,
        boot_disk_size_gb=1000,
        # restart_job_on_worker_restart=True,
        restart_job_on_worker_restart=False,
    )

    if strategy == "spot":
        job_args.update({"scheduling_strategy": Scheduling.Strategy.SPOT.name})
    elif strategy == "dws":
        job_args.update(
            {"scheduling_strategy": Scheduling.Strategy.FLEX_START.name}
        )
    elif strategy == "reservation":
        assert reservation_name != "", (
            "If using a reservation, provide the reservation_name in the "
            "format `projects/{project_id_or_number}/zones/{zone}/"
            "reservations/{reservation_name}`"
        )
        job_args.update(
            {
                "reservation_affinity_type": "SPECIFIC_RESERVATION",
                "reservation_affinity_key": "compute.googleapis.com/reservation-name",
                "reservation_affinity_values": [reservation_name],
            }
        )

    pprint.pprint(job_args)
    if not FLAGS.debug:
        train_job.submit(**job_args)


def main(argv: Sequence[str]) -> None:
    config_file_path = FLAGS.config_file
    print(f"Reading job config from {config_file_path}")
    with open(config_file_path, encoding="utf-8") as config_file:
        config = json.load(config_file)

    project_id = config["project_id"]
    region = config["region"]
    zone = config["zone"]
    bucket = config["bucket"]
    dataset_bucket = config["dataset_bucket"]
    n_nodes = int(config["nodes"])
    machine_type = config["machine_type"]
    num_gpus_per_node = int(config["gpus_per_node"])
    gpu_type = config["gpu_type"]
    reservation_name = config.get("reservation_name")
    reservation_full_name = (
        f"projects/{project_id}/zones/{zone}/reservations/{reservation_name}"
        if "reservation_name" in config
        else ""
    )

    strategy = config["strategy"]
    recipe_name = config["recipe_name"]
    job_prefix = config["job_prefix"]
    image_uri = config["image_uri"]

    # Job name
    timestamp = (
        datetime.datetime.now()
        .astimezone(timezone("US/Pacific"))
        .strftime("%Y%m%d_%H%M%S")
    )
    job_name = f"{recipe_name}-{timestamp}"
    if job_prefix:
        job_name = f"{job_prefix}-{job_name}"

    base_output_dir = os.path.join("/gcs", bucket, job_name)

    # Training command and args
    entrypoint_cmd = ["python3", "vdt/run.py"]

    dataset_bucket = f"gs://{config["dataset_bucket"]}"

    trainer_args = [
        f"--train_data_gcs={dataset_bucket}",
        "/opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py",
        "--config-path=conf/",
        f"--config-name={recipe_name}.yaml",
        f"exp_manager.explicit_log_dir={base_output_dir}",
        f"exp_manager.dllogger_logger_kwargs.json_file={base_output_dir}/dllogger.json",
        "+exp_manager.create_tensorboard_logger=true",
        "exp_manager.create_checkpoint_callback=false",
        f"trainer.num_nodes={n_nodes}",
        f"trainer.devices={num_gpus_per_node}",
        "trainer.max_steps=10",
        "trainer.log_every_n_steps=1",
        "model.tokenizer.vocab_file=/data/gpt2-vocab.json",
        "model.tokenizer.merge_file=/data/gpt2-merges.txt",
        "model.data.data_prefix=[1.0,/data/hfbpe_gpt_training_data_text_document]",
    ]

    launch_job(
        job_name=job_name,
        project=project_id,
        region=region,
        gcs_bucket=bucket,
        image_uri=image_uri,
        entrypoint_cmd=entrypoint_cmd,
        trainer_args=trainer_args,
        num_nodes=n_nodes,
        machine_type=machine_type,
        num_gpus_per_node=num_gpus_per_node,
        gpu_type=gpu_type,
        strategy=strategy,
        reservation_name=reservation_full_name,
    )


if __name__ == "__main__":
    app.run(main)
