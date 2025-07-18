"""Entrypoint for Vertex Distributed Training container."""

import argparse
import os
import sys
from collections.abc import Sequence
from subprocess import STDOUT, check_output, run

from absl import app, flags, logging
from util import cluster_spec

from retrying import retry

# PyTorch barrier call which synchronizes all of the nodes before launching the training process.
# This makes sure that processes will block until all processes are ready.
# Improves the reliability of spot VM usage for multi-node training jobs

@retry(stop_max_attempt_number=100, wait_exponential_multiplier=1000)
def barrier_with_retry() -> None:
    import torch
    logging.info("Starting barrier on RANK {}".format(os.environ["RANK"]))
    torch.distributed.init_process_group()
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    logging.info("Finished barrier on RANK {}".format(os.environ["RANK"]))

def main(unused_argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_gcs",
        type=str,
        help="Download training data from gcs path",
    )
    args, unknown = parser.parse_known_args()

    for key, val in os.environ.items():
        logging.info("ENV %s=%s", key, val)

    if args.train_data_gcs:
        local_dir = "/data"
        if not os.path.exists(local_dir):
            os.mkdir(local_dir)
        logging.info("downloading %s to %s...", args.train_data_gcs, local_dir)
        check_output(
            [
                "gcloud",
                "storage",
                "cp",
                "-r",
                f"{args.train_data_gcs}/*",
                local_dir,
            ],
            stderr=STDOUT,
        )
        logging.info("%s downloaded.", args.train_data_gcs)

    primary_node_addr, primary_node_port, node_rank, num_nodes = (
        cluster_spec.get_cluster_spec()
    )

    cmd = [
        "torchrun",
        "--nproc-per-node=8",
        f"--nnodes={num_nodes}",
        f"--node_rank={node_rank}",
    ]
    if num_nodes > 1:
        cmd += [
            "--max-restarts=3",
            "--rdzv-backend=static",
            f'--rdzv_id={os.getenv("CLOUD_ML_JOB_ID", primary_node_port)}',
            f"--rdzv-endpoint={primary_node_addr}:{primary_node_port}",
        ]
    cmd += unknown

    logging.info("launching with cmd: \n%s", " \\\n".join(cmd))
    barrier_with_retry()
    run(cmd, stdout=sys.stdout, stderr=sys.stdout, check=True)


if __name__ == "__main__":
    logging.get_absl_handler().python_handler.stream = sys.stdout
    app.run(
        main, flags_parser=lambda _args: flags.FLAGS(_args, known_only=True)
    )
