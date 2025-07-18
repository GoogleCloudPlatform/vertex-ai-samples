"""Get cluster info from environment variables."""

import dataclasses
import json
import os

from absl import logging


@dataclasses.dataclass
class ClusterInfo:
    """Contains information about the cluster.

    Attributes:
      primary_node_addr: The address of the primary node.
      primary_node_port: The port of the primary node.
      node_rank: The rank of the node.
      num_nodes: The number of nodes in the cluster.
    """

    primary_node_addr: str | None = None
    primary_node_port: str | None = None
    node_rank: int = 0
    num_nodes: int = 1

    # Allows unpacking operation like
    # primary_node_addr, primary_node_port, _, _ = ClusterInfo()
    # See https://stackoverflow.com/a/70753113
    def __iter__(self):
        return iter(dataclasses.astuple(self))


def get_cluster_spec() -> ClusterInfo:
    """Parses CLUSTER_SPEC environment variable and returns the cluster info.

    Returns:
      A ClusterInfo object.
    """
    cluster_spec = os.getenv("CLUSTER_SPEC", None)

    # If CLUSTER_SPEC is not set, use individual vars to construct cluster info.
    if not cluster_spec:
        cluster_info = ClusterInfo(
            primary_node_addr=os.getenv("MASTER_ADDR", None),
            primary_node_port=os.getenv("MASTER_PORT", None),
            node_rank=int(os.getenv("RANK", "0")),
            num_nodes=int(os.getenv("NNODES", "1")),
        )
        return cluster_info

    cluster_data = json.loads(cluster_spec)
    # Get primary node info
    primary_node = cluster_data["cluster"]["workerpool0"][0]
    logging.info("primary node: %s", primary_node)
    primary_node_addr, primary_node_port = primary_node.split(":")
    logging.info("primary node address: %s", primary_node_addr)
    logging.info("primary node port: %s", primary_node_port)

    # Determine node rank of this machine
    workerpool = cluster_data["task"]["type"]
    if workerpool == "workerpool0":
        node_rank = 0
    elif workerpool == "workerpool1":
        # Add 1 for the primary node, since `index` is the index of workerpool1.
        node_rank = cluster_data["task"]["index"] + 1
    else:
        raise ValueError(
            "Only workerpool0 and workerpool1 are supported. Unknown workerpool:"
            f" {workerpool}"
        )
    logging.info("node rank: %s", node_rank)

    # Calculate total nodes.
    num_nodes = 1  # For the primary node.
    if "workerpool1" in cluster_data["cluster"]:
        num_nodes += len(cluster_data["cluster"]["workerpool1"])
    logging.info("num nodes: %s", num_nodes)

    return ClusterInfo(
        primary_node_addr, primary_node_port, node_rank, num_nodes
    )
