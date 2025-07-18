"""Add tests for cluster_spec.py."""

import os

from . import cluster_spec


# TODO(styer): Use pytest instead
class ClusterSpecTest(googletest.TestCase):

    def setUp(self):
        super().setUp()
        self.curr_env_var = os.environ.copy()

    def tearDown(self):
        super().tearDown()
        os.environ = self.curr_env_var

    def test_get_cluster_spec_from_env_vars(self):
        os.environ["CLUSTER_SPEC"] = ""
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "8080"
        os.environ["RANK"] = "0"
        os.environ["NNODES"] = "2"
        cluster_info = cluster_spec.get_cluster_spec()
        self.assertEqual(cluster_info.primary_node_addr, "127.0.0.1")
        self.assertEqual(cluster_info.primary_node_port, "8080")
        self.assertEqual(cluster_info.node_rank, 0)
        self.assertEqual(cluster_info.num_nodes, 2)

    def test_get_cluster_spec_from_cluster_spec(self):
        os.environ[
            "CLUSTER_SPEC"
        ] = """
    {
      "cluster": {
        "workerpool0": [
          "127.0.0.1:8080"
        ],
        "workerpool1": [
          "127.0.0.2:8080",
          "127.0.0.3:8080"
        ]
      },
      "task": {
        "type": "workerpool1",
        "index": 0
      }
    }
    """
        cluster_info = cluster_spec.get_cluster_spec()
        self.assertEqual(cluster_info.primary_node_addr, "127.0.0.1")
        self.assertEqual(cluster_info.primary_node_port, "8080")
        self.assertEqual(cluster_info.node_rank, 1)
        self.assertEqual(cluster_info.num_nodes, 3)


if __name__ == "__main__":
    googletest.main()
