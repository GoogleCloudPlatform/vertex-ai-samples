"""Entrypoint for axolotl train docker."""

import argparse
import json
import os
import subprocess


def _get_multi_node_flags(cluster_spec: str) -> list[str]:
  """Returns the multi-node flags."""
  print(f'CLUSTER_SPEC: {cluster_spec}')

  cluster_data = json.loads(cluster_spec)

  # Get primary node info
  primary_node = cluster_data['cluster']['workerpool0'][0]
  print(f'primary node: {primary_node}')
  primary_node_addr, primary_node_port = primary_node.split(':')
  print(f'primary node address: {primary_node_addr}')
  print(f'primary node port: {primary_node_port}')

  # Determine node rank of this machine
  workerpool = cluster_data['task']['type']
  if workerpool == 'workerpool0':
    node_rank = 0
  else:
    node_rank = cluster_data['task']['index'] + 1
  print(f'node rank: {node_rank}')

  # Calculate total nodes
  num_worker_nodes = len(cluster_data['cluster']['workerpool1'])
  num_nodes = num_worker_nodes + 1  # Add 1 for the primary node
  print(f'num nodes: {num_nodes}')

  return [
      f'--machine_rank={node_rank}',
      f'--num_machines={num_nodes}',
      f'--main_process_ip={primary_node_addr}',
      f'--main_process_port={primary_node_port}',
      '--max_restarts=0',
      '--monitor_interval=120',
      '--dynamo_backend=no',
  ]


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('--config_file')
  parser.add_argument('--huggingface_access_token')
  args, unknown = parser.parse_known_args()

  accelerate_flags = []

  if args.config_file:
    accelerate_flags.append(f'--config_file={args.config_file}')

  if cluster_spec := os.getenv('CLUSTER_SPEC', default=None):
    print('========== Launch on cloud multi nodes ==========')
    accelerate_flags.extend(_get_multi_node_flags(cluster_spec))

  cmd = (
      [
          'accelerate',
          'launch',
      ]
      + accelerate_flags
      + [
          '-m',
          'axolotl.cli.train',
      ]
      + unknown
  )
  print(f'{cmd=}', flush=True)

  env = os.environ.copy()

  if args.huggingface_access_token:
    env['HF_TOKEN'] = args.huggingface_access_token

  subprocess.run(
      cmd,
      check=True,
      env=env,
  )


if __name__ == '__main__':
  main()
