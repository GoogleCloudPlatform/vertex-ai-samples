"""Class that bundles docker related flags."""

import getpass
import os
import pwd


class DockerCommandBuilder:
  """Bundle docker related flags."""

  def __init__(self, docker_uri, shm_size='128gb'):
    self._docker_uri = [docker_uri]

    self._defaults = [
        'docker',
        'run',
        '--gpus=all',
        '--net=host',
        '--rm',
        f'--shm-size={shm_size}',
    ]

    user = getpass.getuser()
    # username ends with `_google_com` is managed by ldap and does not have a
    # corresponding entry in /etc/passwd or /etc/group file. We cannot enable
    # non-root docker user with below method.
    if not user.endswith('_google_com'):
      uid = os.getuid()
      gid = pwd.getpwuid(uid).pw_gid
      self._defaults += [
          f'--user={uid}:{gid}',
          '--volume=/etc/group:/etc/group:ro',
          '--volume=/etc/passwd:/etc/passwd:ro',
      ]
    self._env_vars = []
    self._mount_maps = []

  def add_env_var(self, var, val):
    self._env_vars.append(f'--env={var}={val}')

  def add_mount_map(self, host_path, docker_path):
    self._mount_maps.append(f'--volume={host_path}:{docker_path}')

  def build_cmd(self) -> str:
    return self._defaults + self._env_vars + self._mount_maps + self._docker_uri
