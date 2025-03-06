"""Class that bundles docker related flags."""

import getpass
import os
import pwd


class CommandBuilder:
  """Base class for building commands."""

  def __init__(self):
    self._defaults = []
    self._env_vars = {}

  def add_env_var(self, var: str, val: str) -> None:
    """Add environment variable to the command.

    Args:
      var: environment variable name.
      val: environment variable value.
    """
    self._env_vars[var] = val

  def add_mount_map(self, host_path, docker_path):
    pass


class DockerCommandBuilder(CommandBuilder):
  """Bundle docker related flags."""

  def __init__(self, docker_uri: str, shm_size: str = '128gb'):
    super().__init__()
    self._docker_uri = [docker_uri]
    self.privilege_mode = []

    self._defaults = [
        'docker',
        'run',
        '--gpus=all',
        '--net=host',
        '--rm',
        f'--shm-size={shm_size}',
    ]

    self._mount_maps = []
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

  def add_mount_map(self, host_path, docker_path):
    self._mount_maps.append(f'--volume={host_path}:{docker_path}')

  def add_privilege_mode(self):
    self.privilege_mode = ['--privileged']

  def build_cmd(self) -> str:
    return (
        self._defaults
        + [f'--env={var}={val}' for var, val in self._env_vars.items()]
        + self._mount_maps
        + self.privilege_mode
        + self._docker_uri
    )


class PythonCommandBuilder(CommandBuilder):
  """Bundle Python test command related flags."""

  def __init__(self):
    super().__init__()
    self._defaults = [
        'python3',
        './vertex_vision_model_garden_peft/train/vmg/train_entrypoint.py',
    ]

  def build_cmd(self) -> str:
    os.environ.update(self._env_vars)
    return self._defaults
