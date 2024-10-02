"""Tools to generate CommandBuilder class.

See go/vmg-oss-peft-tests#commandbuilder-class-generation for details.
"""

import argparse
import dataclasses
from typing import List

_DO_NOT_MODIFY_WARNING = """
# DO NOT MODIFY: this file is auto-generated
# See go/vmg-oss-peft-tests#command-builder-genpy
"""

_GETTER_TMPL = """
    @property
    def {}(self):
        return self._{}
"""

_SETTER_TMPL = """
    @{}.setter
    def {}(self, val: {}):
        self._{} = val
"""

_INIT_NAME = """
    def __init__(self):"""

_INIT_FIELDS = """
        self._{} = None"""

_BUILD_CMD = r"""
    def build_cmd(self) -> str:
        cmd = []
        for k, v in self.__dict__.items():
            if v is not None:
                cmd.append(f'--{k[1:]}={v}')
        return cmd
"""


@dataclasses.dataclass
class FlagInfo:
  api_name: str
  impl_name: str
  arg_type: str


def get_flag_info(line: str) -> FlagInfo:
  api_name, impl_name, arg_type = [x.strip() for x in line.split(',')]
  return FlagInfo(api_name, impl_name, arg_type)


def gen_getter(info: FlagInfo) -> str:
  return _GETTER_TMPL.format(info.api_name, info.impl_name)


def gen_setter(info: FlagInfo) -> str:
  return _SETTER_TMPL.format(
      info.api_name, info.api_name, info.arg_type, info.impl_name
  )


def gen_init(infos: List[FlagInfo]) -> str:
  fields = [_INIT_FIELDS.format(i.impl_name) for i in infos]
  return ''.join([_INIT_NAME] + fields)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--flags_def', required=True, help='file path contain flags definition.'
  )
  parser.add_argument(
      '--generated_file',
      required=True,
      help='file path to the generated command builder.',
  )
  parser.add_argument(
      '--class_name',
      required=True,
      help='class name for command build',
  )
  args = parser.parse_args()

  flags_info = []
  with open(args.flags_def, 'r') as flags_f:
    for line in flags_f:
      if not line.startswith('#'):
        flags_info.append(get_flag_info(line))

  with open(args.generated_file, 'w') as gen_f:
    # Disables pylint messages.
    # See https://stackoverflow.com/a/43510297
    print('# pylint: disable=W,C,R', file=gen_f)
    print(_DO_NOT_MODIFY_WARNING, file=gen_f)
    print(f'class {args.class_name}:', file=gen_f)
    print(gen_init(flags_info), file=gen_f)
    for info in flags_info:
      print(gen_getter(info), file=gen_f)
      print(gen_setter(info), file=gen_f)
    print(_BUILD_CMD, file=gen_f)

  print(f'file generated at {args.generated_file}')


if __name__ == '__main__':
  main()
