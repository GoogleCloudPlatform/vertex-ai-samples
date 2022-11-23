#!/usr/bin/env python
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

"""
 This script is used to update variables in the notebook via regex
 It requires variables to be defined in particular format

For example, if your variable was PROJECT_ID, use:

    PROJECT_ID = "[your_project_here]"

Single-quotes also work:

    PROJECT_ID = '[your_project_here]'

Variables in conditionals can also be replaced:

    PROJECT_ID == "[your_project_here]"
"""


def get_updated_value(content: str, variable_name: str, variable_value: str) -> str:
    return re.sub(
        rf"({variable_name}.*? = .*?[\",\'])\[.+?\]([\",\'].*?)",
        rf"\g<1>{variable_value}\g<2>",
        content,
        flags=re.M,
    )
