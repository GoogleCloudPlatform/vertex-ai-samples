# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import google.auth
from google.auth.transport.requests import Request
from openai import OpenAI


def get_gcp_access_token():
  creds, _ = google.auth.default()
  creds.refresh(Request())
  return creds.token


# Get default project ID from environment
_, project_id = google.auth.default()

client = OpenAI(
    base_url=f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/us-central1/endpoints/openapi",
    api_key=get_gcp_access_token(),
)

response = client.chat.completions.create(
    model="google/gemini-2.5-pro",
    messages=[{"role": "user", "content": "Why is the sky blue?"}],
)
print(response.choices[0].message.content)
