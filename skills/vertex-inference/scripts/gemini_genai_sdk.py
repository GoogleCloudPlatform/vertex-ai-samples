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

from google import genai
import google.auth

# Get default project ID from environment
_, project_id = google.auth.default()

# Initialize GenAI Client with Vertex AI backend
# Use location="global" for Preview models (Gemini 2.0)
client = genai.Client(vertexai=True, project=project_id, location="us-central1")

response = client.models.generate_content(
    model="gemini-3-pro", contents="Why is the sky blue?"
)
print(response.text)
