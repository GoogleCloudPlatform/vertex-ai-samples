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

#!/bin/bash
set -e

# Create a temporary directory for the virtual environment
VENV_DIR=$(mktemp -d -t venv_verify.XXXXXX)
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Trap to ensure cleanup happens on exit
cleanup() {
  echo "Cleaning up virtual environment..."
  deactivate 2>/dev/null || true
  rm -rf "$VENV_DIR"
}
trap cleanup EXIT

echo "Installing requirements..."
pip install -q -r scripts/requirements.txt

echo "Running verification tests..."
FAILED=0

# Iterate directly over the files in the scripts directory
for script in scripts/*.py; do
  echo "Running $script..."
  if python3 "$script" > /dev/null 2>&1; then
    echo "  PASS: $script"
  else
    echo "  FAIL: $script"
    python3 "$script" # Run again to show output
    FAILED=1
  fi
done

if [ $FAILED -eq 0 ]; then
  echo "All scripts passed verification!"
  exit 0
else
  echo "Some scripts failed verification."
  exit 1
fi
