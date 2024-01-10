#!/bin/bash

# Setup accelerate config before running trainer.
python -c "from accelerate.utils import write_basic_config; write_basic_config(mixed_precision='fp16')"

accelerate launch "$@"
