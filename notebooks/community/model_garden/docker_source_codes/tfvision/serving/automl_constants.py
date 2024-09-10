"""AutoML tfvision saved_model constants."""

# Tfvision training artifact marcos.
# Exported parameter.yaml in the model directory.
CFG_FILENAME = 'params.yaml'

# Common automl saved_model marcos.
## Type of input to automl saved_model, fixed as image bytes string.
INPUT_TYPE = 'image_bytes'
IMAGE_TENSOR = 'image_tensor'
## Automl IOD saved_model signature input image argument name.
IOD_INPUT_NAME = 'encoded_image'
## ICN saved_model input name.
ICN_INPUT_NAME = 'image_bytes'
## Automl saved_model signature input key argument name.
INPUT_KEY_NAME = 'key'
OUTPUT_KEY_NAME = 'key'

# IOD saved_model marcos.
## IOD class as text output
DETECTION_CLASSES_AS_TEXT = 'detection_classes_as_text'
## Default value for labelmap text lookup table.
LOOKUP_DEFAULT_VALUE = 'unknown'
## Suffix for signature def without input key tensor.
NO_KEY_SIG_DEF_SUFFIX = '_without_key'
