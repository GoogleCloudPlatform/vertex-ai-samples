"""Vertex vision model garden util constants."""

# TfVision Objectives.
OBJECTIVE_IMAGE_CLASSIFICATION = 'icn'
OBJECTIVE_IMAGE_OBJECT_DETECTION = 'iod'
OBJECTIVE_IMAGE_SEGMENTATION = 'isg'
OBJECTIVE_VIDEO_CLASSIFICATION = 'vcn'
OBJECTIVE_VIDEO_ACTION_RECOGNITION = 'var'

# PyTorch Models.
OBJECTIVE_TIMM = 'timm'

# Input file types.
INPUT_FILE_TYPE_CSV = 'csv'
INPUT_FILE_TYPE_JSONL = 'jsonl'
INPUT_FILE_TYPE_COCO_JSON = 'coco_json'

# Output file types.
OUTPUT_FILE_TYPE_TFRECORD = 'tfrecord'
OUTPUT_FILE_TYPE_COCO_JSON = 'coco_json'

# Best evaluation metrics.
IMAGE_CLASSIFICATION_SINGLE_LABEL_BEST_EVAL_METRIC = 'accuracy'
IMAGE_CLASSIFICATION_MULTI_LABEL_BEST_EVAL_METRIC = 'meanPR-AUC'

IMAGE_OBJECT_DETECTION_BEST_EVAL_METRIC = 'AP50'
IMAGE_SEGMENTATION_BEST_EVAL_METRIC = 'mean_iou'

VIDEO_CLASSIFICATION_BEST_EVAL_METRIC = 'accuracy'

# Best checkpoints.
BEST_CKPT_DIRNAME = 'best_ckpt'
BEST_CKPT_EVAL_FILENAME = 'info.json'
BEST_CKPT_STEP_NAME = 'best_ckpt_global_step'
BEST_CKPT_METRIC_COMP = 'higher'

# Reported hyperparameter tuning metric tag.
HP_METRIC_TAG = 'model_performance'

# HPT trial prefix.
TRIAL_PREFIX = 'trial_'

# ML uses from user input.
ML_USE_TRAINING = 'training'
ML_USE_VALIDATION = 'validation'
ML_USE_TEST = 'test'

# COCO json keys
COCO_JSON_ANNOTATIONS = 'annotations'
COCO_JSON_ANNOTATION_IMAGE_ID = 'image_id'
COCO_JSON_ANNOTATION_CATEGORY_ID = 'category_id'
COCO_JSON_CATEGORIES = 'categories'
COCO_JSON_CATEGORY_ID = 'id'
COCO_JSON_CATEGORY_NAME = 'name'
COCO_JSON_FILE_NAME = 'file_name'
COCO_JSON_IMAGES = 'images'
COCO_JSON_IMAGE_ID = 'id'
COCO_JSON_IMAGE_WIDTH = 'width'
COCO_JSON_IMAGE_HEIGHT = 'height'
COCO_JSON_IMAGE_COCO_URL = 'coco_url'
COCO_ANNOTATION_BBOX = 'bbox'

# GCS prefixes
GCS_URI_PREFIX = 'gs://'
GCSFUSE_URI_PREFIX = '/gcs/'

LOCAL_EVALUATION_RESULT_DIR = '/tmp/evaluation_result_dir'
LOCAL_MODEL_DIR = '/tmp/model_dir'
LOCAL_BASE_MODEL_DIR = '/tmp/base_model_dir'
LOCAL_DATA_DIR = '/tmp/data'

# Huggingface files.
HF_MODEL_WEIGHTS_SUFFIX = '.bin'

# PEFT finetuning constants.
TEXT_TO_IMAGE_LORA = 'text-to-image-lora'
SEQUENCE_CLASSIFICATION_LORA = 'sequence-classification-lora'
CAUSAL_LANGUAGE_MODELING_LORA = 'causal-language-modeling-lora'
INSTRUCT_LORA = 'instruct-lora'

# Precision modes for loading model weights.
PRECISION_MODE_4 = '4bit'
PRECISION_MODE_8 = '8bit'
PRECISION_MODE_16 = 'float16'
PRECISION_MODE_32 = 'float32'
