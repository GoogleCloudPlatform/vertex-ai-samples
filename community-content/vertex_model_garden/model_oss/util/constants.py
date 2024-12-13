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
HP_LOSS_TAG = 'model_loss'

# Reported places.
REPORT_TO_NONE = 'none'
REPORT_TO_WANDB = 'wandb'
REPORT_TO_TENSORBOARD = 'tensorboard'

# HPT trial prefix.
TRIAL_PREFIX = 'trial_'

# ML uses from user input.
ML_USE_TRAINING = 'training'
ML_USE_VALIDATION = 'validation'
ML_USE_TEST = 'test'

# COCO json keys.
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

# GCS prefixes.
GCS_URI_PREFIX = 'gs://'
GCSFUSE_URI_PREFIX = '/gcs/'

LOCAL_EVALUATION_RESULT_DIR = '/tmp/evaluation_result_dir'
LOCAL_MODEL_DIR = '/tmp/model_dir'
LOCAL_LORA_DIR = '/tmp/lora_dir'
LOCAL_BASE_MODEL_DIR = '/tmp/base_model_dir'
LOCAL_DATA_DIR = '/tmp/data'
LOCAL_OUTPUT_DIR = '/tmp/output_dir'
LOCAL_PREDICTION_RESULT_DIR = '/tmp/prediction_result_dir'
SHARED_MEM_DIR = '/dev/shm'

# Huggingface files.
HF_MODEL_WEIGHTS_SUFFIX = '.bin'

# PEFT finetuning constants.
TEXT_TO_IMAGE = 'text-to-image'
TEXT_TO_IMAGE_LORA = 'text-to-image-lora'
TEXT_TO_IMAGE_DREAMBOOTH = 'text-to-image-dreambooth'
TEXT_TO_IMAGE_DREAMBOOTH_LORA = 'text-to-image-dreambooth-lora'
TEXT_TO_IMAGE_DREAMBOOTH_LORA_SDXL = 'text-to-image-dreambooth-lora-sdxl'
SEQUENCE_CLASSIFICATION_LORA = 'sequence-classification-lora'
MERGE_CAUSAL_LANGUAGE_MODEL_LORA = 'merge-causal-language-model-lora'
QUANTIZE_MODEL = 'quantize-model'
INSTRUCT_LORA = 'instruct-lora'
VALIDATE_DATASET_WITH_TEMPLATE = 'validate-dataset-with-template'
DEFAULT_TEXT_COLUMN_IN_DATASET = 'quote'
DEFAULT_TEXT_COLUMN_IN_QUANTIZATION_DATASET = 'text'
DEFAULT_INSTRUCT_COLUMN_IN_DATASET = 'text'

FINAL_CHECKPOINT_DIRNAME = 'checkpoint-final'

# ImageBind inference constants.
FEATURE_EMBEDDING_GENERATION = 'feature-embedding-generation'
ZERO_SHOT_CLASSIFICATION = 'zero-shot-classification'

# Precision modes for loading model weights.
PRECISION_MODE_2 = '2bit'
PRECISION_MODE_3 = '3bit'
PRECISION_MODE_4 = '4bit'
PRECISION_MODE_8 = '8bit'
PRECISION_MODE_FP8 = 'float8'  # to use fbgemm_fp8 quantization
PRECISION_MODE_16 = 'float16'
PRECISION_MODE_16B = 'bfloat16'
PRECISION_MODE_32 = 'float32'

# Quantization modes.
GPTQ = 'gptq'
AWQ = 'awq'

# AWQ versions.
GEMM = 'GEMM'
GEMV = 'GEMV'

# Environment variable keys.
PRIVATE_BUCKET_ENV_KEY = 'AIP_PRIVATE_BUCKET_NAME'

# Kfp pipeline constants.
TFVISION_TRAIN_OUTPUT_ARTIFACT_NAME = 'checkpoint_dir'

# Vertex IOD type.
AUTOML = 'AUTOML'
MODEL_GARDEN = 'MODEL_GARDEN'

# LRU Disk Cache constants.
MD5_HASHMAP_FILENAME = 'md5_hashmap.json'

# Prediction request keys.
PREDICT_INSTANCE_KEY = 'instances'
PREDICT_INSTANCE_IMAGE_KEY = 'image'
PREDICT_INSTANCE_POSE_IMAGE_KEY = 'pose_image'
PREDICT_INSTANCE_TEXT_KEY = 'text'
PREDICT_INSTANCE_PROMPT_KEY = 'prompt'

PREDICT_PARAMETERS_KEY = 'parameters'
PREDICT_PARAMETERS_NUM_INFERENCE_STEPS_KEY = 'num_inference_steps'
PREDICT_PARAMETERS_HEIGHT_KEY = 'height'
PREDICT_PARAMETERS_WIDTH_KEY = 'width'
PREDICT_PARAMETERS_GUIDANCE_SCALE_KEY = 'guidance_scale'
PREDICT_PARAMETERS_NEGATIVE_PROMPT_KEY = 'negative_prompt'
PREDICT_PARAMETERS_LORA_ID_KEY = 'lora_id'
PREDICT_PARAMETERS_IGNORE_LORA_CACHE_KEY = 'ignore_lora_cache'

PREDICT_OUTPUT_KEY = 'output'
