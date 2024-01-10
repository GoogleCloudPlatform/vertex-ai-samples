"""Detectron2 trainer helper functions."""

import argparse
from detectron2.data.datasets import register_coco_instances


def extend_parser_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
  """Adds additional model-garden related arguments."""
  parser.add_argument(
      "--train_dataset_name",
      required=False,
      default="",
      type=str,
      help=(
          "The training dataset name for registration. "
          "For example: 'balloon_train'."
      ),
  )
  parser.add_argument(
      "--train_coco_json_file",
      required=False,
      default="",
      type=str,
      help="The path to the training coco-json format file.",
  )
  parser.add_argument(
      "--train_image_root",
      required=False,
      default="",
      type=str,
      help="The path to the root folder containing the training images.",
  )
  parser.add_argument(
      "--val_dataset_name",
      required=False,
      default="",
      type=str,
      help=(
          "The validation dataset name for registration. "
          "For example: 'balloon_val'."
      ),
  )
  parser.add_argument(
      "--val_coco_json_file",
      required=False,
      default="",
      type=str,
      help="The path to the validation coco-json format file.",
  )
  parser.add_argument(
      "--val_image_root",
      required=False,
      default="",
      type=str,
      help="The path to the root folder containing the validation images.",
  )
  parser.add_argument(
      "--output_dir",
      required=True,
      type=str,
      help="The path to the output directory.",
  )
  # Add hyper-parameter tuning related variables.
  parser.add_argument(
      "--lr",
      type=float,
      default=0.00025,
      help="The learning rate to be tuned.",
  )
  parser.add_argument(
      "--hp_eval_task",
      type=str,
      choices=["bbox", "segm"],
      default="bbox",
      help="The task choice for HP tuning.",
  )
  return parser


def register_dataset(args: argparse.Namespace):
  """Register the input dataset in Detectron2 Coco format."""
  if args.train_dataset_name:
    register_coco_instances(
        name=args.train_dataset_name,
        metadata={},
        json_file=args.train_coco_json_file,
        image_root=args.train_image_root,
    )
  if args.val_dataset_name:
    register_coco_instances(
        name=args.val_dataset_name,
        metadata={},
        json_file=args.val_coco_json_file,
        image_root=args.val_image_root,
    )