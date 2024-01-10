"""Custom handler for video object tracking models."""

import logging
import os
import tempfile
from typing import Any, List, Optional, Tuple

from bytetracker import BYTETracker
import cv2
from google.cloud import aiplatform
import imageio.v2 as iio
from PIL import Image
import tensorflow as tf
import torch
from ts.torch_handler.base_handler import BaseHandler

from util import commons
from util import fileutils
import visualization_utils

_VIDEO_URI = "video_uri"
_DATA = "data"
_TRACK_THRESHOLD = 0.45
_TRACK_BUFFER = 25
_MATCH_THRESHOLD = 0.8


class VideoObjectTrackingHandler(BaseHandler):
  """Custom handler for video object tracking models."""

  def initialize(self, context: Any) -> None:
    properties = context.system_properties
    self.map_location = (
        "cuda"
        if torch.cuda.is_available() and properties.get("gpu_id") is not None
        else "cpu"
    )
    self.device = torch.device(
        self.map_location + ":" + str(properties.get("gpu_id"))
        if torch.cuda.is_available() and properties.get("gpu_id") is not None
        else self.map_location
    )
    self.manifest = context.manifest

    detection_endpoint_id = os.environ.get("DETECTION_ENDPOINT", None)
    if detection_endpoint_id:
      self.detection_endpoint = aiplatform.Endpoint(detection_endpoint_id)
      endpoint_label_map = os.environ.get("LABEL_MAP", None)
      if endpoint_label_map:
        endpoint_label_map_file = endpoint_label_map
        self.label_map = commons.get_label_map(endpoint_label_map_file)
      else:
        raise ValueError(
            "LABEL MAP must be provided with DETECTION ENDPOINT:"
            f" {self.detection_endpoint}"
        )

    self.track_thresh = os.environ.get("TRACK_THRESHOLD", _TRACK_THRESHOLD)
    self.track_buffer = os.environ.get("TRACK_BUFFER", _TRACK_BUFFER)
    self.match_thresh = os.environ.get("MATCH_THRESHOLD", _MATCH_THRESHOLD)
    self.save_video_results = bool(int(os.environ.get("SAVE_VIDEO_RESULTS", 0)))
    self.output_bucket = os.environ.get("OUTPUT_BUCKET", None)
    if not self.output_bucket:
      raise ValueError("Empty Output Bucket.")
    self.initialized = True
    logging.info("Handler initialization done.")

  def preprocess(
      self, data: Any
  ) -> Tuple[Optional[List[str]], Optional[List[Image.Image]]]:
    """Preprocesses the input data.

    Args:
        data (Any): Input data.

    Returns:
        List of videos uris.
    """
    video_uris = None
    if _VIDEO_URI in data[0]:
      video_uris = [item[_VIDEO_URI] for item in data]
    # TorchServe's default handlers expect each instance
    # to be wrapped in a data field for batch prediction.
    if _DATA in data[0]:
      video_uris = [item[_DATA][_VIDEO_URI] for item in data]
    return video_uris

  def inference(self, data: Any, *args, **kwargs) -> List[Any]:
    """Runs object detection and tracking inference on a video frame by frame.

    If using yolo detection, the function uses the ultralytics yolo models for
    IOD, otherwise it uses the provided IOD endpoint and associated the selected
    tracking method to the detections.

    Args:
        data: List of video files.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        List of video frame annotations and/or output decorated video uris.
    """
    gcs_video_files = data
    video_preds = []
    for gcs_video_file in gcs_video_files:
      results_info = {}
      temp_text_file = tempfile.NamedTemporaryFile(delete=False, mode="w+t")
      local_video_file_name, remote_video_file_name = (
          fileutils.download_video_from_gcs_to_local(gcs_video_file)
      )
      remote_text_file_name = remote_video_file_name.replace(
          "overlay.mp4", "annotations.txt"
      )

      cap = cv2.VideoCapture(local_video_file_name)
      fps = cap.get(cv2.CAP_PROP_FPS)
      temp_local_video_file_name = fileutils.get_output_video_file(
          local_video_file_name
      )
      if self.save_video_results:
        self.video_writer = iio.get_writer(
            temp_local_video_file_name,
            format="FFMPEG",
            mode="I",
            fps=float(fps),
            codec="h264",
        )
      self.tracker = BYTETracker(
          track_thresh=self.track_thresh,
          track_buffer=self.track_buffer,
          match_thresh=self.match_thresh,
          frame_rate=fps,
      )
      frame_idx = 1
      while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
          break

        dets_np = commons.get_object_detection_endpoint_predictions(
            self.detection_endpoint, frame
        )
        dets_tf = tf.convert_to_tensor(dets_np)
        online_targets = self.tracker.update(dets_tf, None)
        if online_targets.size > 0:
          frame = visualization_utils.overlay_tracking_results(
              frame_idx,
              frame,
              online_targets,
              label_map=self.label_map,
              temp_text_file_path=temp_text_file.name,
          )

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.save_video_results:
          self.video_writer.append_data(frame)
        logging.info(
            "Finished processing frame %s for video %s.",
            frame_idx,
            gcs_video_file,
        )
        frame_idx += 1

      self.video_writer.close()
      cap.release()
      if self.save_video_results:
        fileutils.upload_video_from_local_to_gcs(
            self.output_bucket,
            local_video_file_name,
            remote_video_file_name,
            temp_local_video_file_name,
        )
        results_info["output_video"] = "{}/{}".format(
            self.output_bucket, remote_video_file_name
        )

      fileutils.release_text_assets(
          self.output_bucket,
          temp_text_file.name,
          remote_text_file_name,
      )
      results_info["annotations"] = "{}/{}".format(
          self.output_bucket, remote_text_file_name
      )
      video_preds.append(results_info)

    return video_preds

  def handle(self, data: Any, context: Any) -> List[Any]:
    model_input = self.preprocess(data)
    model_out = self.inference(model_input)
    output = self.postprocess(model_out)
    return output

  def postprocess(self, inference_result: List[Any]) -> List[Any]:
    return inference_result
