"""Lib for handling video prediction requests.

The VCN inference algorithm is as follows:
1. Find all video frames within the given clip according to the sampling FPS.
2. Create possibly overlapping sliding windows according to the num_frames and
overlap_frames parameters. The last window might have a larger overlap if it
doesn't exactly fit.
3. Run model inference on each sliding window and compute softmax to obtain
probabilities.
4. Average the probabilities over all sliding windows.

The VAR inference algorithm is very similar to VCN, with a few differences:
1. The last sliding window is discarded if it does not exactly fit.
2. Instead of averaging, the postprocessing consists of temporal nonmaximal
suppression and removing background and low-confidence labels.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Dict, Optional, Sequence, Union, cast

from absl import logging
import cv2
import numpy as np
import tensorflow as tf

from util import constants
from util import fileutils


_JSON_LABEL_KEY = 'label'
_JSON_GCS_URI_KEY = 'content'
_JSON_CONFIDENCE_KEY = 'confidence'
_JSON_START_TIME_KEY = 'timeSegmentStart'
_JSON_END_TIME_KEY = 'timeSegmentEnd'
_BACKGROUND_LABEL = 0
_JSON_REQUIRED_KEYS = [
    _JSON_GCS_URI_KEY,
    _JSON_START_TIME_KEY,
    _JSON_END_TIME_KEY,
]
_IMAGE_WIDTH = int(os.environ.get('IMAGE_WIDTH', '172'))
_IMAGE_HEIGHT = int(os.environ.get('IMAGE_HEIGHT', '172'))


@dataclasses.dataclass
class DetectionOutput:
  timestamp: float
  label: int
  confidence: float

  def to_json_obj(self) -> Dict[str, Union[int, float]]:
    """Encodes self as a dict for JSON serialization."""
    return {
        _JSON_LABEL_KEY: self.label,
        _JSON_START_TIME_KEY: self.timestamp,
        _JSON_END_TIME_KEY: self.timestamp,
        _JSON_CONFIDENCE_KEY: self.confidence,
    }


def create_detection_output(
    timestamp: float, predictions: np.ndarray
) -> DetectionOutput:
  label = np.argmax(predictions).item()
  confidence: float = predictions[label].item()
  return DetectionOutput(timestamp, label, confidence)


class SlidingWindow:
  """Represents a sliding window with start / end timestamps."""

  def __init__(self, fps: float, frames: Sequence[int]):
    if not frames:
      raise ValueError('Sliding window cannot be empty.')
    self.frames = frames
    self.start_time = frames[0] / fps
    self.end_time = frames[-1] / fps
    self.frame_data: list[Optional[np.ndarray]] = []
    self.clear_frame_data()

  def load_cache_from(self, other: SlidingWindow) -> int:
    """Loads cache from another sliding window if possible."""
    cache_count = 0
    for i, frame in enumerate(self.frames):
      try:
        other_idx = other.frames.index(frame)
        self.frame_data[i] = other.frame_data[other_idx]
        cache_count += 1
      except ValueError:
        # Cache miss.
        pass
    return cache_count

  def load_frames(self, video: Any) -> Sequence[np.ndarray]:
    """Loads frames of this sliding window from a video."""
    for i, frame in enumerate(self.frames):
      if self.frame_data[i] is None:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, frame = video.read()
        if not ret:
          raise IOError(f'Failed to read video at frame {frame}.')
        self.frame_data[i] = cv2.resize(frame, (_IMAGE_WIDTH, _IMAGE_HEIGHT))
    return cast(Sequence[np.ndarray], self.frame_data)

  def clear_frame_data(self) -> None:
    """Clears frame data of this sliding window to reduce memory usage."""
    self.frame_data: list[Optional[np.ndarray]] = [None] * len(self)

  def __len__(self) -> int:
    return len(self.frames)

  @property
  def middle_timestamp(self) -> float:
    return (self.start_time + self.end_time) / 2


def _get_sliding_windows(
    frames: Sequence[int],
    original_fps: float,
    window_size: int,
    overlap: int,
    flush_last_window: bool,
) -> Sequence[SlidingWindow]:
  """Computes a list of sliding windows from frames.

  Args:
    frames: A list of frame indices.
    original_fps: Frames per second of the original video.
    window_size: Number of frames in a single window.
    overlap: Number of overlapping frames in adjacent windows.
    flush_last_window: Where to flush the last window if there are not enough
      frames left.

  Returns:
    A list of sliding windows, each has a list of frame indices. The last two
    windows might have a larger overlap if the last window does not exactly fit
    and flush_last_window is set to True.

  Raises:
    ValueError: Arguments are invalid.
  """
  if window_size <= overlap:
    raise ValueError(f'Window size {window_size} <= overlap {overlap}')
  total_frames = len(frames)
  windows: list[SlidingWindow] = []
  for i in range(0, total_frames, window_size - overlap):
    if i == 0 or i + window_size <= total_frames:
      windows.append(SlidingWindow(original_fps, frames[i : i + window_size]))
    elif i + overlap < total_frames and flush_last_window:
      # Some frames in this window are not covered by the previous window.
      windows.append(
          SlidingWindow(
              original_fps, frames[total_frames - window_size : total_frames]
          )
      )
  return windows


def _sample_frame_indices(
    start_time: float,
    end_time: float,
    original_fps: float,
    sample_fps: float,
    max_frames: int,
    padding_left: int = 0,
    padding_right: int = 0,
) -> Sequence[int]:
  """Samples frames from start_time to end_time by sample_fps.

  Args:
    start_time: Start timestamp in seconds.
    end_time: End timestamp in seconds.
    original_fps: Frames per second of the original video.
    sample_fps: Number of frames to sample per second.
    max_frames: Total number of frames in the video.
    padding_left: Padding to add to the start in frames. Padded frames will be
      duplicates of the first frame.
    padding_right: Padding to add to the end in frames. Padded frames will be
      duplicates of the last frame.

  Returns:
    A list of sampled frame indices.
  """
  ret = [
      min(max_frames - 1, round(t * original_fps))
      for t in np.arange(start_time, end_time, 1 / sample_fps)
  ]
  if ret:
    ret = [ret[0]] * padding_left + ret + [ret[-1]] * padding_right
  return ret


class VideoPredictionExecutor:
  """Represents a Video prediction request with a video clip."""

  def __init__(self, gcs_uri: str, start_time: float, end_time: float):
    self._gcs_uri = gcs_uri
    self._start_time = start_time
    self._end_time = end_time
    self.windows: Sequence[SlidingWindow] = []
    self._last_window: SlidingWindow = None

  def _read_frames_from_window(
      self, video: Any, new_window: SlidingWindow
  ) -> Sequence[np.ndarray]:
    """Reads video frames from the new window.

    Args:
      video: Video loaded with cv2.
      new_window: A list of sorted frame indices in the new window.

    Returns:
      Frame data from the video as a list of numpy arrays.

    Raises:
      IOError: Failed to read video.
    """
    # Caches frames as much as possible.
    if self._last_window is not None:
      cache_count = new_window.load_cache_from(self._last_window)
      logging.info('Cached %d frames.', cache_count)
      self._last_window.clear_frame_data()
    self._last_window = new_window
    return new_window.load_frames(video)

  def _predict(
      self, model: Any, video: Any, batched_windows: Sequence[SlidingWindow]
  ) -> np.ndarray:
    """Run model inference on specific frames of a video.

    Args:
      model: MoViNet model.
      video: Video loaded with cv2.
      batched_windows: A batch of sliding windows to predict. Each element is an
        integer frame index. Must have equal number of frames in each window.

    Returns:
      Prediction results.

    Raises:
      ValueError: Batched windows are not sorted, or do not have equal number of
        frames in each window.
      IOError: Failed to read video.
    """
    if any(
        (
            len(window) != len(batched_windows[0])
            for window in batched_windows[1:]
        )
    ):
      raise ValueError(
          'Batched windows do not have equal number of frames in each window.'
      )
    batch = []
    logging.info('Loading video frames...')
    for window in batched_windows:
      logging.info('Predict frames: %s', window.frames)
      frames = self._read_frames_from_window(video, window)
      batch.append(frames)
    input_tensor = tf.convert_to_tensor(batch, dtype=tf.float32) / 255.0
    logging.info('Predict: Input tensor shape %s', input_tensor.shape)
    predictions = model({'image': input_tensor})
    logging.info('Running softmax on predictions...')
    predictions = tf.nn.softmax(predictions, axis=1)
    return predictions.numpy()

  def get_prediction(
      self,
      model: Any,
      batch_size: int,
      fps: float,
      num_frames: int,
      overlap_frames: int,
      objective: str,
  ) -> Sequence[np.ndarray]:
    """Predicts the video clip with the model.

    Args:
      model: The loaded MoViNet model.
      batch_size: Batch size for prediction.
      fps: Video sampling FPS.
      num_frames: Number of frames in a single predictions. If the model is
        exported with a fixed input shape, this must match its num_frames
        dimension.
      overlap_frames: Number of overlapping frames of consecutive sliding
        windows.
      objective: A string `vcn` or `var`.

    Returns:
      A list of floats as the prediction response.

    Raises:
      IOError: The video fails to load.
      ValueError: Some arguments are invalid.
    """
    if objective not in [
        constants.OBJECTIVE_VIDEO_CLASSIFICATION,
        constants.OBJECTIVE_VIDEO_ACTION_RECOGNITION,
    ]:
      raise ValueError(f'{objective} objective is not supported.')

    # cv2 expects a local path so we need to download the video from GCS.
    local_file_path = fileutils.generate_tmp_path(
        os.path.splitext(self._gcs_uri)[1]
    )
    logging.info('Downloading %s to %s...', self._gcs_uri, local_file_path)
    fileutils.download_gcs_file_to_local(self._gcs_uri, local_file_path)
    logging.info('Download %s complete.', self._gcs_uri)

    # Loads video.
    video = cv2.VideoCapture(local_file_path)
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    original_fps = video.get(cv2.CAP_PROP_FPS)
    if not original_fps:
      # 0 or None indicates the video is invalid.
      raise IOError(f'Failed to load {self._gcs_uri}.')
    video_length = total_frames / original_fps
    self._start_time = max(0, self._start_time)
    self._end_time = min(video_length, self._end_time)
    padding = (
        (num_frames // 2)
        if objective == constants.OBJECTIVE_VIDEO_ACTION_RECOGNITION
        else 0
    )

    # Computes sliding windows.
    frame_indices = _sample_frame_indices(
        self._start_time,
        self._end_time,
        original_fps,
        fps,
        total_frames,
        padding,
        padding,
    )
    logging.info('Frame indices: %s', frame_indices)
    self.windows = _get_sliding_windows(
        frame_indices,
        original_fps,
        num_frames,
        overlap_frames,
        objective != 'var',
    )
    if not self.windows:
      raise ValueError(
          f'No sliding windows found from {self._start_time} to'
          f' {self._end_time}.'
      )
    self._last_window = None

    # Runs inference.
    predictions = []
    for i in range(0, len(self.windows), batch_size):
      predictions.extend(
          self._predict(model, video, self.windows[i : i + batch_size])
      )
    return predictions


def parse_request(req_json: Any) -> VideoPredictionExecutor:
  """Parses VideoPredictionExecutor from request JSON object.

  Args:
    req_json: Request JSON object.

  Returns:
    Parsed VideoPredictionExecutor.

  Raises:
    ValueError: Request JSON object is invalid.
  """
  for key in _JSON_REQUIRED_KEYS:
    if key not in req_json:
      raise ValueError(f'{key} not found in {req_json}.')
  gcs_uri = req_json[_JSON_GCS_URI_KEY]
  start_time = float(req_json[_JSON_START_TIME_KEY].removesuffix('s'))
  end_time = float(req_json[_JSON_END_TIME_KEY].removesuffix('s'))
  return VideoPredictionExecutor(gcs_uri, start_time, end_time)


def postprocess_vcn(predictions: Sequence[np.ndarray]) -> Sequence[float]:
  """Aggregates VCN predictions of sliding windows."""
  return np.mean(predictions, axis=0).tolist()


def temporal_nonmaximal_suppression(
    detections: Sequence[DetectionOutput], min_gap_time: float
) -> Sequence[DetectionOutput]:
  """Nonmaximal suppression for key frame detection.

  For consecutive packets of the same label within a pre-defined duration, we
  only keep the one with the highest confidence score. Such duration can be
  determined by performing data analysis on users' dataset.

  Args:
    detections: A list of DetectionOutputs.
    min_gap_time: Minimum time between consecutive key frames of the same label
      in seconds.

  Returns:
    DetectionOutput after nonmaximal suppression sorted in ascending timestamps.
  """
  max_label = max([detection.label for detection in detections])
  prev_detections: list[Optional[DetectionOutput]] = [None] * (max_label + 1)
  ret: list[DetectionOutput] = []
  by_time = lambda x: x.timestamp
  for detection in sorted(detections, key=by_time):
    prev_detection = prev_detections[detection.label]
    prev_detections[detection.label] = detection
    if not prev_detection:
      continue
    if detection.timestamp - prev_detection.timestamp > min_gap_time:
      ret.append(prev_detection)
      continue
    detection.confidence = max(detection.confidence, prev_detection.confidence)
  ret.extend((d for d in prev_detections if d is not None))
  return sorted(ret, key=by_time)


def postprocess_var(
    windows: Sequence[SlidingWindow],
    predictions: Sequence[np.ndarray],
    confidence_threshold: float,
    min_gap_time: float,
) -> Sequence[Dict[str, Any]]:
  """Generates a list of detected keyframes from sliding window predictions.

  Args:
    windows: Sliding windows.
    predictions: A list of predictions of sliding windows.
    confidence_threshold: Only probabilities greater than this threshold will
      contribute to the final result.
    min_gap_time: Minimum time between consecutive key frames of the same label
      in seconds. Used in temporal nonmaximal suppression.

  Returns:
    A sequence of dictionaries, each item has the following keys:
    - label: Integer label of the detection result.
    - timeSegmentStart: Start timestamp in seconds.
    - timeSegmentEnd: End timestamp in seconds. Always equals timeSegmentStart.
  """
  if len(windows) != len(predictions):
    raise ValueError('Mismatched # of windows with # of predictions.')

  # Creates detection results from windows, filtering out the background label.
  detections = [
      create_detection_output(window.middle_timestamp, predictions[i])
      for i, window in enumerate(windows)
  ]

  # Temporal nonmaximal suppression.
  detections = temporal_nonmaximal_suppression(detections, min_gap_time)

  # Filters out ones with low confidence and the background label.
  return [
      x.to_json_obj()
      for x in detections
      if x.label != _BACKGROUND_LABEL and x.confidence > confidence_threshold
  ]
