"""Image and bounding box visualization util lib."""

from typing import Dict, List, Optional
import cv2
import numpy as np
from PIL import ImageColor


def draw_bounding_box_on_image(
    image: np.ndarray,
    ymin: float,
    xmin: float,
    ymax: float,
    xmax: float,
    color: str,
    thickness: int = 4,
    display_str_list: Optional[List[str]] = None,
) -> np.ndarray:
  """Draws a bounding box on an image.

  Args:
    image: The image to draw the bounding box on.
    ymin: The minimum y-coordinate of the bounding box.
    xmin: The minimum x-coordinate of the bounding box.
    ymax: The maximum y-coordinate of the bounding box.
    xmax: The maximum x-coordinate of the bounding box.
    color: The color of the bounding box.
    thickness: The thickness of the bounding box lines. Defaults to 4.
    display_str_list: List of strings to display in new line inside the bounding
      box.

  Returns:
    An image with a bounding box.
  """
  color = ImageColor.getrgb(color)
  cv2.rectangle(
      image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, thickness
  )
  # Display the strings below the bounding box
  for i, display_str in enumerate(display_str_list):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    thickness = 1
    text_width, text_height = cv2.getTextSize(
        display_str, font, scale, thickness
    )[0]
    text_bottom = int(ymin - i * text_height)
    text_left = int(xmin)
    cv2.rectangle(
        image,
        (text_left, text_bottom - text_height),
        (text_left + text_width, text_bottom),
        color,
        -1,
    )
    cv2.putText(
        image,
        display_str,
        (text_left, text_bottom),
        font,
        scale,
        (0, 0, 0),
        thickness,
    )

  return image


def draw_boxes(
    image: np.ndarray,
    boxes: List[List[float]],
    track_ids: List[int],
    class_names: List[str],
    scores: List[float],
    max_boxes: int = 40,
    min_score: float = 0.05,
) -> np.ndarray:
  """Overlays labeled boxes on an image with formatted scores and label names.

  Args:
      image: The image to overlay the boxes on.
      boxes: List of bounding box coordinates [xmin, ymin, xmax, ymax].
      track_ids: List of track IDs corresponding to each box.
      class_names: List of class names corresponding to each box.
      scores: List of scores corresponding to each box.
      max_boxes: Maximum number of boxes to draw. Defaults to 40.
      min_score: Minimum score threshold for displaying a box. Defaults to 0.05.

  Returns:
      PIL.Image.Image: The image with the labeled boxes overlay.
  """
  colors = list(ImageColor.colormap.values())
  for i in range(min(len(boxes), max_boxes)):
    if scores[i] >= min_score:
      xmin, ymin, xmax, ymax = boxes[i]
      display_str = "{}-{}: {}%".format(
          track_ids[i], class_names[i], int(100 * scores[i])
      )
      color = colors[hash(class_names[i]) % len(colors)]
      image = draw_bounding_box_on_image(
          image,
          ymin,
          xmin,
          ymax,
          xmax,
          color,
          display_str_list=[display_str],
      )
  return image


def overlay_tracking_results(
    frame_idx: int,
    image_np: np.ndarray,
    tracker_outputs: np.ndarray,
    model_names: Optional[Dict[int, str]] = None,
    label_map: Optional[Dict[str, Dict[int, str]]] = None,
    temp_text_file_path: Optional[str] = None,
) -> np.ndarray:
  """Overlays the results on the image.

  Args:
      frame_idx: frame index.
      image_np: Input image.
      tracker_outputs: Tracker outputs.
      model_names: label map for yolo models.
      label_map: label map for IOD detector model.
      temp_text_file_path: tempfile to save annotations.

  Returns:
      Decorated output frame.
  """
  dboxes = tracker_outputs[:, :4]
  dtracks = tracker_outputs[:, 4]
  dclasses = tracker_outputs[:, 5]
  dscores = tracker_outputs[:, 6]
  dclasses_as_text = []
  for detection_class in dclasses:
    if model_names:
      dclasses_as_text.append(model_names[int(detection_class)])
    elif label_map:
      dclasses_as_text.append(label_map["label_map"][int(detection_class)])
    else:
      dclasses_as_text.append("")

  plotted_img = np.array(
      draw_boxes(
          image=image_np,
          boxes=dboxes,
          track_ids=dtracks,
          class_names=dclasses_as_text,
          scores=dscores,
      )
  )
  track_anno_list = []
  for i, box in enumerate(dboxes):
    result_list = [dtracks[i], dscores[i], dclasses[i]]
    xyxy_anno = [np.round(item.item(), 2) for item in box]
    tracks_anno = (
        [frame_idx]
        + [np.round(item.item(), 2) for item in result_list]
        + xyxy_anno
    )
    track_anno_list.append(tracks_anno)
    with open(temp_text_file_path, "a") as file:
      file.write(", ".join([str(item) for item in tracks_anno]) + "\n")

  return plotted_img