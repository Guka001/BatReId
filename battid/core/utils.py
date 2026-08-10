from pathlib import Path

import cv2
import torch
from shapely.geometry import LineString, Point
from shapely.geometry import box as shapely_box

from battid.models.tracking import Track


def select_roi_from_video(video_path: Path) -> dict[str, int]:
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read a frame from {video_path}")

    print("Draw the illumination ROI rectangle, then press ENTER/SPACE. Press 'c' to cancel.")
    x, y, w, h = cv2.selectROI("Mark illumination ROI", frame, showCrosshair=True)
    cv2.destroyAllWindows()

    if w == 0 or h == 0:
        raise RuntimeError("No ROI was selected.")

    return {"x1": int(x), "y1": int(y), "x2": int(x + w), "y2": int(y + h)}


def track_crosses_roi(track: Track, roi: dict[str, int]) -> bool:
    """Test whether a track's path ever enters the marked ROI.

    Args:
        track (Track): The track to test.
        roi (dict[str, int]): The Region of interest

    Returns:
        bool: True if the track crosses the ROI, False otherwise.
    """

    rect = shapely_box(roi["x1"], roi["y1"], roi["x2"], roi["y2"])
    frames_sorted = sorted(track.history_frame_to_centroid.keys())

    if len(frames_sorted) == 1:
        pt = Point(track.history_frame_to_centroid[frames_sorted[0]])
        return bool(rect.intersects(pt))

    for f1, f2 in zip(frames_sorted[:-1], frames_sorted[1:], strict=True):
        p1 = track.history_frame_to_centroid[f1]
        p2 = track.history_frame_to_centroid[f2]
        segment = LineString([p1, p2])
        if segment.intersects(rect):
            return True
    return False


def compute_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """
    Computes the IOU between two object coordinates.

    Args:
        box_a (tuple[float, float, float, float]): Bounding-box coordinates of the first object
        box_b (tuple[float, float, float, float]): Bounding-box coordinates of the second object

    Returns:
        (float): The IOU score
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area <= 0.0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0.0 else 0.0


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"

    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes or hours:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")

    return " ".join(parts)


def get_detection_workers(max_cpu_workers: int) -> int:
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        return 1
    return max_cpu_workers


def pad_and_clamp_bbox(
    crop_padding_ratio: float,
    bbox: tuple[float, float, float, float],
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    """
    Expands a bbox by `self._crop_padding_ratio` of its own width/height on
    each side, then clamps the result to the image bounds.

    Args:
        crop_padding_ratio: The crop padding ratio.
        bbox: (x1, y1, x2, y2) in pixel coordinates.
        img_w: Width of the source frame, in pixels.
        img_h: Height of the source frame, in pixels.

    Returns:
        (x1, y1, x2, y2) as ints, padded and clamped to [0, img_w] x [0, img_h].
    """
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1

    pad_x = box_w * crop_padding_ratio
    pad_y = box_h * crop_padding_ratio

    x1 -= pad_x
    y1 -= pad_y
    x2 += pad_x
    y2 += pad_y

    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(img_w, int(round(x2)))
    y2 = min(img_h, int(round(y2)))

    return x1, y1, x2, y2


def bbox_touches_border(margin: int, x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> bool:
    """
    Checks whether a (padded, clamped) bbox sits close enough to the frame
    edge that the animal is likely cut off.

    Args:
        margin: Margin threshold
        x1, y1, x2, y2: Padded and clamped bbox coordinates, in pixels.
        img_w: Width of the source frame, in pixels.
        img_h: Height of the source frame, in pixels.

    Returns:
        (bool): True if the bbox is within the border margin on any side.
    """

    return x1 <= margin or y1 <= margin or x2 >= img_w - margin or y2 >= img_h - margin
