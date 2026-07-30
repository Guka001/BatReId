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
