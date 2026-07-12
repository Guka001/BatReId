from pathlib import Path

import cv2
from shapely.geometry import LineString, Point
from shapely.geometry import box as shapely_box

from .tracking import Track


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
    """Test whether a track's path ever enters the marked ROI."""
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
