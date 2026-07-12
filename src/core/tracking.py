import logging

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

BBox = tuple[float, float, float, float]


def iou(bb1: BBox, bb2: BBox) -> float:
    x1, y1 = max(bb1[0], bb2[0]), max(bb1[1], bb2[1])
    x2, y2 = min(bb1[2], bb2[2]), min(bb1[3], bb2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    area2 = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


class Track:
    """
    A single continuous bat trajectory. Frame-to-frame continuity is
    maintained via a constant-velocity Kalman filter on bounding box
    center + size, which lets the track survive brief detection gaps
    without breaking.
    """

    _next_id: int = 1

    def __init__(self, bbox: BBox, frame_idx: int) -> None:
        self.id: int = Track._next_id
        Track._next_id += 1
        self._kf: KalmanFilter = self._init_kf(bbox)
        self.time_since_update: int = 0
        self.frames: list[int] = [frame_idx]
        # frame_idx -> (cx, cy) pixel centroid; used later for the ROI-crossing test
        self.history_frame_to_centroid: dict[int, tuple[float, float]] = {frame_idx: self._centroid(bbox)}

    @staticmethod
    def _centroid(bbox: BBox) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def _init_kf(self, bbox: BBox) -> KalmanFilter:
        # state: [cx, cy, w, h, vcx, vcy, vw, vh]
        kf = KalmanFilter(dim_x=8, dim_z=4)
        cx, cy = self._centroid(bbox)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        kf.x[:4] = np.array([[cx], [cy], [w], [h]])
        kf.F = np.eye(8)
        for i in range(4):
            kf.F[i, i + 4] = 1.0
        kf.H = np.zeros((4, 8))
        kf.H[:4, :4] = np.eye(4)
        kf.P *= 10.0
        kf.R *= 1.0
        kf.Q *= 0.01
        return kf

    def predict(self) -> BBox:
        self._kf.predict()
        return self._state_to_bbox()

    def update(self, bbox: BBox, frame_idx: int) -> None:
        cx, cy = self._centroid(bbox)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        self._kf.update(np.array([[cx], [cy], [w], [h]]))
        self.time_since_update = 0
        self.frames.append(frame_idx)
        self.history_frame_to_centroid[frame_idx] = (cx, cy)

    def _state_to_bbox(self) -> BBox:
        cx, cy, w, h = self._kf.x[:4].flatten()
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


class SimpleSORT:
    """
    Minimal SORT implementation: Kalman prediction + Hungarian/IoU matching
    per frame.
    """

    def __init__(self, max_age: int = 2, iou_threshold: float = 0.2) -> None:
        """Initialize the tracker with parameters for track management.

        Args:
            max_age (int): Maximum number of frames a track can remain unmatched
                before being considered finished.

            iou_threshold (float): Minimum IoU required to match a detection to
                an existing track.
        """

        self._logger = logging.getLogger(__name__)
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.active_tracks: list[Track] = []
        self.finished_tracks: list[Track] = []

    def step(self, detections: list[BBox], frame_idx: int) -> None:
        predicted_bboxes = [t.predict() for t in self.active_tracks]

        row_ind, col_ind, cost = np.array([], dtype=int), np.array([], dtype=int), None
        if self.active_tracks and detections:
            cost = np.zeros((len(self.active_tracks), len(detections)))
            for i, pbb in enumerate(predicted_bboxes):
                for j, det in enumerate(detections):
                    cost[i, j] = 1.0 - iou(pbb, det)
            row_ind, col_ind = linear_sum_assignment(cost)

        matched_tracks, matched_dets = set(), set()
        for r, c in zip(row_ind, col_ind, strict=True):
            if cost is None:
                raise ValueError("Cost matrix is None.")

            if cost[r, c] <= (1.0 - self.iou_threshold):
                self.active_tracks[r].update(detections[c], frame_idx)
                matched_tracks.add(r)
                matched_dets.add(c)

        for i, t in enumerate(self.active_tracks):
            if i not in matched_tracks:
                t.time_since_update += 1

        for j, det in enumerate(detections):
            if j not in matched_dets:
                self.active_tracks.append(Track(det, frame_idx))

        still_active = []
        for t in self.active_tracks:
            if t.time_since_update > self.max_age:
                self.finished_tracks.append(t)
            else:
                still_active.append(t)
        self.active_tracks = still_active

    def finalize(self) -> list[Track]:
        self.finished_tracks.extend(self.active_tracks)
        self.active_tracks = []
        return self.finished_tracks
