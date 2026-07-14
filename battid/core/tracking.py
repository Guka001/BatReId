import logging

import numpy as np
import supervision as sv


class Track:
    """
    Represents a single tracked object's trajectory across frames.

    Accumulates the sequence of frame indices in which the object was
    detected along with its pixel-space centroid at each frame, enabling
    later spatial analysis such as ROI-crossing tests.
    """

    def __init__(self, track_id: int) -> None:
        self.id: int = track_id
        self.frames: list[int] = []
        self.history_frame_to_centroid: dict[int, tuple[float, float]] = {}

    def add(self, frame_idx: int, centroid: tuple[float, float]) -> None:
        self.frames.append(frame_idx)
        self.history_frame_to_centroid[frame_idx] = centroid


class BatTracker:
    """
    Assigns persistent IDs to per-frame bounding-box detections over time.

    Wraps a ByteTrack instance to associate detections across frames into
    tracks, maintaining a `Track` object per unique tracked ID with its
    full frame-by-frame centroid history.
    """

    def __init__(
        self,
        frame_rate: int = 25,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
    ) -> None:
        """
        Args:
            frame_rate: Video frame rate (fps), used by ByteTrack to scale
                its internal buffers.
            track_activation_threshold: Minimum detection confidence required
                to activate a new track.
            lost_track_buffer: Number of frames a track is kept alive without
                a matching detection before being dropped.
            minimum_matching_threshold: Minimum IoU (or similarity) required
                to match a detection to an existing track.
        """

        self._logger: logging.Logger = logging.getLogger(__name__)
        self._byte_track: sv.ByteTrack = sv.ByteTrack(
            frame_rate=frame_rate,
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
        )
        self._tracks: dict[int, Track] = {}

    def step(self, detections: list[tuple[float, float, float, float, float]], frame_idx: int) -> None:
        if detections:
            xyxy = np.array([d[:4] for d in detections], dtype=float)
            confidence = np.array([d[4] for d in detections], dtype=float)
        else:
            xyxy = np.empty((0, 4), dtype=float)
            confidence = np.empty((0,), dtype=float)

        sv_detections = sv.Detections(xyxy=xyxy, confidence=confidence)
        tracked = self._byte_track.update_with_detections(sv_detections)

        if tracked.tracker_id is None:
            return

        for box, track_id in zip(tracked.xyxy, tracked.tracker_id, strict=True):
            track_id = int(track_id)
            x1, y1, x2, y2 = box
            centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)  # bbox center (u,v)
            if track_id not in self._tracks:
                self._tracks[track_id] = Track(track_id)
            self._tracks[track_id].add(frame_idx, centroid)  # at frame x, bat y was at (u,v)

    def finalize(self) -> list[Track]:
        return list(self._tracks.values())
