import logging
from itertools import combinations

import numpy as np
import supervision as sv

from battid.core.utils import compute_iou
from battid.models.tracking import OverlapEpisode, Track


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
        self._raw_overlap_frames: dict[tuple[int, int], list[tuple[int, float]]] = {}
        self._overlap_iou_threshold: float = 0.0

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

        # collect frame's (track_id, bbox) pairs
        frame_entries: list[tuple[int, tuple[float, float, float, float]]] = []

        for box, track_id in zip(tracked.xyxy, tracked.tracker_id, strict=True):
            track_id = int(track_id)
            x1, y1, x2, y2 = box
            centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)  # bbox center (u,v)
            bbox = (float(x1), float(y1), float(x2), float(y2))

            if track_id not in self._tracks:
                self._tracks[track_id] = Track(track_id)
            self._tracks[track_id].add(frame_idx, centroid, bbox)  # at frame x, bat y was at (u,v)

            frame_entries.append((track_id, bbox))

        for (id_a, box_a), (id_b, box_b) in combinations(frame_entries, 2):
            iou = compute_iou(box_a, box_b)
            if iou > self._overlap_iou_threshold:
                pair = (id_a, id_b) if id_a < id_b else (id_b, id_a)
                self._raw_overlap_frames.setdefault(pair, []).append((frame_idx, iou))

    def finalize(self) -> list[Track]:
        return list(self._tracks.values())

    def number_of_ids(self) -> int:
        return len(self._tracks)

    def overlap_episodes(self) -> list[OverlapEpisode]:
        """
        Groups per-frame overlaps into contiguous episodes per ID-pair.

        A run of frames counts as one episode as long as consecutive
        overlapping frame indices are adjacent (difference of 1).
        NOTE: a gap ends the episode and a new overlap starts a new one.
        """
        episodes: list[OverlapEpisode] = []

        for (id_a, id_b), frame_iou_pairs in self._raw_overlap_frames.items():
            frame_iou_pairs = sorted(frame_iou_pairs, key=lambda pair: pair[0])

            episode_frames: list[int] = []
            episode_ious: list[float] = []
            prev_frame: int | None = None

            for frame_idx, iou in frame_iou_pairs:
                if prev_frame is not None and frame_idx - prev_frame > 1:
                    episode = self._make_episode(id_a, id_b, episode_frames, episode_ious)
                    if episode is not None:
                        episodes.append(episode)
                    episode_frames = []
                    episode_ious = []
                episode_frames.append(frame_idx)
                episode_ious.append(iou)
                prev_frame = frame_idx

            episode = self._make_episode(id_a, id_b, episode_frames, episode_ious)
            if episode is not None:
                episodes.append(episode)

        return episodes

    @staticmethod
    def _make_episode(
        id_a: int,
        id_b: int,
        episode_frames: list[int],
        episode_ious: list[float],
    ) -> OverlapEpisode | None:
        """Builds an OverlapEpisode from an accumulated run of frames, or
        returns None if the run is empty."""
        if not episode_frames:
            return None
        return OverlapEpisode(
            track_id_a=id_a,
            track_id_b=id_b,
            start_frame=episode_frames[0],
            end_frame=episode_frames[-1],
            frame_ious=list(episode_ious),
        )

    def num_overlap_episodes(self) -> int:
        return len(self.overlap_episodes())
