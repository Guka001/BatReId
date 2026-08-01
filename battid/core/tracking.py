import logging
import math
from collections import deque
from itertools import combinations

import numpy as np
from boxmot.trackers.bbox.ocsort import OcSort

from battid.core.utils import compute_iou
from battid.models.tracking import OverlapEpisode, Track


class BatTracker:
    """
    Turns per-frame bat bounding-box detections into persistent, per-animal
    tracks across a video.

    Internally this wraps OC-SORT for the frame-to-frame association, and
    layers two extra behaviors on top:

    - Birth backfill: when a new track is born, it walks backward through
      recent low-confidence detections to recover the frames just before
      the animal became confident enough to trigger a track, so tracks
      start closer to when the bat actually entered the frame.
    - Duplicate suppression: after tracking, short-lived tracks that are
      fully explained by (contained within) another track for their whole
      lifetime are dropped, since these are usually tracker artifacts
      rather than real second animals.
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.3,
        min_hits: int = 1,
        min_low_confidence: float = 0.05,
        use_low_confidence_association: bool = True,
        max_junk_bbox_area_fraction: float = 0.9,
        enable_birth_backfill: bool = True,
        max_backfill_frames: int = 15,
        backfill_gate_fraction: float = 0.05,
        backfill_min_confidence: float = 0.0,
        enable_duplicate_suppression: bool = True,
        dedup_containment_threshold: float = 0.8,
        dedup_max_duplicate_track_length: float = math.inf,
    ) -> None:
        """
        Args:
            image_width: Width (pixels) of the frames detections are given in.

            track_activation_threshold: Minimum detection confidence to start a new track.

            lost_track_buffer: Number of frames a track survives without a new match before being dropped.

            minimum_matching_threshold: plain IoU, higher = stricter.

            min_hits: Number of consecutive matched frames a brand-new
                track needs before `step()` starts reporting it.
                Set to 1 (default) to emit from the very first accepted
                detection.

            min_low_confidence: Detections scoring above this and below
                `track_activation_threshold` can still extend an existing
                track (maps to OC-SORT's `min_conf`, used only when
                `use_low_confidence_association=True`).

            use_low_confidence_association: Whether to use low-confidence
                detections at all for continuity.

            max_junk_bbox_area_fraction: Detections whose bbox covers more
                than this fraction of the frame area are dropped before
                reaching the tracker (and before the backfill buffer, since
                both draw from the same filtered list).

            enable_birth_backfill: Switch for the backward-backfill
                feature described in the class docstring.

            max_backfill_frames: How many frames backward a fresh track is
                allowed to search for a spatially-consistent lead-in chain.
                Also bounds the size of the raw-detection rolling buffer.

            backfill_gate_fraction: Gate for accepting a candidate
                box during backfill. A candidate whose centroid is farther than this
                from the backward-extrapolated predicted position is
                rejected, which is what stops the walk at a genuine gap or
                a sharp, unpredictable turn rather than forcing a splice.

            backfill_min_confidence: Extra confidence floor applied only to
                backfill candidates (independent of `min_low_confidence`,
                which governs live OC-SORT continuation).

            enable_duplicate_suppression: Switch for the post-tracking
                duplicate-track pruning pass. Set False to keep every track
                OC-SORT produces, including phantom duplicates.

            dedup_containment_threshold: A short track is only pruned as a
                duplicate of another if, on *every* frame it exists in, its
                box is contained within the other track's box by at least
                this fraction.

            dedup_max_duplicate_track_length: A track is only eligible to be
                pruned as a duplicate if it has at most this many frames.
        """
        if min_low_confidence > track_activation_threshold:
            raise ValueError(
                "min_low_confidence must be <= track_activation_threshold "
                f"(got {min_low_confidence} > {track_activation_threshold})"
            )

        self._logger: logging.Logger = logging.getLogger(__name__)
        self._image_width: int = image_width
        self._image_height: int = image_height

        self._dummy_frame: np.ndarray = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        self._max_junk_bbox_area_fraction: float = max_junk_bbox_area_fraction

        self._tracker: OcSort = OcSort(
            det_thresh=track_activation_threshold,
            max_age=lost_track_buffer,
            min_hits=min_hits,
            iou_threshold=minimum_matching_threshold,
            min_conf=min_low_confidence,
            use_byte=use_low_confidence_association,
        )

        self._tracks: dict[int, Track] = {}
        self._raw_overlap_frames: dict[tuple[int, int], list[tuple[int, float]]] = {}
        self._overlap_iou_threshold: float = 0.0

        # --- birth-backfill state ---
        self._enable_birth_backfill: bool = enable_birth_backfill
        self._max_backfill_frames: int = max_backfill_frames
        self._backfill_gate_px: float = backfill_gate_fraction * math.hypot(image_width, image_height)
        self._backfill_min_confidence: float = backfill_min_confidence
        # rolling window of (frame_idx, filtered_detections); filtered_detections
        # is a list of (x1, y1, x2, y2, conf) tuples
        self._raw_buffer: deque[tuple[int, list[tuple[float, float, float, float, float]]]] = deque(
            maxlen=max_backfill_frames + 1
        )

        # --- duplicate-track pruning state (post-tracking, in finalize()) ---
        self._enable_duplicate_suppression: bool = enable_duplicate_suppression
        self._dedup_containment_threshold: float = dedup_containment_threshold
        self._dedup_max_duplicate_track_length: float = dedup_max_duplicate_track_length

    def step(self, detections: list[tuple[float, float, float, float, float]], frame_idx: int) -> None:
        """
        Feeds one frame's worth of detections into the tracker.

        Args:
            detections: List of (x1, y1, x2, y2, conf) tuples in pixel coordinates, where (x1, y1) is the
                top-left corner and (x2, y2) is the bottom-right corner of the bounding box,
                and conf is the detection confidence score.

            frame_idx: The index of the current frame (0-based).
        """
        frame_area = self._image_width * self._image_height
        filtered = [
            d for d in detections if (d[2] - d[0]) * (d[3] - d[1]) <= self._max_junk_bbox_area_fraction * frame_area
        ]

        if self._enable_birth_backfill:
            self._raw_buffer.append((frame_idx, filtered))

        if filtered:
            # OC-SORT's expected detection layout: x1, y1, x2, y2, conf, cls
            dets = np.array([[d[0], d[1], d[2], d[3], d[4], 0.0] for d in filtered], dtype=np.float32)
        else:
            dets = np.empty((0, 6), dtype=np.float32)

        tracked = self._tracker.update(dets, self._dummy_frame)

        if tracked.shape[0] == 0:
            return

        # collect frame's (track_id, bbox) pairs
        frame_entries: list[tuple[int, tuple[float, float, float, float]]] = []
        newly_born: list[int] = []

        for row in tracked:
            x1, y1, x2, y2, track_id = float(row[0]), float(row[1]), float(row[2]), float(row[3]), int(row[4])
            centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)  # bbox center (u,v)
            bbox = (x1, y1, x2, y2)

            if track_id not in self._tracks:
                self._tracks[track_id] = Track(track_id)
                newly_born.append(track_id)

            self._tracks[track_id].add(frame_idx, centroid, bbox)  # at frame x, bat y was at (u,v)
            frame_entries.append((track_id, bbox))

        for (id_a, box_a), (id_b, box_b) in combinations(frame_entries, 2):
            iou = compute_iou(box_a, box_b)
            if iou > self._overlap_iou_threshold:
                pair = (id_a, id_b) if id_a < id_b else (id_b, id_a)
                self._raw_overlap_frames.setdefault(pair, []).append((frame_idx, iou))

        if self._enable_birth_backfill:
            for track_id in newly_born:
                self._backfill_track_start(self._tracks[track_id], frame_idx)

    def _prune_duplicate_tracks(self) -> None:
        """
        Drops a track B as a spurious duplicate of another track A only if,
        across B's *entire* lifetime, the following holds:

          1. B is short-lived (<= `dedup_max_duplicate_track_length` frames).
          2. Every frame B exists in, A also has a box in. i.e. B never has an
             existence independent of A.
          3. On every one of those shared frames, B's box is contained
             within A's box by at least `dedup_containment_threshold`.
        """
        track_ids = sorted(self._tracks.keys())
        to_drop: set[int] = set()

        for b_id in track_ids:
            b = self._tracks[b_id]
            if len(b.frames) > self._dedup_max_duplicate_track_length:
                continue

            for a_id in track_ids:
                if a_id == b_id or a_id in to_drop:
                    continue
                a = self._tracks[a_id]

                if not all(f in a.history_frame_to_bbox for f in b.frames):
                    continue  # b has an existence independent of a

                fully_contained = True
                for f in b.frames:
                    bx1, by1, bx2, by2 = b.history_frame_to_bbox[f]
                    ax1, ay1, ax2, ay2 = a.history_frame_to_bbox[f]

                    ix1, iy1 = max(bx1, ax1), max(by1, ay1)
                    ix2, iy2 = min(bx2, ax2), min(by2, ay2)
                    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
                    containment = inter / b_area if b_area > 0 else 0.0

                    if containment < self._dedup_containment_threshold:
                        fully_contained = False
                        break

                if fully_contained:
                    to_drop.add(b_id)
                    break

        for b_id in to_drop:
            del self._tracks[b_id]
            self._raw_overlap_frames = {
                pair: frames for pair, frames in self._raw_overlap_frames.items() if b_id not in pair
            }

    def _frame_already_claimed(self, frame_idx: int, exclude_track_id: int) -> bool:
        """True if some *other* track already has a detection at `frame_idx`."""
        return any(
            frame_idx in track.history_frame_to_bbox for tid, track in self._tracks.items() if tid != exclude_track_id
        )

    def _backfill_track_start(self, track: Track, birth_frame: int) -> None:
        """
        Walk backward from `birth_frame` through the raw-detection buffer,
        greedily matching the nearest candidate centroid to a backward-
        extrapolated predicted position each step. Stops as soon as a frame
        has no usable candidate.
        """
        buffer_by_frame = dict(self._raw_buffer)

        predicted = track.history_frame_to_centroid[birth_frame]
        velocity = (0.0, 0.0)
        recovered: list[tuple[int, tuple[float, float], tuple[float, float, float, float]]] = []

        frame = birth_frame - 1
        steps = 0
        while frame >= 0 and steps < self._max_backfill_frames:
            if self._frame_already_claimed(frame, track.id):
                break

            candidates = buffer_by_frame.get(frame)
            if not candidates:
                break

            pred_x = predicted[0] - velocity[0]
            pred_y = predicted[1] - velocity[1]

            best = None
            best_dist = None
            for d in candidates:
                if d[4] < self._backfill_min_confidence:
                    continue
                cx, cy = (d[0] + d[2]) / 2.0, (d[1] + d[3]) / 2.0
                dist = math.hypot(cx - pred_x, cy - pred_y)
                if best_dist is None or dist < best_dist:
                    best_dist, best = dist, d

            if best is None or (best_dist is not None and best_dist > self._backfill_gate_px):
                break

            cx, cy = (best[0] + best[2]) / 2.0, (best[1] + best[3]) / 2.0
            bbox = (best[0], best[1], best[2], best[3])
            centroid = (cx, cy)
            recovered.append((frame, centroid, bbox))

            velocity = (predicted[0] - cx, predicted[1] - cy)
            predicted = centroid
            frame -= 1
            steps += 1

        for frame_idx, centroid, bbox in reversed(recovered):
            track.add(frame_idx, centroid, bbox)

    def finalize(self) -> list[Track]:
        """
        Ends tracking and returns the final list of `Track` objects.
        """
        if self._enable_duplicate_suppression:
            self._prune_duplicate_tracks()
        return list(self._tracks.values())

    def number_of_ids(self) -> int:
        """Number of distinct tracks currently held (reflects pruning if `finalize()` has run)."""
        return len(self._tracks)

    def overlap_episodes(self) -> list[OverlapEpisode]:
        """
        Groups per-frame overlaps into contiguous episodes per ID-pair.

        A run of frames counts as one episode as long as consecutive
        overlapping frame indices are adjacent (difference of 1).
        NOTE: a gap ends the episode and a new overlap starts a new one.
        Use this to find periods where two tracked bats' boxes overlapped,
        e.g. to flag frames where identity could plausibly have been
        swapped between them.
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
        """Builds an `OverlapEpisode` from one accumulated run of overlapping frames, or `None` if the run is empty."""
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
        """Total number of overlap episodes across all track pairs. Shorthand for `len(overlap_episodes())`."""
        return len(self.overlap_episodes())
