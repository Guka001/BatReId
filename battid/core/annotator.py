import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from battid.core.frame_generator import FrameGenerator
from battid.core.sequencer.sequencer import (
    CROP_PADDING_RATIO,
    DETECTION_CONF_THRESHOLD,
    LOST_TRACK_BUFFER,
    MIN_MATCHING_THRESHOLD,
    MIN_TRACK_LENGTH,
)
from battid.core.tracking import BatTracker
from battid.core.utils import format_duration, pad_and_clamp_bbox, track_crosses_roi
from battid.models.detection_model_output import DCOutput
from battid.models.tracking import Track, TrackingResult

matplotlib.use("Agg")
logger = logging.getLogger(__name__)

VALID_TRACK_COLOR = (0, 255, 0)  # green  - crosses the ROI
TOO_SHORT_TRACK_COLOR = (0, 0, 255)  # red    - discarded: below MIN_TRACK_LENGTH
NOT_IN_ROI_TRACK_COLOR = (0, 165, 255)  # orange - discarded: never crosses the ROI
ROI_COLOR = (255, 191, 0)  # deep sky blue

CATEGORY_INFO = {
    "valid": (VALID_TRACK_COLOR, "valid"),
    "too_short": (TOO_SHORT_TRACK_COLOR, "discarded_too_short"),
    "not_in_roi": (NOT_IN_ROI_TRACK_COLOR, "discarded_not_in_roi"),
}

BBox = tuple[float, float, float, float]
FrameAnnotations = dict[int, list[tuple[BBox, tuple[int, int, int], int]]]


class Annotator:
    def __init__(self, output: Path, roi: dict[str, int]) -> None:
        self._output: Path = output
        self._output.mkdir(exist_ok=True)
        self._videos_path: Path = output.joinpath("videos")
        self._videos_path.mkdir(exist_ok=True)
        self._plots_path: Path = output.joinpath("plots")
        self._plots_path.mkdir(exist_ok=True)

        self._roi: dict[str, int] = roi

    @staticmethod
    def _run_tracking(img_w: int, img_h: int, animal_category_ids: set[str], detections: DCOutput) -> TrackingResult:
        per_frame_detections: dict[int, list[tuple[float, float, float, float, float]]] = {}

        for image_result in detections.images:
            frame_idx = FrameGenerator.parse_frame_index(image_result.file)
            boxes = []
            for det in image_result.detections:
                if det.category not in animal_category_ids:
                    continue

                x, y, w, h = det.bbox

                # Denormalize the bounding box coordinates to pixel values
                boxes.append((x * img_w, y * img_h, (x + w) * img_w, (y + h) * img_h, det.conf))

            per_frame_detections[frame_idx] = boxes

        tracker = BatTracker(
            image_width=img_w,
            image_height=img_h,
            track_activation_threshold=DETECTION_CONF_THRESHOLD,
            lost_track_buffer=LOST_TRACK_BUFFER,
            minimum_matching_threshold=MIN_MATCHING_THRESHOLD,
        )

        for frame_idx in sorted(per_frame_detections):
            tracker.step(per_frame_detections[frame_idx], frame_idx)

        return TrackingResult(
            tracks=tracker.finalize(),
            num_ids=tracker.number_of_ids(),
            overlap_episodes=tracker.overlap_episodes(),
            duration=0.0,
        )

    @staticmethod
    def _categorize(track: Track, roi: dict[str, int]) -> str:
        if len(track.frames) < MIN_TRACK_LENGTH:
            return "too_short"
        if track_crosses_roi(track, roi):
            return "valid"
        return "not_in_roi"

    @staticmethod
    def _collect_track_bboxes(
        track: Track, img_w: int, img_h: int, frame_start: int | None = None, frame_end: int | None = None
    ) -> dict[int, BBox]:
        """Pad/clamp the bbox of `track` for every frame in [frame_start, frame_end]
        (defaults to the track's own first/last detected frame)."""
        bboxes: dict[int, BBox] = {}
        start = frame_start if frame_start is not None else min(track.frames)
        end = frame_end if frame_end is not None else max(track.frames)

        for frame_idx in range(start, end + 1):
            bbox = track.history_frame_to_bbox.get(frame_idx)
            if bbox is None:
                continue

            bboxes[frame_idx] = pad_and_clamp_bbox(CROP_PADDING_RATIO, bbox, img_w, img_h)

        return bboxes

    def _accumulate_track(
        self, track: Track, img_w: int, img_h: int, color: tuple[int, int, int], annotations: FrameAnnotations
    ) -> None:
        """Add every bbox of `track` into `annotations`, tagged with `color` and the track's id."""
        for frame_idx, bbox in self._collect_track_bboxes(track, img_w, img_h).items():
            annotations.setdefault(frame_idx, []).append((bbox, color, track.id))

    def _render_annotated_video(
        self,
        frames_dir: Path,
        output_video_path: Path,
        fps: int,
        annotations: FrameAnnotations,
        frame_range: tuple[int, int] | None = None,
    ) -> None:
        """
        Draw the ROI and every accumulated (bbox, color, track_id) onto its frame and
        write out a single video. Frames with no annotations still get the ROI drawn
        and are otherwise written through unchanged.

        If `frame_range` is given, only that (inclusive) window of frames is
        rendered.
        """
        frame_paths = sorted(
            (p for p in frames_dir.iterdir() if p.is_file()),
            key=lambda p: FrameGenerator.parse_frame_index(str(p)),
        )
        if not frame_paths:
            raise ValueError(f"No frames found in {frames_dir}")

        if frame_range is not None:
            start, end = frame_range
            frame_paths = [p for p in frame_paths if start <= FrameGenerator.parse_frame_index(str(p)) <= end]
            if not frame_paths:
                raise ValueError(f"No frames found in {frames_dir} within range {frame_range}")

        first_frame = cv2.imread(str(frame_paths[0]))
        if first_frame is None:
            raise ValueError(f"Could not read frame {frame_paths[0]}")
        height, width = first_frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise OSError(f"Could not open video writer for {output_video_path}")

        roi_x1, roi_y1 = self._roi["x1"], self._roi["y1"]
        roi_x2, roi_y2 = self._roi["x2"], self._roi["y2"]

        try:
            for frame_path in frame_paths:
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    logger.warning(f"Could not read frame {frame_path}, skipping")
                    continue

                cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), ROI_COLOR, 2)

                frame_idx = FrameGenerator.parse_frame_index(str(frame_path))
                for bbox, color, track_id in annotations.get(frame_idx, []):
                    x1, y1, x2, y2 = (int(v) for v in bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame, str(track_id), (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
                    )

                writer.write(frame)
        finally:
            writer.release()

    def _render_combined_video(
        self,
        tracks: list[Track],
        frames_dir: Path,
        fps: int,
        img_w: int,
        img_h: int,
        search_key: str,
    ) -> dict[str, int]:
        """Render one full-length video with every track drawn simultaneously,
        color-coded by category."""
        annotations: FrameAnnotations = {}
        counts = {"valid": 0, "too_short": 0, "not_in_roi": 0}

        for track in tracks:
            category = self._categorize(track, self._roi)
            counts[category] += 1
            color, _ = CATEGORY_INFO[category]
            self._accumulate_track(track, img_w, img_h, color, annotations)

        output_video_path = self._videos_path.joinpath(f"{search_key}_annotated.mp4")
        self._render_annotated_video(frames_dir, output_video_path, fps, annotations)

        return counts

    # tab20 stores 10 hue pairs back-to-back (dark0, light0, dark1, light1, ...).
    # Reordering to all darks first, then all lights, means track colors stay
    # maximally distinct in hue until more than 10 tracks are drawn in one
    # plot. only then do repeated/shade colors start appearing.
    _TAB20_DISTINCT_ORDER: tuple[int, ...] = tuple(range(0, 20, 2)) + tuple(range(1, 20, 2))

    @classmethod
    def _track_color(cls, index: int) -> tuple[float, float, float, float]:
        cmap = cast(ListedColormap, plt.get_cmap("tab20"))
        colors = cast(Sequence[tuple[float, float, float, float]], cmap.colors)
        color = colors[cls._TAB20_DISTINCT_ORDER[index % len(cls._TAB20_DISTINCT_ORDER)]]
        return cast(tuple[float, float, float, float], tuple(color))

    def _render_flight_path_plot(self, tracks: list[Track], img_w: int, img_h: int, search_key: str) -> None:

        fig, ax = plt.subplots(figsize=(10, 10 * img_h / img_w))
        ax.set_xlim(0, img_w)
        ax.set_ylim(0, img_h)
        ax.invert_yaxis()  # image-space: y grows downward, keep plot oriented like the video
        ax.set_aspect("equal")
        ax.set_title(f"Flight paths — {search_key}")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")

        roi_x1, roi_y1 = self._roi["x1"], self._roi["y1"]
        roi_x2, roi_y2 = self._roi["x2"], self._roi["y2"]
        ax.add_patch(
            Rectangle(
                (roi_x1, roi_y1),
                roi_x2 - roi_x1,
                roi_y2 - roi_y1,
                fill=False,
                edgecolor="deepskyblue",
                linewidth=1.5,
                linestyle="--",
            )
        )

        legend_handles = [Line2D([0], [0], color="deepskyblue", linestyle="--", linewidth=1.5, label="ROI")]

        for i, track in enumerate(sorted(tracks, key=lambda t: min(t.frames))):
            category = self._categorize(track, self._roi)
            color = self._track_color(i)

            ordered_frames = sorted(track.frames)
            xs = [track.history_frame_to_centroid[f][0] for f in ordered_frames]
            ys = [track.history_frame_to_centroid[f][1] for f in ordered_frames]

            ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.5, color=color, alpha=0.85)
            ax.scatter(xs[0], ys[0], color="green", s=110, zorder=5, edgecolors="black", linewidths=0.5)
            ax.scatter(xs[-1], ys[-1], color="red", s=110, zorder=5, edgecolors="black", linewidths=0.5)
            ax.annotate(
                str(track.id),
                (xs[0], ys[0]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=7,
                fontweight="bold",
                color="black",
                zorder=6,
            )
            ax.annotate(
                str(track.id),
                (xs[-1], ys[-1]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=7,
                fontweight="bold",
                color="black",
                zorder=6,
            )

            legend_handles.append(Line2D([0], [0], marker="o", color=color, label=f"Track {track.id} ({category})"))

        legend_handles.append(
            Line2D([0], [0], marker="o", linestyle="", color="green", markeredgecolor="black", label="Start")
        )
        legend_handles.append(
            Line2D([0], [0], marker="o", linestyle="", color="red", markeredgecolor="black", label="End")
        )

        ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
        fig.tight_layout()

        output_path = self._plots_path.joinpath(f"{search_key}_flightpaths.png")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def render_single_track_clips(
        self, videos_path: list[Path], detections_path: list[Path], frames: list[Path]
    ) -> None:

        frames_map, videos_map = self._build_maps(frames, videos_path)

        for detection_path in detections_path:
            if not detection_path.is_file() or detection_path.suffix != ".json":
                continue

            search_key = detection_path.stem
            frames_dir = frames_map[search_key]
            video_file = videos_map[search_key]

            detections = DCOutput.model_validate_json(detection_path.read_text(encoding="utf-8"))
            fps = FrameGenerator.get_video_fps(video_file)
            img_w, img_h = FrameGenerator.get_frame_size(frames_dir)
            animal_category_ids = {
                cat_id for cat_id, name in detections.detection_categories.items() if name == "animal"
            }

            tracking_result = self._run_tracking(img_w, img_h, animal_category_ids, detections)

            output_dir = self._videos_path.joinpath(search_key)
            output_dir.mkdir(exist_ok=True)

            for track in tracking_result.tracks:
                category = self._categorize(track, self._roi)
                color, suffix = CATEGORY_INFO[category]

                annotations: FrameAnnotations = {}
                self._accumulate_track(track, img_w, img_h, color, annotations)

                frame_start, frame_end = min(track.frames), max(track.frames)
                output_video_path = output_dir.joinpath(f"{search_key}_track_{track.id}_{suffix}.mp4")
                self._render_annotated_video(
                    frames_dir, output_video_path, fps, annotations, frame_range=(frame_start, frame_end)
                )

    @staticmethod
    def _build_maps(frames: list[Path], videos_path: list[Path]) -> tuple[dict[str, Path], dict[str, Path]]:
        frames_map: dict[str, Path] = {}
        videos_map: dict[str, Path] = {}

        for frame in frames:
            if not frame.is_dir():
                raise NotADirectoryError(f"{frame} is not a valid directory")
            frames_map[frame.stem] = frame

        for video in videos_path:
            if not video.is_file():
                raise FileNotFoundError(f"Could not find file {video}")
            videos_map[video.stem] = video

        return frames_map, videos_map

    def annotate(self, videos_path: list[Path], detections_path: list[Path], frames: list[Path]) -> None:
        frames_map, videos_map = self._build_maps(frames, videos_path)

        for detection_path in detections_path:
            if not detection_path.is_file() or detection_path.suffix != ".json":
                continue

            start = time.time()
            logger.info(f"Parsing {detection_path}")

            detections: DCOutput = DCOutput.model_validate_json(detection_path.read_text(encoding="utf-8"))

            search_key = detection_path.stem
            frames_dir = frames_map.get(search_key)
            if frames_dir is None:
                raise ValueError(f"Could not find directory {search_key} in the provided list of frames")

            video_file = videos_map.get(search_key)
            if video_file is None:
                raise ValueError(f"Could not find the video file for detection {detection_path}")

            fps = FrameGenerator.get_video_fps(video_file)
            img_w, img_h = FrameGenerator.get_frame_size(frames_dir)

            animal_category_ids = {
                cat_id for cat_id, name in detections.detection_categories.items() if name == "animal"
            }

            tracking_result = self._run_tracking(img_w, img_h, animal_category_ids, detections)

            counts = self._render_combined_video(tracking_result.tracks, frames_dir, fps, img_w, img_h, search_key)
            self._render_flight_path_plot(tracking_result.tracks, img_w, img_h, search_key)

            duration = time.time() - start
            logger.info(
                f"Finished parsing {detection_path}. "
                f"{tracking_result.num_ids} track(s): "
                f"{counts['valid']} valid, {counts['too_short']} too_short, {counts['not_in_roi']} not_in_roi. "
                f"{format_duration(duration)}"
            )
