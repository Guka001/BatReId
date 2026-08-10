import logging
import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image

from battid.core.detection_steps import create_video_frames, run_detection
from battid.core.frame_generator import FrameGenerator
from battid.core.object_detection.mgd5 import MGD5
from battid.core.tracking import BatTracker
from battid.core.utils import bbox_touches_border, pad_and_clamp_bbox, select_roi_from_video, track_crosses_roi
from battid.models.detection_model_output import DCOutput
from battid.models.report import DetectionReport, FrameGenerationReport, TrackingReport
from battid.models.sequence_output import SequenceRecord
from battid.models.tracking import Track, TrackingResult

MIN_TRACK_LENGTH: int = 20
LOST_TRACK_BUFFER: int = 10
MIN_MATCHING_THRESHOLD: float = 0.25
CROP_PADDING_RATIO: float = 0.15
DETECTION_CONF_THRESHOLD: float = 0.3
BORDER_TOUCH_MARGIN: int = 10


class Sequencer(ABC):
    """
    Utility class responsible for converting a single video file into one or
    more frame sequences.
    """

    def __init__(self, output: Path, roi: dict[str, int] | None = None) -> None:
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._confidence_threshold: float = DETECTION_CONF_THRESHOLD
        self._min_track_len: int = MIN_TRACK_LENGTH
        self._lost_track_buffer: int = LOST_TRACK_BUFFER
        self._minimum_matching_threshold: float = MIN_MATCHING_THRESHOLD
        self._crop_padding_ratio: float = CROP_PADDING_RATIO
        self._border_touch_margin: int = BORDER_TOUCH_MARGIN

        self._output: Path = output
        self._output.mkdir(exist_ok=True)
        self._frames_output: Path = self._output.joinpath("frames")
        self._frames_output.mkdir(exist_ok=True)
        self._sequences_output: Path = self._output.joinpath("sequences")
        self._sequences_output.mkdir(exist_ok=True)

        self._detector: MGD5 = MGD5()
        self._roi: dict[str, int] | None = roi

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

    def _create_video_frames(self, video: Path) -> tuple[Path, int, int, int, FrameGenerationReport]:
        """Create frame sequences from a video file.

        Args:
            video (Path): Path to the source video file to be processed.

        Returns:
            (Tuple[Path, int, int, int, FrameGenerationReport]):
             The path to the folder containing the generated frames,
             the width and height of the frames, the video fps and process report.
        """
        return create_video_frames(video, self._frames_output)

    def _run_detection_step(self, frames_path: Path, video: Path) -> tuple[DCOutput, DetectionReport]:
        """
        Runs the object detection step on each frame found in the underlying path of the video

        Args:
            frames_path (Path): The path containing all frames.
            video (Path): The path to the video corresponding to the frames.

        Returns:
            (tuple[DCOutput, DetectionReport]). A tuple of detection results and report of the process.
        """
        detections, _detection_path, report = run_detection(self._detector, frames_path, video, self._output)
        return detections, report

    def _run_tracking_step(self, detections: DCOutput, img_w: int, img_h: int, fps: int) -> TrackingResult:
        self._logger.info("Tracking detections")
        animal_category_ids = {cat_id for cat_id, name in detections.detection_categories.items() if name == "animal"}
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
            track_activation_threshold=self._confidence_threshold,
            lost_track_buffer=self._lost_track_buffer,
            minimum_matching_threshold=self._minimum_matching_threshold,
        )

        start_time = time.time()

        for frame_idx in sorted(per_frame_detections):
            tracker.step(per_frame_detections[frame_idx], frame_idx)

        duration = time.time() - start_time
        self._logger.info("Tracking completed")

        return TrackingResult(
            tracks=tracker.finalize(),
            num_ids=tracker.number_of_ids(),
            overlap_episodes=tracker.overlap_episodes(),
            duration=duration,
        )

    def _track_and_export(
        self, video: Path, frames_path: Path, detections: DCOutput, img_w: int, img_h: int, fps: int, crop: bool
    ) -> TrackingReport:

        tracking_result = self._run_tracking_step(detections, img_w, img_h, fps)
        overlaps_count = len(tracking_result.overlap_episodes)

        records = []
        flight_coordinates: dict[int, list[tuple[float, float, float, float]]] = {}
        discarded_flight_coordinates: dict[int, list[tuple[float, float, float, float]]] = {}

        for track in tracking_result.tracks:
            frames_count = len(track.frames)
            self._logger.info(f"Parsing Track {track.id}")

            if frames_count < self._min_track_len:
                discarded_flight_coordinates[track.id] = list(track.history_frame_to_bbox.values())
                self._logger.warning(
                    f"Discarding Track {track.id} due to insufficient frame size."
                    f"Expected: >={self._min_track_len} frames, Actual: {frames_count} frames"
                )
                continue

            if self._roi is None:
                raise ValueError(
                    "ROI has not been set. Please set the ROI before generating sequences or "
                    "use override_roi=True to select a new ROI."
                )

            if track_crosses_roi(track, self._roi):
                sequence = self._export_sequence(video, frames_path, track, img_w, img_h, crop)
                if not sequence.empty:
                    records.append(sequence)
                    flight_coordinates[track.id] = list(track.history_frame_to_bbox.values())
                else:
                    self._logger.warning(
                        f"Discarding Track {track.id}. Track bbox coordinates are not fully enclosed within the video"
                    )
            else:
                self._logger.warning(f"Discarding Track {track.id}. This Track doesn't pass through the specified ROI")
                discarded_flight_coordinates[track.id] = list(track.history_frame_to_bbox.values())

        self._logger.info(f"{len(records)} accepted sequence(s) for {video.name}")

        return TrackingReport(
            description=f"Tracking detections for {video.name}",
            duration=tracking_result.duration,
            min_track_length=self._min_track_len,
            minimum_matching_threshold=self._minimum_matching_threshold,
            number_of_survived_lost_tracks=self._lost_track_buffer,
            overlaps=overlaps_count > 0,
            number_of_overlaps=overlaps_count,
            raw_number_of_unique_tracks=tracking_result.num_ids,
            number_of_unique_tracks_kept=len(records),
            flights=flight_coordinates,
            discarded_flights=discarded_flight_coordinates,
        )

    def _export_sequence(
        self, video: Path, frames_path: Path, track: Track, img_w: int, img_h: int, crop: bool
    ) -> SequenceRecord:
        """Copy the contiguous frame range of an accepted track into its own
        sequence folder.

        Args:
            video (Path): Path to the source video file.
            frames_path (Path): Path to the folder containing the generated frames.
            track (Track): The track object representing the accepted sequence.
            img_w (int): Width of the video frames, in pixels.
            img_h (int): Height of the video frames, in pixels.
            crop (bool): If True, export each frame cropped (and padded) to the
                track's bbox for that frame, instead of the full frame.

        Returns:
            (SequenceRecord): A record containing metadata about the exported sequence.
        """
        self._logger.info("Exporting sequences")

        frame_start, frame_end = min(track.frames), max(track.frames)
        sequence_dir = self._sequences_output.joinpath(f"{video.stem}_track{track.id}")
        sequence_dir.mkdir(exist_ok=True)

        exported_frames: list[int] = []

        for frame_idx in range(frame_start, frame_end + 1):
            bbox = track.history_frame_to_bbox.get(frame_idx)
            if bbox is None:
                continue

            cx1, cy1, cx2, cy2 = None, None, None, None
            if crop:
                cx1, cy1, cx2, cy2 = pad_and_clamp_bbox(self._crop_padding_ratio, bbox, img_w, img_h)

                if cx2 <= cx1 or cy2 <= cy1:
                    self._logger.warning(f"Degenerate crop box for track {track.id} frame {frame_idx}, skipping")
                    continue

                if bbox_touches_border(self._border_touch_margin, cx1, cy1, cx2, cy2, img_w, img_h):
                    self._logger.debug(
                        f"Track {track.id} frame {frame_idx}: padded bbox touches frame border, "
                        f"likely partial animal body, skipping"
                    )
                    continue
            else:
                x1, y1, x2, y2 = bbox
                if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
                    continue  # bbox clipped by frame edge, skip

            frame_file = frames_path.joinpath(f"{video.name}_frame_{frame_idx:04d}.jpg")
            if not frame_file.exists():
                self._logger.warning(f"Expected frame file not found: {frame_file}")
                continue

            dest_file = sequence_dir.joinpath(frame_file.name)

            if crop:
                with Image.open(frame_file) as img:
                    if cx1 is None or cy1 is None or cx2 is None or cy2 is None:
                        raise ValueError("Invalid bbox coordinates for export")

                    img.crop((cx1, cy1, cx2, cy2)).save(dest_file)
            else:
                shutil.copy(frame_file, dest_file)

            exported_frames.append(frame_idx)

        if not exported_frames:
            self._logger.warning(f"Track {track.id} for {video.name} had no in-bounds frames to export")
        else:
            self._logger.info("Export completed")

        return SequenceRecord(
            video=str(video),
            track_id=track.id,
            frame_start=frame_start,
            frame_end=frame_end,
            output_dir=str(sequence_dir),
            empty=len(exported_frames) == 0,
        )

    @staticmethod
    def compute_roi(video: Path) -> dict[str, int]:
        """Prompt the user to select a region of interest (ROI) from a video.

        Args:
            video (Path): Path to the source video file from which to select the ROI.

        Returns:
            (dict[str, int]): A dictionary containing the coordinates of the selected ROI.
        """

        return select_roi_from_video(video)

    @abstractmethod
    def generate_sequences(
        self,
        videos: list[Path],
        override_roi: bool = False,
        crop: bool = False,
        generate_report: bool = False,
    ) -> None:
        """Generate consecutive frame sequences from the given videos

        Args:
            videos (Path): Path to the source video files to be processed.
            override_roi (bool): If True, class ROI will be overwritten with the new determined value.
            crop (bool): If True, each frame in the sequence will be cropped to only include bat.
            generate_report (bool): If True, a pipeline report will be generated for each sequence.
        """

        raise NotImplementedError()

    @abstractmethod
    def generate_sequences_from_detections(
        self,
        videos: list[Path],
        detection_paths: list[Path],
        frames_paths: list[Path],
        override_roi: bool = False,
        crop: bool = False,
        generate_report: bool = False,
    ) -> None:
        """Generate consecutive frame sequences from the given videos and detections

        Args:
            videos (Path): Path to the source video files to be processed.
            detection_paths (list[Path]): List of detection outputs files corresponding to the videos.
            frames_paths (list[Path]): List of paths to the folders containing the generated frames for each video.
                These must be the frames the detections were generated from.
            override_roi (bool): If True, class ROI will be overwritten with the new determined value.
            crop (bool): If True, each frame in the sequence will be cropped to only include bat.
            generate_report (bool): If True, a pipeline report will be generated for each sequence.
        """

        raise NotImplementedError()
