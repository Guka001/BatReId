import logging
import shutil
from pathlib import Path

from .frame_generator import FrameGenerator
from .models.detection_model_output import DCOutput
from .models.sequence_output import SequenceRecord
from .object_detection.mgd5 import MGD5
from .tracking import SimpleSORT, Track
from .utils import select_roi_from_video, track_crosses_roi


class Sequencer:
    """
    Utility class responsible for converting a single video file into one or
    more frame sequences.
    """

    def __init__(self, output: Path) -> None:
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._confidence_threshold: float = 0.3
        self._min_track_len: int = 2

        self._output: Path = output
        self._output.mkdir(exist_ok=True)
        self._frames_output: Path = self._output.joinpath("frames")
        self._frames_output.mkdir(exist_ok=True)
        self._sequences_output: Path = self._output.joinpath("sequences")
        self._sequences_output.mkdir(exist_ok=True)

        self._detector: MGD5 = MGD5()
        self._roi: dict[str, int] | None = None

    def _create_video_frames(self, video: Path) -> tuple[Path, int, int]:
        """Create frame sequences from a video file.

        Args:
            video (Path): Path to the source video file to be processed.

        Returns:
            (Tuple[Path, int, int]): The path to the folder containing the generated
            frames, and the width and height of the frames.
        """

        self._logger.info(f"Generating video frames for file {video}")

        destination = self._frames_output.joinpath(video.stem)
        FrameGenerator.deconstruct_video_into_frames(video, destination)
        img_w, img_h = FrameGenerator.get_frame_size(destination)
        self._logger.info("Generation completed")

        return destination, img_w, img_h

    def _run_tracking(self, detections: DCOutput, img_w: int, img_h: int) -> list[Track]:
        """Filter detections by category and confidence, denormalize bboxes
        to pixel coordinates, and feed them frame-by-frame into a SORT tracker.
        """
        animal_category_ids = {cat_id for cat_id, name in detections.detection_categories.items() if name == "animal"}

        per_frame_detections: dict[int, list[tuple[float, float, float, float]]] = {}
        for image_result in detections.images:
            frame_idx = FrameGenerator.parse_frame_index(image_result.file)
            boxes = []
            for det in image_result.detections:
                if det.category not in animal_category_ids:
                    continue
                if det.conf < self._confidence_threshold:
                    continue
                x, y, w, h = det.bbox
                boxes.append((x * img_w, y * img_h, (x + w) * img_w, (y + h) * img_h))
            per_frame_detections[frame_idx] = boxes

        tracker = SimpleSORT()
        for frame_idx in sorted(per_frame_detections):
            tracker.step(per_frame_detections[frame_idx], frame_idx)

        return tracker.finalize()

    def _export_sequence(self, video: Path, frames_path: Path, track: Track) -> SequenceRecord:
        """Copy the contiguous frame range of an accepted track into its own
        sequence folder.

        Args:
            video (Path): Path to the source video file.
            frames_path (Path): Path to the folder containing the generated frames.
            track (Track): The track object representing the accepted sequence.

        Returns:
            (SequenceRecord): A record containing metadata about the exported sequence.
        """
        frame_start, frame_end = min(track.frames), max(track.frames)
        sequence_dir = self._sequences_output.joinpath(f"{video.stem}_track{track.id}")
        sequence_dir.mkdir(exist_ok=True)

        for frame_idx in range(frame_start, frame_end + 1):
            frame_file = frames_path.joinpath(f"{video.name}_frame_{frame_idx:04d}.jpg")
            if frame_file.exists():
                shutil.copy(frame_file, sequence_dir.joinpath(frame_file.name))
            else:
                self._logger.warning(f"Expected frame file not found: {frame_file}")

        return SequenceRecord(
            video=str(video),
            track_id=track.id,
            frame_start=frame_start,
            frame_end=frame_end,
            output_dir=str(sequence_dir),
        )

    def generate_sequence(self, video: Path) -> list[SequenceRecord]:
        """Run the full per-video pipeline: ROI setup (once), frame
        extraction, detection, tracking, ROI-based filtering, and export.

        Args:
            video (Path): Path to the source video file to be processed.

        Returns:
            (list[SequenceRecord]): One record per accepted sequence.
        """
        if self._roi is None:
            self._roi = select_roi_from_video(video)

        frames_path, img_w, img_h = self._create_video_frames(video)

        self._logger.info(f"Running detection on frames for {video}")

        detections = self._detector.run_detection(frames_path)
        detections.save(self._output.joinpath(f"{video.stem}_detections.json"))

        tracks = self._run_tracking(detections, img_w, img_h)

        records = []
        for track in tracks:
            if len(track.frames) < self._min_track_len:
                continue
            if track_crosses_roi(track, self._roi):
                records.append(self._export_sequence(video, frames_path, track))

        self._logger.info(f"{len(records)} accepted sequence(s) for {video}")
        return records
