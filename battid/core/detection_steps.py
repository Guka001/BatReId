import logging
from pathlib import Path

from battid.core.frame_generator import FrameGenerator
from battid.core.object_detection.base_detection_model import BaseDetectionModel
from battid.models.detection_model_output import DCOutput
from battid.models.report import DetectionReport, FrameGenerationReport


_logger: logging.Logger = logging.getLogger(__name__)


def create_video_frames(video: Path, frames_output: Path) -> tuple[Path, int, int, int, FrameGenerationReport]:
    """Extract frames for a single video.

    Args:
        video (Path): Path to the source video file to be processed.
        frames_output (Path): Root directory frames are written under. A
            subfolder named after the video is created inside it.

    Returns:
        (tuple[Path, int, int, int, FrameGenerationReport]):
         The path to the folder containing the generated frames,
         the width and height of the frames, the video fps and process report.
    """
    _logger.info(f"Generating video frames for file {video}")

    destination = frames_output.joinpath(video.stem)
    duration, frames = FrameGenerator.deconstruct_video_into_frames(video, destination)

    report = FrameGenerationReport(
        description=f"Generating frames for video: {video.name}",
        duration=duration,
        number_of_frames_generated=frames,
    )

    img_w, img_h = FrameGenerator.get_frame_size(destination)
    fps = FrameGenerator.get_video_fps(video)

    _logger.info("Generation completed")

    return destination, img_w, img_h, fps, report


def run_detection(
    detector: BaseDetectionModel, frames_path: Path, video: Path, output: Path
) -> tuple[DCOutput, Path, DetectionReport]:
    """Run object detection on the extracted frames of a single video.

        detector (BaseDetectionModel): Detection model to run on the frames.
        frames_path (Path): Path containing all frames for the video.
        video (Path): Path to the video corresponding to the frames.
        output (Path): Directory the detection JSON result is saved under.

    Returns:
        (tuple[DCOutput, Path, DetectionReport]). The detection results, the
        path the results were saved to, and a report of the process.
    """
    _logger.info(f"Running detection on frames for {video}")
    detections, duration = detector.run_detection(frames_path)

    detection_path = output.joinpath(f"{video.stem}.json")
    detections.save(detection_path)

    report = DetectionReport(
        description=f"Running Megadetector detections on video: {video.name}", duration=duration
    )

    return detections, detection_path, report
