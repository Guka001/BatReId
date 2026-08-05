import logging
import time
from pathlib import Path

from battid.models.detection_model_output import DetectionGenerationRecord
from battid.core.detection_generator.detection_generator import DetectionGenerator
from battid.core.detection_steps import create_video_frames, run_detection
from battid.core.object_detection.base_detection_model import BaseDetectionModel
from battid.core.utils import format_duration


class SerialDetectionGenerator(DetectionGenerator):
    def __init__(self, output: Path, detector: BaseDetectionModel | None = None) -> None:
        super().__init__(output, detector)
        self._logger: logging.Logger = logging.getLogger(__name__)

    def generate(self, videos: list[Path]) -> list[DetectionGenerationRecord]:
        self._logger.info(f"Number of videos: {len(videos)}")

        records: list[DetectionGenerationRecord] = []
        start_time = time.time()

        for video in videos:
            self._logger.info(f"Parsing video {video}")

            try:
                frames_path, img_w, img_h, fps, frame_report = create_video_frames(video, self._frames_output)
                detections, detection_path, detection_report = run_detection(
                    self._detector, frames_path, video, self._output
                )

                records.append(
                    DetectionGenerationRecord(
                        video=video,
                        frames_path=frames_path,
                        detection_path=detection_path,
                        detections=detections,
                        img_w=img_w,
                        img_h=img_h,
                        fps=fps,
                        reports=[frame_report, detection_report],
                    )
                )
            except Exception as e:
                self._logger.error(f"An error occurred when parsing video {video}: {e}")

        duration = time.time() - start_time
        self._logger.info(f"Process duration: {format_duration(duration)}")

        return records
