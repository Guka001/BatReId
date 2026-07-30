import logging
import time
from pathlib import Path

from battid.core.report_generation import PipelineStats, write_reports
from battid.core.sequencer.sequencer import Sequencer
from battid.core.utils import format_duration
from battid.models.report import Report


class SerialSequencer(Sequencer):
    def __init__(self, output: Path, roi: dict[str, int] | None = None) -> None:
        super().__init__(output, roi)
        self._logger: logging.Logger = logging.getLogger()

    def generate_sequences(
        self, videos: list[Path], override_roi: bool = False, crop: bool = False, generate_report: bool = False
    ) -> None:

        if override_roi:
            self._roi = self.compute_roi(videos[0])

        if self._roi is None:
            raise ValueError(
                "ROI has not been set. Please set the ROI before generating sequences or "
                "use override_roi=True to select a new ROI."
            )

        self._logger.info(f"Number of videos: {len(videos)}")

        stats = PipelineStats()
        start_time = time.time()

        for video in videos:
            self._logger.info(f"Parsing video {video}")
            reports: list[Report] = []

            try:
                frames_path, img_w, img_h, fps, frame_gen_report = self._create_video_frames(video)
                reports.append(frame_gen_report)

                detections, detection_report = self._run_detection_step(frames_path, video)
                reports.append(detection_report)

                tracking_report = self._track_and_export(video, frames_path, detections, img_w, img_h, fps, crop)
                reports.append(tracking_report)

                if generate_report:
                    if self._roi is None:
                        raise ValueError(
                            "ROI has not been set. Please set the ROI before generating sequences or "
                            "use override_roi=True to select a new ROI."
                        )
                    write_reports(reports, video, self._output, self._roi)

                stats.record(video, reports)

            except Exception as e:
                self._logger.error(f"An error occurred when parsing video {video}: {e}")

        stats.write_summary(self._output)

        duration = time.time() - start_time
        self._logger.info(f"Process duration: {format_duration(duration)}")
