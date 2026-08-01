import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from battid.core.report_generation import PipelineStats, write_reports
from battid.core.sequencer.sequencer import Sequencer
from battid.core.utils import format_duration, get_detection_workers
from battid.models.report import Report


class ConcurrentSequencer(Sequencer):
    def __init__(self, output: Path, roi: dict[str, int] | None = None) -> None:
        super().__init__(output, roi)
        self._logger: logging.Logger = logging.getLogger(__name__)

    def _run_extraction_phase(self, videos: list[Path], max_workers: int) -> list[dict[str, Any]]:
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._create_video_frames, video): video for video in videos}

            with tqdm(total=len(futures), desc="Extracting frames", unit="video") as pbar:
                for future in as_completed(futures):
                    video = futures[future]
                    try:
                        frames_path, img_w, img_h, fps, frame_report = future.result()
                        results.append(
                            {
                                "video": video,
                                "frames_path": frames_path,
                                "img_w": img_w,
                                "img_h": img_h,
                                "fps": fps,
                                "reports": [frame_report],
                            }
                        )
                        pbar.set_postfix_str(f"[OK] - {video.name}")
                    except Exception as e:
                        self._logger.error(f"Frame extraction failed for {video}: {e}")
                        pbar.set_postfix_str(f"[N.OK] - {video.name}")
                    finally:
                        pbar.update(1)
        return results

    def _run_detection_phase(self, frame_data: list[dict[str, Any]], max_workers: int) -> list[dict[str, Any]]:
        results = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._run_detection_step, entry["frames_path"], entry["video"]): entry
                for entry in frame_data
            }

            with tqdm(total=len(futures), desc="Running detection", unit="video") as pbar:
                for future in as_completed(futures):
                    entry = futures[future]
                    try:
                        detections, detection_report = future.result()
                        entry["detections"] = detections
                        entry["reports"].append(detection_report)
                        results.append(entry)
                        pbar.set_postfix_str(f"[OK] - {entry['video'].name}")
                    except Exception as e:
                        self._logger.error(f"Detection failed for {entry['video']}: {e}")
                        pbar.set_postfix_str(f"[N.OK] - {entry['video'].name}")
                    finally:
                        pbar.update(1)
        return results

    def _run_tracking_phase(
        self,
        detection_data: list[dict[str, Any]],
        crop: bool,
        max_workers: int,
        generate_report: bool,
        videos: list[Path],
        stats: PipelineStats,
    ) -> None:

        video_reports: dict[Path, list[Report]] = {}

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._track_and_export,
                    entry["video"],
                    entry["frames_path"],
                    entry["detections"],
                    entry["img_w"],
                    entry["img_h"],
                    entry["fps"],
                    crop,
                ): entry
                for entry in detection_data
            }

            with tqdm(total=len(futures), desc="Tracking & exporting", unit="video") as pbar:
                for future in as_completed(futures):
                    entry = futures[future]
                    video = entry["video"]
                    try:
                        tracking_report = future.result()
                        reports = entry["reports"] + [tracking_report]

                        if generate_report:
                            if self._roi is None:
                                raise ValueError(
                                    "ROI has not been set. Please set the ROI before generating sequences or "
                                    "use override_roi=True to select a new ROI."
                                )

                            write_reports(reports, video, self._output, self._roi)

                        video_reports[video] = reports
                        pbar.set_postfix_str(f"[OK] - {video.name}")
                    except Exception as e:
                        self._logger.error(f"Tracking failed for {video}: {e}")
                        pbar.set_postfix_str(f"[N.OK] - {video.name}")
                    finally:
                        pbar.update(1)

        # Rebuild in original input order, and only for videos that actually succeeded
        for video in videos:
            reports = video_reports.get(video)
            if reports is None:
                continue  # failed earlier in the pipeline(already logged)

            stats.record(video, reports)

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

        cpu_cores = os.cpu_count()
        if cpu_cores is None:
            cpu_cores = 1

        cpu_workers = cpu_cores - 1 if cpu_cores > 1 else 1
        detection_workers = get_detection_workers(cpu_workers)

        self._logger.info(f"CPU workers: {cpu_workers}, detection workers: {detection_workers}")
        self._logger.info(f"Number of videos: {len(videos)}")

        stats = PipelineStats()
        start_time = time.time()

        frame_data = self._run_extraction_phase(videos, cpu_workers)
        detection_data = self._run_detection_phase(frame_data, detection_workers)
        self._run_tracking_phase(detection_data, crop, cpu_workers, generate_report, videos, stats)
        stats.write_summary(self._output)

        duration = time.time() - start_time
        self._logger.info(f"Process duration: {format_duration(duration)}")
