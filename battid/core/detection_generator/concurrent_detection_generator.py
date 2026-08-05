import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from battid.models.detection_model_output import DetectionGenerationRecord
from battid.core.detection_generator.detection_generator import DetectionGenerator
from battid.core.detection_steps import create_video_frames, run_detection
from battid.core.object_detection.base_detection_model import BaseDetectionModel
from battid.core.utils import format_duration, get_detection_workers


class ConcurrentDetectionGenerator(DetectionGenerator):
    def __init__(self, output: Path, detector: BaseDetectionModel | None = None) -> None:
        super().__init__(output, detector)
        self._logger: logging.Logger = logging.getLogger(__name__)

    def _run_extraction_phase(self, videos: list[Path], max_workers: int) -> list[dict[str, Any]]:
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(create_video_frames, video, self._frames_output): video for video in videos
            }

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

    def _run_detection_phase(
        self, frame_data: list[dict[str, Any]], max_workers: int
    ) -> list[DetectionGenerationRecord]:
        records: list[DetectionGenerationRecord] = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    run_detection, self._detector, entry["frames_path"], entry["video"], self._output
                ): entry
                for entry in frame_data
            }

            with tqdm(total=len(futures), desc="Running detection", unit="video") as pbar:
                for future in as_completed(futures):
                    entry = futures[future]
                    try:
                        detections, detection_path, detection_report = future.result()
                        records.append(
                            DetectionGenerationRecord(
                                video=entry["video"],
                                frames_path=entry["frames_path"],
                                detection_path=detection_path,
                                detections=detections,
                                img_w=entry["img_w"],
                                img_h=entry["img_h"],
                                fps=entry["fps"],
                                reports=entry["reports"] + [detection_report],
                            )
                        )
                        pbar.set_postfix_str(f"[OK] - {entry['video'].name}")
                    except Exception as e:
                        self._logger.error(f"Detection failed for {entry['video']}: {e}")
                        pbar.set_postfix_str(f"[N.OK] - {entry['video'].name}")
                    finally:
                        pbar.update(1)
        return records

    def generate(self, videos: list[Path]) -> list[DetectionGenerationRecord]:
        cpu_cores = os.cpu_count()
        if cpu_cores is None:
            cpu_cores = 1

        cpu_workers = cpu_cores - 1 if cpu_cores > 1 else 1
        detection_workers = get_detection_workers(cpu_workers)

        self._logger.info(f"CPU workers: {cpu_workers}, detection workers: {detection_workers}")
        self._logger.info(f"Number of videos: {len(videos)}")

        start_time = time.time()

        frame_data = self._run_extraction_phase(videos, cpu_workers)
        records = self._run_detection_phase(frame_data, detection_workers)

        duration = time.time() - start_time
        self._logger.info(f"Process duration: {format_duration(duration)}")

        return records
