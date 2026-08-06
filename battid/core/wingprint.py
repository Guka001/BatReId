import json
import logging
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from battid.core.frame_generator import FrameGenerator
from battid.core.illumination import compute_illumination_metrics, is_wing_visible
from battid.core.utils import pad_and_clamp_bbox
from battid.models.detection_model_output import DCOutput
from battid.models.wingprint import WingVisibilityRecord

DETECTION_CONF_THRESHOLD: float = 0.3
CROP_PADDING_RATIO: float = 0.15
BORDER_TOUCH_MARGIN: int = 10


class WingPrint:
    """
    Classifies whether a bat's wing is clearly visible inside a detection crop.
    """

    def __init__(self, output: Path) -> None:
        self._logger: logging.Logger = logging.getLogger(__name__)

        self._output: Path = output
        self._output.mkdir(exist_ok=True)

        self._illumination_output: Path = self._output.joinpath("illumination")
        self._illumination_output.mkdir(exist_ok=True)

        self._wings_visible_output: Path = self._output.joinpath("wings_visible")
        self._wings_visible_output.mkdir(exist_ok=True)

        self._confidence_threshold: float = DETECTION_CONF_THRESHOLD
        self._crop_padding_ratio: float = CROP_PADDING_RATIO
        self._border_touch_margin: int = BORDER_TOUCH_MARGIN

    def __bbox_touches_border(self, x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> bool:
        margin = self._border_touch_margin
        return x1 <= margin or y1 <= margin or x2 >= img_w - margin or y2 >= img_h - margin

    @staticmethod
    def _build_map(paths: list[Path], expect_dir: bool) -> dict[str, Path]:
        stem_map: dict[str, Path] = {}

        for path in paths:
            if expect_dir and not path.is_dir():
                raise NotADirectoryError(f"{path} is not a valid directory")

            if not expect_dir and not path.is_file():
                raise FileNotFoundError(f"Could not find file {path}")
            stem_map[path.stem] = path

        return stem_map

    def _iter_qualifying_bboxes(
        self, detection_path: Path, frames_dir: Path
    ) -> Iterator[tuple[str, int, int, np.ndarray, int, int, int, int, float]]:
        """Yield `(search_key, frame_idx, detection_index, frame_bgr, x1, y1, x2, y2, confidence)`
        for every animal detection that survives confidence filtering, padding, and the
        border-touch check.
        """
        search_key = detection_path.stem
        detections: DCOutput = DCOutput.model_validate_json(detection_path.read_text(encoding="utf-8"))

        img_w, img_h = FrameGenerator.get_frame_size(frames_dir)
        animal_category_ids = {cat_id for cat_id, name in detections.detection_categories.items() if name == "animal"}

        for image_result in detections.images:
            frame_idx = FrameGenerator.parse_frame_index(image_result.file)
            frame_path = frames_dir.joinpath(image_result.file)
            frame: np.ndarray | None = None

            for detection_index, det in enumerate(image_result.detections):
                if det.category not in animal_category_ids or det.conf < self._confidence_threshold:
                    continue

                x, y, w, h = det.bbox
                bbox = (x * img_w, y * img_h, (x + w) * img_w, (y + h) * img_h)
                x1, y1, x2, y2 = pad_and_clamp_bbox(self._crop_padding_ratio, bbox, img_w, img_h)

                if x2 <= x1 or y2 <= y1:
                    self._logger.warning(f"Degenerate crop for {search_key} frame {frame_idx}, skipping")
                    continue
                if self.__bbox_touches_border(x1, y1, x2, y2, img_w, img_h):
                    self._logger.debug(
                        f"{search_key} frame {frame_idx} det {detection_index}: bbox touches border, "
                        f"likely partial animal body, skipping"
                    )
                    continue

                if frame is None:
                    frame = cv2.imread(str(frame_path))
                    if frame is None:
                        self._logger.warning(f"Could not read frame {frame_path}, skipping its detections")
                        break

                yield search_key, frame_idx, detection_index, frame, x1, y1, x2, y2, det.conf

    def _illumination_records_for_detection_file(
        self, detection_path: Path, frames_dir: Path
    ) -> list[WingVisibilityRecord]:
        search_key = detection_path.stem
        crop_dir = self._wings_visible_output.joinpath(search_key)

        records: list[WingVisibilityRecord] = []

        for _, frame_idx, detection_index, frame, x1, y1, x2, y2, conf in self._iter_qualifying_bboxes(
            detection_path, frames_dir
        ):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            metrics = compute_illumination_metrics(gray, x1, y1, x2, y2)
            visible = is_wing_visible(metrics)

            crop_path_str: str | None = None
            if visible:
                crop_dir.mkdir(exist_ok=True)
                crop_path = crop_dir.joinpath(f"frame{frame_idx:04d}_det{detection_index}.jpg")
                cv2.imwrite(str(crop_path), frame[y1:y2, x1:x2])
                crop_path_str = str(crop_path)

            records.append(
                WingVisibilityRecord(
                    search_key=search_key,
                    frame_idx=frame_idx,
                    detection_index=detection_index,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    metrics=metrics,
                    wing_visible=visible,
                    crop_path=crop_path_str,
                )
            )

        return records

    def detect_clearly_visible_wings(
        self, detections_paths: list[Path], frames: list[Path]
    ) -> list[WingVisibilityRecord]:
        """
        Illumination-based heuristic for wing visibility

        Args:
            detections_paths (list[Path]): List of detection outputs files corresponding to the videos.
            frames (list[Path]): List of paths to the folders containing the generated frames for each video.
                These must be the frames the detections were generated from.

        Returns:
            (list[WingVisibilityRecord]): One record per surviving detection, across all
            inputs.
        """
        frames_map = self._build_map(frames, expect_dir=True)

        all_records: list[WingVisibilityRecord] = []

        for detection_path in detections_paths:
            if not detection_path.is_file() or detection_path.suffix != ".json":
                continue

            search_key = detection_path.stem
            frames_dir = frames_map.get(search_key)
            if frames_dir is None:
                raise ValueError(f"Could not find directory {search_key} in the provided list of frames")

            self._logger.info(f"Scanning {detection_path} for clearly visible wings")
            records = self._illumination_records_for_detection_file(detection_path, frames_dir)
            n_visible = sum(1 for r in records if r.wing_visible)
            self._logger.info(f"{search_key}: {n_visible}/{len(records)} detection(s) with a clearly visible wing")

            self._save_illumination_records(search_key, records)
            all_records.extend(records)

        return all_records

    def _save_illumination_records(self, search_key: str, records: list[WingVisibilityRecord]) -> None:
        records_path = self._illumination_output.joinpath(f"{search_key}.json")
        records_path.write_text(
            json.dumps([r.model_dump() for r in records], indent=2),
            encoding="utf-8",
        )
