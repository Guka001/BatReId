import time
import logging
from pathlib import Path

import cv2
import numpy as np

from battid.models.wingprint import WingPrintMetrics
from battid.models.detection_model_output import DCOutput
from battid.core.frame_generator import FrameGenerator
from battid.core.sequencer.sequencer import CROP_PADDING_RATIO
from battid.core.utils import pad_and_clamp_bbox


class WingPrint:
    def __init__(self, output: Path) -> None:
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._min_laplacian_var: float = 120.0
        self._min_edge_density: float = 0.045
        self._min_bat_pixels: int = 50
        self._min_bat_area_frac: float = 0.03
        self._max_saturation_frac: float = 0.15
        self._min_detection_conf: float = 0.2
        self._crop_padding_ratio: float = CROP_PADDING_RATIO

        self._output: Path = output.joinpath("wing_prints")
        self._output.mkdir(exist_ok=True, parents=True)

    @staticmethod
    def megadetector_bbox_to_pixels(
            bbox_norm: list[float], img_w: int, img_h: int
    ) -> tuple[float, float, float, float]:
        """
        Converts a MegaDetector bbox [x, y, w, h] (normalized 0-1, top-left
        origin) into pixel-space (x1, y1, x2, y2).
        """
        x, y, w, h = bbox_norm
        x1 = x * img_w
        y1 = y * img_h
        x2 = (x + w) * img_w
        y2 = (y + h) * img_h
        return x1, y1, x2, y2

    def score_wing_print_visibility(self, crop: np.ndarray) -> WingPrintMetrics:
        """
        Scores whether a bat's wing venation is clearly visible in a padded
        crop against a black background.

        Args:
            crop: BGR or grayscale crop as returned from `img[y1:y2, x1:x2]`
                using the padded/clamped bbox.

        Returns:
            WingPrintMetrics for the crop
        """
        if crop.ndim == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop

        if gray.size == 0:
            return WingPrintMetrics(
                mean_brightness=0.0,
                p90_brightness=0.0,
                laplacian_var=0.0,
                edge_density=0.0,
                bat_area_frac=0.0,
                saturation_frac=0.0,
                is_clear=False,
            )

        # Background is near-black IR/flash footage
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask_bool = mask > 0

        bat_area_frac = float(mask_bool.sum()) / gray.size

        # Below this, the mask is a sliver of the crop (e.g. a folded/tucked
        # bat caught mostly out of frame) rather than a body with wings
        # spread wide enough to show venation.
        if mask_bool.sum() < self._min_bat_pixels or bat_area_frac < self._min_bat_area_frac:
            return WingPrintMetrics(
                mean_brightness=0.0,
                p90_brightness=0.0,
                laplacian_var=0.0,
                edge_density=0.0,
                bat_area_frac=bat_area_frac,
                saturation_frac=0.0,
                is_clear=False,
            )

        bat_pixels = gray[mask_bool]
        mean_brightness = float(bat_pixels.mean())
        p90_brightness = float(np.percentile(bat_pixels, 90))
        # Fraction of the bat blown out by the flash; past a point this
        # washes out the venation instead of revealing it.
        saturation_frac = float((bat_pixels >= 250).sum()) / mask_bool.sum()

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = float(laplacian[mask_bool].var())

        edges = cv2.Canny(gray, 30, 90)
        edge_density = float((edges[mask_bool] > 0).sum()) / mask_bool.sum()

        is_clear = (
            laplacian_var >= self._min_laplacian_var
            and edge_density >= self._min_edge_density
            and saturation_frac <= self._max_saturation_frac
        )

        return WingPrintMetrics(
            mean_brightness=mean_brightness,
            p90_brightness=p90_brightness,
            laplacian_var=laplacian_var,
            edge_density=edge_density,
            bat_area_frac=bat_area_frac,
            saturation_frac=saturation_frac,
            is_clear=is_clear,
        )

    def detect_wing_prints(self, detections_paths: list[Path], frames_paths: list[Path]) -> None:
        """
        Runs the wing-print detection process.

        For each detections file and its corresponding frames folder, loads the
        detection outputs, crops each detected bat and scores whether the wing print
        is clearly visible in the crop. Visibility is estimated by
        segmenting the bat from the background and computing sharpness and
        edge-density metrics within that region, classifying each detection as
        a clear or not-clear.

        Args:
            detections_paths (list[Path]): List of detection outputs files.
            frames_paths (list[Path]): List of paths to the folders containing the generated frames.
                These must be the frames the detections were generated from.
        """

        frames_map: dict[str, Path] = {}
        for frame in frames_paths:
            if not frame.is_dir():
                raise NotADirectoryError(f"{frame} is not a valid directory")
            frames_map[frame.stem] = frame

        for detection_path in detections_paths:
            if not detection_path.is_file() or detection_path.suffix != ".json":
                continue

            start = time.time()
            self._logger.info(f"Parsing {detection_path}")

            detections: DCOutput = DCOutput.model_validate_json(detection_path.read_text(encoding="utf-8"))

            search_key = detection_path.stem
            frames_dir = frames_map.get(search_key)
            if frames_dir is None:
                raise ValueError(f"Could not find directory {search_key} in the provided list of frames")

            img_w, img_h = FrameGenerator.get_frame_size(frames_dir)
            animal_category_ids = {
                cat_id for cat_id, name in detections.detection_categories.items() if name == "animal"
            }

            for image_result in detections.images:
                animal_detections = [
                    (idx, det)
                    for idx, det in enumerate(image_result.detections)
                    if det.category in animal_category_ids and det.conf >= self._min_detection_conf
                ]
                if not animal_detections:
                    continue

                frame_path = frames_dir / image_result.file
                img = cv2.imread(str(frame_path))
                if img is None:
                    self._logger.warning(f"Could not read frame {frame_path}")
                    continue

                for det_idx, det in animal_detections:
                    bbox_px = self.megadetector_bbox_to_pixels(det.bbox, img_w, img_h)
                    x1, y1, x2, y2 = pad_and_clamp_bbox(
                        self._crop_padding_ratio, bbox_px, img_w, img_h
                    )
                    crop = img[y1:y2, x1:x2]

                    metrics = self.score_wing_print_visibility(crop)
                    if not metrics.is_clear:
                        continue

                    crop_name = f"{Path(image_result.file).stem}_det{det_idx}.jpg"
                    cv2.imwrite(str(self._output / crop_name), crop)

            self._logger.info(f"Finished {detection_path} in {time.time() - start:.2f}s")
