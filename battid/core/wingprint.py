import logging
import time
from pathlib import Path
from typing import NamedTuple

import cv2
import joblib
import numpy as np
from skimage.feature import graycomatrix, graycoprops

from battid.core.frame_generator import FrameGenerator
from battid.core.sequencer.sequencer import BORDER_TOUCH_MARGIN, CROP_PADDING_RATIO
from battid.core.utils import pad_and_clamp_bbox
from battid.models.detection_model_output import DCOutput
from battid.models.wingprint import WingPrintMetrics

# Order the trained classifier expects its input vector
FEATURE_ORDER: list[str] = [
    "laplacian_var",
    "edge_density",
    "bat_area_frac",
    "saturation_frac",
    "mask_aspect",
    "mean_brightness",
    "p90_brightness",
    "homogeneity",
    "correlation",
    "contrast",
    "energy",
]

_MODEL_PATH = Path(__file__).parent / "wingprint_rf_model.joblib"

# Below this many mask pixels, per-pixel statistics (especially the GLCM
# texture features) are too noisy to be meaningful.
# treated as "not clear"
_MIN_MASK_PIXELS = 50


class RawWingPrintFeatures(NamedTuple):
    mean_brightness: float
    p90_brightness: float
    laplacian_var: float
    edge_density: float
    bat_area_frac: float
    saturation_frac: float
    mask_aspect: float
    homogeneity: float
    correlation: float
    contrast: float
    energy: float

    def as_vector(self) -> list[float]:
        values = self._asdict()
        return [values[name] for name in FEATURE_ORDER]


def extract_wing_print_features(gray: np.ndarray) -> RawWingPrintFeatures | None:
    """
    Computes the raw exposure/sharpness/texture feature set a padded bat crop
    is scored on. Returns None if the crop is too small/empty to segment a
    bat mask out of at all.

    Args:
        gray: single-channel (grayscale) padded crop.

    Returns:
        RawWingPrintFeatures, or None if no usable bat mask was found.
    """
    if gray.size == 0:
        return None

    # Background is near-black IR/flash footage
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_bool = mask > 0

    if mask_bool.sum() < _MIN_MASK_PIXELS:
        return None

    bat_area_frac = float(mask_bool.sum()) / gray.size

    ys, xs = np.where(mask_bool)
    mask_w = int(xs.max() - xs.min() + 1)
    mask_h = int(ys.max() - ys.min() + 1)
    mask_aspect = mask_w / mask_h

    bat_pixels = gray[mask_bool]
    mean_brightness = float(bat_pixels.mean())
    p90_brightness = float(np.percentile(bat_pixels, 90))
    # Fraction of the bat blown out by the flash
    saturation_frac = float((bat_pixels >= 250).sum()) / mask_bool.sum()

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_var = float(laplacian[mask_bool].var())

    edges = cv2.Canny(gray, 30, 90)
    edge_density = float((edges[mask_bool] > 0).sum()) / mask_bool.sum()

    # GLCM texture descriptors, computed on the bat's bounding-box patch with
    # background pixels zeroed out. Separate membrane venation from fur/body texture
    patch = gray[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].copy()
    patch_mask = mask_bool[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    patch[~patch_mask] = 0
    quantized = (patch.astype(np.float32) / 256 * 32).astype(np.uint8)
    glcm = graycomatrix(  # type: ignore[no-untyped-call]
        quantized,
        distances=[3],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=32,
        symmetric=True,
        normed=True,
    )
    homogeneity = float(graycoprops(glcm, "homogeneity").mean())  # type: ignore[no-untyped-call]
    correlation = float(graycoprops(glcm, "correlation").mean())  # type: ignore[no-untyped-call]
    contrast = float(graycoprops(glcm, "contrast").mean())  # type: ignore[no-untyped-call]
    energy = float(graycoprops(glcm, "energy").mean())  # type: ignore[no-untyped-call]

    return RawWingPrintFeatures(
        mean_brightness=mean_brightness,
        p90_brightness=p90_brightness,
        laplacian_var=laplacian_var,
        edge_density=edge_density,
        bat_area_frac=bat_area_frac,
        saturation_frac=saturation_frac,
        mask_aspect=mask_aspect,
        homogeneity=homogeneity,
        correlation=correlation,
        contrast=contrast,
        energy=energy,
    )


class WingPrint:
    def __init__(self, output: Path) -> None:
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._min_detection_conf: float = 0.2
        self._clear_probability_threshold: float = 0.5
        self._crop_padding_ratio: float = CROP_PADDING_RATIO

        self._model = joblib.load(_MODEL_PATH)

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
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop

        features = extract_wing_print_features(gray)
        if features is None:
            return WingPrintMetrics(
                mean_brightness=0.0,
                p90_brightness=0.0,
                laplacian_var=0.0,
                edge_density=0.0,
                bat_area_frac=0.0,
                saturation_frac=0.0,
                mask_aspect=0.0,
                homogeneity=0.0,
                correlation=0.0,
                contrast=0.0,
                energy=0.0,
                clear_probability=0.0,
                is_clear=False,
            )

        clear_probability = float(self._model.predict_proba([features.as_vector()])[0, 1])
        is_clear = clear_probability >= self._clear_probability_threshold

        return WingPrintMetrics(
            **features._asdict(),
            clear_probability=clear_probability,
            is_clear=is_clear,
        )

    def detect_wing_prints(self, detections_paths: list[Path], frames_paths: list[Path]) -> None:
        """
        Runs the wing-print detection process.

        For each detections file and its corresponding frames folder, loads the
        detection outputs, crops each detected bat and scores whether the wing print
        is clearly visible in the crop.

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
                    x1, y1, x2, y2 = pad_and_clamp_bbox(self._crop_padding_ratio, bbox_px, img_w, img_h)
                    crop = img[y1:y2, x1:x2]

                    metrics = self.score_wing_print_visibility(crop)
                    if not metrics.is_clear:
                        continue

                    margin = BORDER_TOUCH_MARGIN
                    if x1 <= margin or y1 <= margin or x2 >= img_w - margin or y2 >= img_h - margin:
                        crop_name = f"{Path(image_result.file).stem}_det{det_idx}.jpg"
                        cv2.imwrite(str(self._output / crop_name), crop)

            self._logger.info(f"Finished {detection_path} in {time.time() - start:.2f}s")
