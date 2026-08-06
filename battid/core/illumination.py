import cv2
import numpy as np

from battid.models.wingprint import IlluminationMetrics

# Decision thresholds

# The deciding factor is *completeness*, not raw brightness: a dim wing whose
# full outline (attachment to tip) is captured counts as visible. A bright
# wing that's cut off mid-membrane, or so overexposed it's washed to a flat
# white blob, does not.
#   - VISIBLE_DELTA segments enough of it as "wing tissue" (bright enough,
#     relative to its own local background, to be distinguishable at all) -
#     gated by MIN_VISIBLE_AREA_FRACTION so a stray hot pixel doesn't count.

#   - EDGE_TOUCH_MARGIN_PX: touching means the wing extends past what was
#     captured, i.e. only part of it is actually visible.

#   - it isn't overexposed: SATURATION_FRACTION_CAP bounds how much of the
#     crop is blown out to near-white, which erases the vein pattern.

VISIBLE_DELTA: float = 12.0
MIN_VISIBLE_AREA_FRACTION: float = 0.04
EDGE_TOUCH_MARGIN_PX: int = 4
SATURATION_LEVEL: int = 250
SATURATION_FRACTION_CAP: float = 0.15

# How far the "surround" ring used for `background_median` extends beyond the
# bbox, as a multiple of the bbox's own width/height on each side.
SURROUND_EXPAND_RATIO: float = 1.0

_MORPH_KERNEL = np.ones((5, 5), np.uint8)


def _expand_bbox(x1: int, y1: int, x2: int, y2: int, ratio: float, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    box_w, box_h = x2 - x1, y2 - y1
    pad_x, pad_y = box_w * ratio, box_h * ratio
    return (
        max(0, int(x1 - pad_x)),
        max(0, int(y1 - pad_y)),
        min(img_w, int(x2 + pad_x)),
        min(img_h, int(y2 + pad_y)),
    )


def compute_background_median(gray_frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    """Median brightness of the ring of pixels around the bbox (the bbox
    itself excluded), used as a per-detection local-scene brightness baseline.
    """
    img_h, img_w = gray_frame.shape[:2]
    ex1, ey1, ex2, ey2 = _expand_bbox(x1, y1, x2, y2, SURROUND_EXPAND_RATIO, img_w, img_h)
    region = gray_frame[ey1:ey2, ex1:ex2]

    mask = np.ones(region.shape, dtype=bool)
    inner_x1, inner_y1 = x1 - ex1, y1 - ey1
    inner_x2, inner_y2 = inner_x1 + (x2 - x1), inner_y1 + (y2 - y1)
    mask[inner_y1:inner_y2, inner_x1:inner_x2] = False

    ring_pixels = region[mask]
    if ring_pixels.size == 0:
        return float(np.median(gray_frame))
    return float(np.median(ring_pixels))


def _segment_visible_region(
    crop: np.ndarray, background_median: float
) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    """Find the region of `crop` that's distinguishable from its local
    background - i.e. plausibly wing/body tissue rather than empty scene.

    Returns:
        (mask, bbox): `mask` is the cleaned binary mask (uint8, 0/255).
        `bbox` is `(x1, y1, x2, y2)` in crop-local pixel coordinates for the
        union of all sizeable contours, or None if nothing segmented.
    """
    mask: np.ndarray = (crop > background_median + VISIBLE_DELTA).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask, None

    min_area = 0.005 * mask.size
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]
    if not boxes:
        boxes = [cv2.boundingRect(max(contours, key=cv2.contourArea))]

    bx1 = min(bx for bx, _by, _bw, _bh in boxes)
    by1 = min(by for _bx, by, _bw, _bh in boxes)
    bx2 = max(bx + bw for bx, _by, bw, _bh in boxes)
    by2 = max(by + bh for _bx, by, _bw, bh in boxes)

    return mask, (bx1, by1, bx2, by2)


def compute_illumination_metrics(gray_frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> IlluminationMetrics:
    """Compute visibility/completeness metrics for one bbox against its local surroundings.

    Args:
        gray_frame: The full grayscale frame the bbox was detected in.
        x1, y1, x2, y2: Bbox pixel coordinates, already padded/clamped to the frame.

    Returns:
        (IlluminationMetrics): The metrics for this bbox.
    """
    crop = gray_frame[y1:y2, x1:x2].astype(np.float64)
    crop_h, crop_w = crop.shape[:2]
    background_median = compute_background_median(gray_frame, x1, y1, x2, y2)
    brightness_p95 = float(np.percentile(crop, 95))

    mask, region_bbox = _segment_visible_region(crop, background_median)
    if region_bbox is None:
        visible_area_fraction = 0.0
        mask_touches_edge = False
    else:
        rx1, ry1, rx2, ry2 = region_bbox
        visible_area_fraction = float(cv2.countNonZero(mask)) / mask.size
        mask_touches_edge = (
            rx1 <= EDGE_TOUCH_MARGIN_PX
            or ry1 <= EDGE_TOUCH_MARGIN_PX
            or rx2 >= crop_w - EDGE_TOUCH_MARGIN_PX
            or ry2 >= crop_h - EDGE_TOUCH_MARGIN_PX
        )

    return IlluminationMetrics(
        brightness_mean=float(crop.mean()),
        brightness_std=float(crop.std()),
        brightness_p95=brightness_p95,
        background_median=background_median,
        relative_brightness=brightness_p95 - background_median,
        visible_area_fraction=visible_area_fraction,
        mask_touches_edge=mask_touches_edge,
        saturation_fraction=float(np.mean(crop >= SATURATION_LEVEL)),
    )


def is_wing_visible(metrics: IlluminationMetrics) -> bool:
    """Decide whether a full wing is clearly visible from its completeness metrics.

    A wing counts as visible when a substantial, un-truncated region is
    distinguishable from the local background and isn't washed out by
    overexposure - regardless of how dim the scene is overall.
    """
    return (
        metrics.visible_area_fraction >= MIN_VISIBLE_AREA_FRACTION
        and not metrics.mask_touches_edge
        and metrics.saturation_fraction <= SATURATION_FRACTION_CAP
    )
