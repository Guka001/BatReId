from pydantic import BaseModel, Field


class WingFeatures(BaseModel):
    brightness_mean: float
    brightness_p90: float
    brightness_std: float
    bright_pixel_fraction: float
    laplacian_var: float
    entropy: float
    edge_density: float
    hough_line_count: int
    aspect_ratio: float
    solidity: float
    convexity_defect_score: float
    gradient_orientation_concentration: float

    def to_vector(self, feature_names: list[str]) -> list[float]:
        """Return this record's values ordered to match `feature_names`."""
        values = self.model_dump()
        return [float(values[name]) for name in feature_names]

    @staticmethod
    def names() -> list[str]:
        return list(WingFeatures.model_fields.keys())


class WingCropRecord(BaseModel):
    search_key: str
    frame_idx: int
    detection_index: int
    bbox: tuple[int, int, int, int]
    confidence: float
    crop_path: str
    features: WingFeatures
    label: int | None = Field(default=None, description="1 = wing clearly visible, 0 = not visible")

    @property
    def crop_id(self) -> str:
        return f"{self.search_key}_frame{self.frame_idx:04d}_det{self.detection_index}"


class IlluminationMetrics(BaseModel):
    brightness_mean: float
    brightness_std: float
    brightness_p95: float
    background_median: float = Field(
        description="Median brightness of the ring around the bbox, i.e. local scene brightness"
    )
    relative_brightness: float = Field(description="brightness_p95 - background_median")
    visible_area_fraction: float = Field(
        description="Fraction of the bbox distinguishable from its local background (candidate wing/body tissue)"
    )
    mask_touches_edge: bool = Field(
        description="True if the visible region's bounding box touches the crop edge, i.e. it's likely truncated"
    )
    saturation_fraction: float = Field(description="Fraction of bbox pixels at/near sensor saturation")


class WingVisibilityRecord(BaseModel):
    search_key: str
    frame_idx: int
    detection_index: int
    bbox: tuple[int, int, int, int]
    confidence: float
    metrics: IlluminationMetrics
    wing_visible: bool
    crop_path: str | None = Field(default=None, description="Set only when wing_visible is True")

    @property
    def crop_id(self) -> str:
        return f"{self.search_key}_frame{self.frame_idx:04d}_det{self.detection_index}"
