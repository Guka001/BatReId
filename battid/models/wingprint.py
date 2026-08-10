from pydantic import BaseModel


class WingPrintMetrics(BaseModel):
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
    clear_probability: float
    is_clear: bool
