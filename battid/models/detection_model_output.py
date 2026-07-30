from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class Detection(BaseModel):
    category: str
    conf: float = Field(ge=0.0, le=1.0)
    bbox: list[float] = Field(min_length=4, max_length=4)

    @field_validator("bbox")
    @classmethod
    def bbox_must_be_normalized(cls, v: list[float]) -> list[float]:
        if any(val < 0.0 or val > 1.0 for val in v):
            raise ValueError(f"bbox values must be normalized (0-1), got {v}")
        return [round(val, 4) for val in v]


class ImageResult(BaseModel):
    file: str
    detections: list[Detection]


class DCOutput(BaseModel):
    images: list[ImageResult]
    detection_categories: dict[str, str]

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))
