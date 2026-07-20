from abc import ABC
from enum import StrEnum
from typing import ClassVar, Self

from pydantic import BaseModel, ValidationError, model_validator


class Task(StrEnum):
    FrameGeneration = "Frame Generation"
    ObjectDetection = "Object Detection"
    Tracking = "Tracking"


class Report(ABC, BaseModel):
    duration: float
    description: str
    task: ClassVar[Task]


class FrameGenerationReport(Report):
    task = Task.FrameGeneration
    number_of_frames_generated: int


class DetectionReport(Report):
    task = Task.ObjectDetection


class TrackingReport(Report):
    task = Task.Tracking
    min_track_length: int
    minimum_matching_threshold: float
    number_of_survived_lost_tracks: int
    overlaps: bool
    number_of_overlaps: int
    raw_number_of_unique_tracks: int
    number_of_unique_tracks_kept: int
    flights: dict[int, list[tuple[float, float, float, float]]]
    discarded_flights: dict[int, list[tuple[float, float, float, float]]]

    @model_validator(mode="after")
    def _validate_tracks(self) -> Self:
        if len(self.flights.keys()) != self.number_of_unique_tracks_kept:
            raise ValidationError("Number of tracks kept must match flight projections")

        return self
