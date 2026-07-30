from pydantic import BaseModel, ConfigDict, Field


class Track:
    """
    Represents a single tracked object's trajectory across frames.

    Accumulates the sequence of frame indices in which the object was
    detected along with its pixel-space centroid at each frame.
    """

    def __init__(self, track_id: int) -> None:
        self.id: int = track_id
        self.frames: list[int] = []
        self.history_frame_to_centroid: dict[int, tuple[float, float]] = {}
        self.history_frame_to_bbox: dict[int, tuple[float, float, float, float]] = {}

    def add(
        self,
        frame_idx: int,
        centroid: tuple[float, float],
        bbox: tuple[float, float, float, float],
    ) -> None:
        self.frames.append(frame_idx)
        self.history_frame_to_centroid[frame_idx] = centroid
        self.history_frame_to_bbox[frame_idx] = bbox


class OverlapEpisode(BaseModel):
    """A contiguous run of frames in which two tracked IDs' boxes overlapped."""

    track_id_a: int
    track_id_b: int
    start_frame: int
    end_frame: int
    frame_ious: list[float] = Field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return len(self.frame_ious)

    @property
    def mean_iou(self) -> float:
        return sum(self.frame_ious) / len(self.frame_ious) if self.frame_ious else 0.0


class TrackingResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    tracks: list[Track]
    num_ids: int
    overlap_episodes: list[OverlapEpisode]
    duration: float
