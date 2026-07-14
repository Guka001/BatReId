from pydantic import BaseModel


class SequenceRecord(BaseModel):
    video: str
    track_id: int
    frame_start: int
    frame_end: int
    output_dir: str
