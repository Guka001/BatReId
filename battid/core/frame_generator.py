import logging
import re
from pathlib import Path

import cv2


class FrameGenerator:
    """Utility class to convert a video file into a sequence of image frames.

    This class provides functionality for reading a video file and writing each
    frame as a sequentially numbered JPEG image into a destination directory.
    """

    _logger: logging.Logger = logging.getLogger(__name__)
    _FRAME_IDX_RE = re.compile(r"_frame_(\d+)\.jpg$")

    @classmethod
    def deconstruct_video_into_frames(cls, source: Path, destination: Path) -> None:
        """Extract all frames from a video file and save them as JPEG images.

        Args:
            source (Path): Path to the source video file. Must exist and be a file.
            destination (Path): Directory where extracted frames will be written.
            Wil be created if it does not already exist.
        """
        if not source.is_file():
            raise FileNotFoundError(f"Source: {source} must be a valid file")

        destination.mkdir(exist_ok=True)
        cls._logger.info(f"Parsing video {source}")

        video: cv2.VideoCapture = cv2.VideoCapture(source)
        frame_idx: int = 0

        while True:
            success, frame = video.read()
            if not success:
                break

            out = destination.joinpath(f"{source.name}_frame_{frame_idx:04d}.jpg")
            cv2.imwrite(out, frame)
            frame_idx += 1

        video.release()
        cls._logger.info("Parsing successful")

    @staticmethod
    def get_frame_size(frames_path: Path) -> tuple[int, int]:
        """Get the width and height of the first frame in a sequence.

        Args:
            frames_path (Path): Path to the folder containing the generated frames.

        Returns:
            (tuple[int, int]): Width and height of the first frame.
        """

        first_frame = next(frames_path.glob("*.jpg"))
        image = cv2.imread(str(first_frame))
        if image is None:
            raise ValueError(f"Could not read image: {first_frame}")

        height, width = image.shape[:2]
        return width, height

    @staticmethod
    def get_video_fps(video: Path) -> int:
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return int(round(fps)) if fps and fps > 0 else 25

    @classmethod
    def parse_frame_index(cls, frame_filename: str) -> int:
        """Extract the frame index from a filename produced by FrameGenerator,
        e.g. 'video1.mp4_frame_0042.jpg' -> 42.
        """
        match = cls._FRAME_IDX_RE.search(frame_filename)
        if not match:
            raise ValueError(f"Could not parse frame index from filename: {frame_filename}")
        return int(match.group(1))
