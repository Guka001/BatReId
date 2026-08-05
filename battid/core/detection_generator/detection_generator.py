import logging
from abc import ABC, abstractmethod
from pathlib import Path

from battid.core.object_detection.base_detection_model import BaseDetectionModel
from battid.core.object_detection.mgd5 import MGD5
from battid.models.detection_model_output import DetectionGenerationRecord


class DetectionGenerator(ABC):
    """
    Utility class responsible for turning raw video files into extracted
    frames plus their object-detection results.
    """

    def __init__(self, output: Path, detector: BaseDetectionModel | None = None) -> None:
        self._logger: logging.Logger = logging.getLogger(__file__)
        self._output: Path = output
        self._output.mkdir(exist_ok=True)
        self._frames_output: Path = self._output.joinpath("frames")
        self._frames_output.mkdir(exist_ok=True)

        self._detector: BaseDetectionModel = detector or MGD5()

    @abstractmethod
    def generate(self, videos: list[Path]) -> list[DetectionGenerationRecord]:
        """Extract frames and run detection for each video.

        Videos that fail frame extraction or detection are logged and
        skipped rather than aborting the whole batch.

        Args:
            videos (list[Path]): Path to the source video files to process.

        Returns:
            (list[DetectionGenerationRecord]): A list of detection records for
            successfully processed video.
        """
        raise NotImplementedError()
