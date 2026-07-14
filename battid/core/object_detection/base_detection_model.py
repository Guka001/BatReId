from abc import ABC, abstractmethod
from pathlib import Path

from ..models.detection_model_output import DCOutput


class BaseDetectionModel(ABC):
    """
    Utility base class that defines the contract for detection models which
    extract bounding box coordinates for objects found in images.
    """

    @abstractmethod
    def run_detection(self, image_folder_path: Path) -> DCOutput:
        """Run object detection on all images in a folder and save results.

        Processes the images located in `image_folder_path` and returns
        the detection results as an instance of the class `DCOutput`.

        Args:
            image_folder_path (Path): Directory containing input image files.

        Returns:
            (DCOutput): The detection results.
        """
        pass
