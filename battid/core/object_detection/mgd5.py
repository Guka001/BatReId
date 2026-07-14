import logging
import shutil
from pathlib import Path

from megadetector.detection.run_detector_batch import load_and_run_detector_batch, write_results_to_file

from ..models.detection_model_output import DCOutput
from .base_detection_model import BaseDetectionModel


class MGD5(BaseDetectionModel):
    def __init__(self) -> None:
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._model_name: str = "MDV5A"
        self._tmp: Path = Path(__file__).parent.joinpath(".tmp")

    def run_detection(self, image_folder_path: Path) -> DCOutput:
        if not image_folder_path.is_dir():
            raise NotADirectoryError(f"Source {image_folder_path} must be a directory")

        self._logger.info(f"Running detection. Using model {self._model_name}")

        results = load_and_run_detector_batch(self._model_name, str(image_folder_path))

        self._tmp.mkdir(exist_ok=True)
        out = self._tmp.joinpath("result.json")
        write_results_to_file(results, str(out), relative_path_base=image_folder_path, detector_file=self._model_name)

        if out.exists():
            result = DCOutput.model_validate_json(out.read_text())
            shutil.rmtree(self._tmp)
            return result

        raise Exception("An unexpected error occurred while generating the detection result object")
