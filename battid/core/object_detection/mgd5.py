import logging
import shutil
import tempfile
import time
from pathlib import Path

from megadetector.detection.run_detector_batch import load_and_run_detector_batch, write_results_to_file

from battid.core.object_detection.base_detection_model import BaseDetectionModel
from battid.models.detection_model_output import DCOutput


class MGD5(BaseDetectionModel):
    def __init__(self) -> None:
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._model_name: str = "MDV5A"
        self._batch_size: int = 16

    def run_detection(self, image_folder_path: Path) -> tuple[DCOutput, float]:
        if not image_folder_path.is_dir():
            raise NotADirectoryError(f"Source {image_folder_path} must be a directory")

        self._logger.info(f"Running detection. Using model {self._model_name}")

        start_time = time.time()
        results = load_and_run_detector_batch(self._model_name, str(image_folder_path), batch_size=self._batch_size)
        duration = time.time() - start_time

        tmp_dir = Path(tempfile.mkdtemp(prefix=f"mgd5_{image_folder_path.name}_"))
        out = tmp_dir.joinpath("result.json")
        write_results_to_file(results, str(out), relative_path_base=image_folder_path, detector_file=self._model_name)

        if out.exists():
            result = DCOutput.model_validate_json(out.read_text())
            shutil.rmtree(tmp_dir)
            return result, duration

        raise Exception("An unexpected error occurred while generating the detection result object")
