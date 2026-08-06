"""
Splits the reviewed dataset into:
  - valid/    : images + annotations, exported in COCO format.

  - invalid_frames.csv   : filepaths of frames marked invalid.

  - unreviewed_frames.csv : filepaths of frames with neither tag.

Usage:
    python export_dataset.py <output_dir>
"""

import argparse
import csv
from pathlib import Path
from typing import Any

import fiftyone as fo
import fiftyone.types as fot

from battid.app import _DATASET_NAME


def _detect_label_field(dataset: fo.Dataset) -> Any:
    schema = dataset.get_field_schema()
    detection_fields = [
        name
        for name, field in schema.items()
        if isinstance(field, fo.EmbeddedDocumentField) and field.document_type is fo.Detections
    ]

    if not detection_fields:
        return None
    if len(detection_fields) > 1:
        raise ValueError(
            f"Multiple bounding box fields found: {detection_fields}. "
            "Not sure which one to export - remove the one you don't "
            "need, or edit this script to hardcode your choice."
        )

    return detection_fields[0]


def export_results(dataset_name: str, output_dir: Path) -> None:
    dataset = fo.load_dataset(dataset_name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_view = dataset.match_tags("valid")
    invalid_view = dataset.match_tags("invalid")

    reviewed_ids = set(valid_view.values("id")) | set(invalid_view.values("id"))
    unreviewed_view = dataset.exclude(reviewed_ids) if reviewed_ids else dataset

    print(f"Valid: {len(valid_view)}  Invalid: {len(invalid_view)}  Unreviewed: {len(unreviewed_view)}")

    # --- Valid frames ---
    if len(valid_view) > 0:
        label_field = _detect_label_field(dataset)
        valid_export_dir = output_dir / "valid"

        if label_field is not None:
            print(f"Using label field: '{label_field}'")
            valid_view.export(
                export_dir=str(valid_export_dir),
                dataset_type=fot.COCODetectionDataset,
                label_field=label_field,
            )
            print(f"Exported {len(valid_view)} valid frames + annotations to {valid_export_dir}")
        else:
            valid_view.export(
                export_dir=str(valid_export_dir),
                dataset_type=fot.ImageDirectory,
            )
            print(f"No annotations yet. Exported {len(valid_view)} valid frames to {valid_export_dir}")
    else:
        print("No valid frames to export yet.")

    # --- Invalid frames ---
    if len(invalid_view) > 0:
        invalid_manifest = output_dir / "invalid_frames.csv"
        _write_manifest(invalid_manifest, invalid_view)
        print(f"Wrote {len(invalid_view)} invalid frame paths to {invalid_manifest}")

    # --- Anything not yet reviewed: Flag ---
    if len(unreviewed_view) > 0:
        unreviewed_manifest = output_dir / "unreviewed_frames.csv"
        _write_manifest(unreviewed_manifest, unreviewed_view)
        print(f"Warning: {len(unreviewed_view)} frames not yet reviewed - see {unreviewed_manifest}")


def _write_manifest(path: Path, view: fo.DatasetView) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "video_id", "frame_number"])
        for sample in view.select_fields(["filepath", "video_id", "frame_number"]):
            writer.writerow([sample.filepath, sample.video_id, sample.frame_number])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_results(_DATASET_NAME, args.output_dir)
