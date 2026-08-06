import re
from pathlib import Path

import fiftyone as fo

FRAME_GLOB = "**/*.jpg"


def _parse_video_and_frame_id(path: Path) -> tuple[str, int | None]:
    video_id = path.parent.name
    match = re.search(r"(\d+)", path.stem)
    frame_number = int(match.group(1)) if match else None
    return video_id, frame_number


def build_dataset(frames_dir: Path, dataset_name: str) -> fo.Dataset:
    frames_dir = Path(frames_dir)

    if dataset_name in fo.list_datasets():
        dataset = fo.load_dataset(dataset_name)
        print(f"Loaded existing dataset '{dataset_name}' ({len(dataset)} samples).")
    else:
        dataset = fo.Dataset(dataset_name, persistent=True)
        print(f"Created new persistent dataset '{dataset_name}'.")

    existing_paths = set(dataset.values("filepath"))

    new_samples = []
    for frame_path in sorted(frames_dir.glob(FRAME_GLOB)):
        abs_path = str(frame_path.resolve())
        if abs_path in existing_paths:
            continue

        video_id, frame_number = _parse_video_and_frame_id(frame_path)

        sample = fo.Sample(filepath=abs_path)
        sample["video_id"] = video_id
        sample["frame_number"] = frame_number

        sample["valid"] = None
        new_samples.append(sample)

    if new_samples:
        dataset.add_samples(new_samples)
        print(f"Added {len(new_samples)} new frames.")
    else:
        print("No new frames found (dataset already up to date).")

    dataset.save()
    print(f"Dataset '{dataset_name}' now has {len(dataset)} total samples.")

    return dataset
