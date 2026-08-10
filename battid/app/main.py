import argparse
import socket
from pathlib import Path
from typing import Any

import fiftyone as fo

from battid.app import _DATASET_NAME
from battid.app.builder import build_dataset

PORT = 5151


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "frames_dir",
        type=Path,
        help="Path to the directory containing the pipeline's output frames",
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help=(
            "Bind to all interfaces so the app is reachable from other "
            "machines on the network. Default is localhost-only."
        ),
    )
    return parser.parse_args()


def get_local_ip() -> Any:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except OSError:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def run() -> None:
    args = parse_args()

    # Ingest any new frames.
    dataset = build_dataset(args.frames_dir, _DATASET_NAME)

    address = "0.0.0.0" if args.remote else "127.0.0.1"  # noqa: S104

    session = fo.launch_app(
        dataset,
        remote=True,
        address=address,
        port=PORT,
        auto=False,
    )

    display_host = get_local_ip() if args.remote else "localhost"

    print("=" * 60)
    print(f"Application Server is running at http://{display_host}:{PORT}")
    print("Press Ctrl+C to stop the server.")
    print("=" * 60)

    session.wait()


if __name__ == "__main__":
    run()
