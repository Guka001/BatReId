"""Utilities for writing human-readable pipeline reports"""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from battid.core.utils import format_duration
from battid.models.report import Report, Task, TrackingReport

matplotlib.use("Agg")  # headless
_EXCLUDED_FROM_GENERIC_DUMP = {"description", "duration", "flights", "discarded_flights"}


def _group_reports_by_task(reports: list[Report]) -> dict[Task, list[Report]]:
    grouped: dict[Task, list[Report]] = defaultdict(list)
    for report in reports:
        grouped[report.task].append(report)
    return grouped


def _plot_flight(
    track_id: int,
    bboxes: list[tuple[float, float, float, float]],
    roi: dict[str, int],
    output_path: Path,
) -> None:
    centroids = [((x1 + x2) / 2.0, (y1 + y2) / 2.0) for x1, y1, x2, y2 in bboxes]
    xs = [c[0] for c in centroids]
    ys = [c[1] for c in centroids]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(xs, ys, "-o", color="tab:blue", markersize=3, linewidth=1, label=f"Track {track_id}")
    ax.plot(xs[0], ys[0], "o", color="green", markersize=8, label="Start")
    ax.plot(xs[-1], ys[-1], "o", color="red", markersize=8, label="End")

    roi_rect = Rectangle(
        (roi["x1"], roi["y1"]),
        roi["x2"] - roi["x1"],
        roi["y2"] - roi["y1"],
        linewidth=1.5,
        edgecolor="orange",
        facecolor="none",
        linestyle="--",
        label="ROI",
    )
    ax.add_patch(roi_rect)

    ax.set_title(f"Track {track_id} flight path")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_reports(
    reports: list[Report],
    video: Path,
    output_root: Path,
    roi: dict[str, int],
) -> Path:
    video_report_dir = output_root.joinpath("reports", video.stem)
    video_report_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = video_report_dir.joinpath("plots")
    plots_dir.mkdir(exist_ok=True)

    grouped = _group_reports_by_task(reports)

    lines: list[str] = [f"# Pipeline report — {video.name}", ""]

    for task in Task:
        task_reports = grouped.get(task)
        if not task_reports:
            continue

        lines.append(f"## {task.value}")
        lines.append("")

        for report in task_reports:
            lines.append(f"- **Description**: {report.description}")
            lines.append(f"- **Duration**: {format_duration(report.duration)}")

            extra_fields = report.model_dump(exclude=_EXCLUDED_FROM_GENERIC_DUMP)
            for field_name, value in extra_fields.items():
                lines.append(f"- **{field_name}**: {value}")

            if isinstance(report, TrackingReport):
                if report.discarded_flights:
                    discard_lengths = [len(bboxes) for bboxes in report.discarded_flights.values()]
                    avg_len = sum(discard_lengths) / len(discard_lengths)
                    lines.append(
                        f"- **Discarded tracks**: {len(discard_lengths)} "
                        f"(avg length: {avg_len:.1f} frames, min: {min(discard_lengths)}, "
                        f"max: {max(discard_lengths)}, threshold: {report.min_track_length})"
                    )
                else:
                    lines.append("- **Discarded tracks**: 0")

                if not report.flights:
                    lines.append("- No tracks kept — no flight plots generated.")
                else:
                    lines.append("- **Flight plots (kept)**:")
                    for track_id, bboxes in report.flights.items():
                        plot_filename = f"track_{track_id}_flight.png"
                        _plot_flight(track_id, bboxes, roi, plots_dir.joinpath(plot_filename))
                        lines.append(f"  - Track {track_id}: `plots/{plot_filename}`")

                if report.discarded_flights:
                    lines.append("- **Flight plots (discarded)**:")
                    for track_id, bboxes in report.discarded_flights.items():
                        plot_filename = f"discarded_track_{track_id}_flight.png"
                        _plot_flight(track_id, bboxes, roi, plots_dir.joinpath(plot_filename))
                        lines.append(f"  - Track {track_id}: `plots/{plot_filename}`")

            lines.append("")
            lines.append("---")
            lines.append("")

    report_file = video_report_dir.joinpath("report.md")
    report_file.write_text("\n".join(lines), encoding="utf-8")

    return report_file


@dataclass
class PipelineStats:
    """
    Accumulates cross-video stats:
    overlap counts and discarded-track lengths, pooled across all videos.
    """

    overlaps_per_video: dict[str, int] = field(default_factory=dict)
    discarded_lengths_per_video: dict[str, list[int]] = field(default_factory=dict)

    def record(self, video: Path, reports: list[Report]) -> None:
        tracking_report = next((r for r in reports if isinstance(r, TrackingReport)), None)
        if tracking_report is None or not tracking_report.overlaps:
            return

        self.overlaps_per_video[video.stem] = tracking_report.number_of_overlaps
        self.discarded_lengths_per_video[video.stem] = [
            len(bboxes) for bboxes in tracking_report.discarded_flights.values()
        ]

    def _plot_overlaps_per_video(self, output_path: Path) -> None:
        """Bar chart: number of overlap episodes per video, across the dataset."""
        video_stems = list(self.overlaps_per_video.keys())
        counts = [self.overlaps_per_video[stem] for stem in video_stems]

        fig, ax = plt.subplots(figsize=(max(8, len(video_stems) * 0.5), 6))
        ax.bar(range(len(video_stems)), counts, color="tab:orange")
        ax.set_xticks(range(len(video_stems)))
        ax.set_xticklabels(video_stems, rotation=75, ha="right", fontsize=7)
        ax.set_ylabel("Number of overlap episodes")
        ax.set_title("Overlap episodes per video")

        if counts:
            mean_overlaps = sum(counts) / len(counts)
            ax.axhline(mean_overlaps, color="tab:blue", linestyle="--", linewidth=1, label=f"Mean: {mean_overlaps:.1f}")
            ax.legend()

        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    def write_summary(self, output_root: Path) -> Path:
        reports_dir = output_root.joinpath("reports")
        plots_dir = reports_dir.joinpath("plots")
        plots_dir.mkdir(parents=True, exist_ok=True)

        total_overlaps = sum(self.overlaps_per_video.values())
        videos_with_overlaps = sum(1 for count in self.overlaps_per_video.values() if count > 0)
        mean_overlaps = total_overlaps / len(self.overlaps_per_video) if self.overlaps_per_video else 0.0

        all_discarded_lengths = [length for lengths in self.discarded_lengths_per_video.values() for length in lengths]

        lines: list[str] = ["# Cross-video summary", ""]

        lines.append("## Overlaps")
        lines.append(f"- Total overlap episodes across all videos: {total_overlaps}")
        lines.append(f"- Videos with at least one overlap: {videos_with_overlaps} / {len(self.overlaps_per_video)}")
        lines.append(f"- Mean overlaps per video: {mean_overlaps:.2f}")

        if self.overlaps_per_video:
            plot_filename = "overlaps_per_video.png"
            self._plot_overlaps_per_video(plots_dir.joinpath(plot_filename))
            lines.append(f"- Plot: `plots/{plot_filename}`")

        lines.append("")

        lines.append("## Discarded tracks")
        if all_discarded_lengths:
            avg_len = sum(all_discarded_lengths) / len(all_discarded_lengths)
            lines.append(f"- Total discarded tracks: {len(all_discarded_lengths)}")
            lines.append(f"- Average length: {avg_len:.1f} frames")
            lines.append(f"- Min / Max: {min(all_discarded_lengths)} / {max(all_discarded_lengths)}")
        else:
            lines.append("- No discarded tracks across any video.")
        lines.append("")

        summary_file = reports_dir.joinpath("summary.md")
        summary_file.write_text("\n".join(lines), encoding="utf-8")
        return summary_file
