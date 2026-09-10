"""Process local video excerpts with a temporary workspace and no scheduler.

Prerequisites: the root uv environment and FFmpeg (HFlow downloads its managed
build if needed). No model service or API key. This reads local inputs, writes
temporary episode artifacts, removes those artifacts, and prints measurements.
It never modifies the source videos or writes a persistent catalog.

Run from the repository root::

    uv run python examples/embedded_worker.py recording-a.mp4 recording-b.mp4 \
        --duration-s 5 --max-workers 2
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

from hflow import (
    App,
    CheckResult,
    Episode,
    MeasurementValue,
    Stage,
    VideoImportConfig,
    checks,
    import_video_episode,
)


def process_videos(
    source_paths: Sequence[Path],
    *,
    duration_s: float,
    max_workers: int,
) -> list[dict[str, MeasurementValue]]:
    """Return input-ordered camera measurements after removing task artifacts."""

    with TemporaryDirectory(prefix="hflow-worker-") as workspace_directory:
        workspace_path = Path(workspace_directory)
        application = App("video-worker", data_root=workspace_path, default_checks=())

        @application.check(version="1")
        def camera_quality(episode: Episode) -> CheckResult:
            return checks.camera_frame_stats(episode)

        episode_paths = [
            import_video_episode(
                source_path,
                workspace_path / "inputs" / f"episode-{source_index}.mcap",
                VideoImportConfig(duration_s=duration_s),
            )
            for source_index, source_path in enumerate(source_paths)
        ]
        batch_report = application.process_many(
            episode_paths,
            record=False,
            stages=(Stage.SYNC, Stage.META),
            max_workers=max_workers,
        )
        measurements: list[dict[str, MeasurementValue]] = []
        for report in batch_report.reports:
            check_run = report.check("camera_quality")
            if report.has_errors or check_run.result is None:
                raise RuntimeError(f"camera check did not complete: {report.summary()}")
            measurements.append(dict(check_run.result.measurements))
        return measurements


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--max-workers", type=int, default=2)
    arguments = parser.parse_args()
    measurements = process_videos(
        arguments.videos,
        duration_s=arguments.duration_s,
        max_workers=arguments.max_workers,
    )
    print(json.dumps(measurements, indent=2))


if __name__ == "__main__":
    main()
