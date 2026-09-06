"""Build one faulted egocentric episode and export its evidence as a timeline bundle.

Prerequisites: ffmpeg on ``PATH`` and a source video (any container ffmpeg
reads). The Build AI checks call HFlow's hosted check service, which admits one
request at a time and about ten per minute sustained after a 100-request
burst, so a 30 second excerpt at 1 fps takes roughly one minute per check.

External side effects: sends one JPEG per sampled frame to
``https://api.hflow.dev``; writes an HFlow workspace under ``--data-root``.

Run from the repository root:

    uv run python examples/fault_timeline/build.py \
        --source /path/to/egocentric.mp4 --out data/fault-timeline

Result: ``<out>/episode.mp4`` (the camera stream, browser-playable) and
``<out>/timeline.json`` (the episode's recorded time axis, every interval the
checks found with seconds relative to that axis, and the video's offset on it).
Black and frozen segments are injected into the excerpt; hands-absent and
no-manipulation spans are whatever the sampled Build AI checks found in the
footage.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import hflow
from hflow.build_ai_vlm_checks import (
    FrameSampling,
    HFlowHostedExecution,
    register_active_manipulation,
    register_hand_visibility,
)
from hflow.camera_video import CAMERA_VIDEO_VERSION, camera_video, video_artifact_name
from hflow.curation import open_catalog_connection
from hflow.testing import VideoEpisodeSpec, write_video_episode

NANOSECONDS_PER_SECOND = 1_000_000_000


def build_pipeline(data_root: Path, *, sample_fps: float) -> hflow.App:
    """The default checks plus the video artifact and the sampled Build AI checks."""
    app = hflow.App("fault-timeline", data_root=data_root)
    app.enrich(version=CAMERA_VIDEO_VERSION)(camera_video)
    sampling = FrameSampling(fps=sample_fps)
    register_hand_visibility(app, execution=HFlowHostedExecution(), sampling=sampling)
    register_active_manipulation(app, execution=HFlowHostedExecution(), sampling=sampling)
    return app


def export_timeline_bundle(
    *, data_root: Path, report: hflow.TestReport, camera_topic: str, out: Path
) -> Path:
    """Write ``timeline.json`` and copy the camera MP4 beside it."""
    catalog_entry = report.catalog_entry
    if catalog_entry is None:
        raise RuntimeError("the pipeline recorded nothing; run with record=True")
    episode_id = catalog_entry.episode_id
    connection = open_catalog_connection(data_root / "catalog")
    try:
        episode_row = connection.execute(
            "SELECT run_fingerprint, start_ns, end_ns FROM episodes_latest WHERE episode_id = ?",
            [episode_id],
        ).fetchone()
        if episode_row is None or episode_row[1] is None:
            raise RuntimeError("the episode row carries no time bounds")
        run_fingerprint, start_ns, end_ns = episode_row
        interval_rows = connection.execute(
            "SELECT label, start_ns, end_ns, check_name FROM intervals "
            "WHERE episode_id = ? AND run_fingerprint = ? ORDER BY start_ns, label",
            [episode_id, run_fingerprint],
        ).fetchall()
        measurement_rows = connection.execute(
            "SELECT key, value_double, value_text FROM measurements_latest "
            "WHERE episode_id = ? ORDER BY key",
            [episode_id],
        ).fetchall()
    finally:
        connection.close()

    measurements = {
        key: value_double if value_text is None else value_text
        for key, value_double, value_text in measurement_rows
    }
    video_uri = measurements[f"artifact/{video_artifact_name(camera_topic)}"]
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(video_uri), out / "episode.mp4")

    def relative_seconds(absolute_ns: int) -> float:
        return (absolute_ns - start_ns) / NANOSECONDS_PER_SECOND

    bundle = {
        "episode_id": episode_id,
        "camera": camera_topic,
        "duration_s": (end_ns - start_ns) / NANOSECONDS_PER_SECOND,
        "video": {
            "file": "episode.mp4",
            "start_s": measurements[f"{camera_topic}/video_start_s"],
            "fps": measurements[f"{camera_topic}/video_fps"],
        },
        "intervals": [
            {
                "label": label,
                "kind": label.split(":", 1)[0],
                "check_name": check_name,
                "start_s": relative_seconds(interval_start_ns),
                "end_s": relative_seconds(interval_end_ns),
            }
            for label, interval_start_ns, interval_end_ns, check_name in interval_rows
        ],
        "checks": sorted({run.check.name: run.check.version for run in report.checks}.items()),
    }
    bundle_path = out / "timeline.json"
    bundle_path.write_text(json.dumps(bundle, indent=2) + "\n")
    return bundle_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, required=True, help="source video file")
    parser.add_argument("--out", type=Path, required=True, help="bundle output directory")
    parser.add_argument("--data-root", type=Path, default=None, help="HFlow workspace")
    parser.add_argument("--source-start-s", type=float, default=35.0)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--image-hz", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=456)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--black", type=float, nargs=2, default=(8.0, 9.5), metavar=("S", "E"))
    parser.add_argument("--freeze", type=float, nargs=2, default=(26.0, 28.5), metavar=("S", "E"))
    parser.add_argument("--sample-fps", type=float, default=1.0)
    arguments = parser.parse_args()

    out: Path = arguments.out
    data_root: Path = arguments.data_root if arguments.data_root is not None else out / "workspace"
    spec = VideoEpisodeSpec(
        source_start_s=arguments.source_start_s,
        duration_s=arguments.duration_s,
        image_hz=arguments.image_hz,
        image_width=arguments.width,
        image_height=arguments.height,
        black_segment=tuple(arguments.black),
        freeze_segment=tuple(arguments.freeze),
        task="factory_assembly_excerpt",
    )
    source_episode = write_video_episode(arguments.source, out / "faulted_input.mcap", spec)
    print(f"wrote faulted input episode {source_episode}")

    app = build_pipeline(data_root, sample_fps=arguments.sample_fps)
    report = app.process(source_episode, verbose=True)
    camera_topic = f"/{spec.camera_name}/compressed"
    bundle_path = export_timeline_bundle(
        data_root=data_root, report=report, camera_topic=camera_topic, out=out
    )
    print(f"wrote {bundle_path} and {out / 'episode.mp4'}")


if __name__ == "__main__":
    main()
