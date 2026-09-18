"""Plan complete source windows and extract bounded original-frame previews.

Prerequisites: the root uv environment, FFmpeg/ffprobe, and a local video.
Run from the repository root::

    uv run python examples/sample_source_video.py recording.mp4 --output data/source-frames

Writes JPEGs under a new output directory and prints one JSON record per window.
The source is read only. HFlow may download its managed FFmpeg binaries; no
model calls, cloud storage, scheduler, or credentials are used. Failed later
windows leave already completed windows for inspection; reruns need a new output
directory and do not resume a previous run.
"""

import argparse
import json
from pathlib import Path

from hflow import SourceFrameSampling, SourceSamplingMode, plan_source_windows, sample_source_frames
from hflow.media import UnreadableVideo, UnsupportedVideo, VideoProperties, probe_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-window-millis", type=int, default=120_000)
    parser.add_argument("--maximum-frames", type=int, default=16)
    parser.add_argument(
        "--mode",
        type=SourceSamplingMode,
        choices=list(SourceSamplingMode),
        default=SourceSamplingMode.UNIFORM,
    )
    arguments = parser.parse_args()
    settings = SourceFrameSampling(
        mode=arguments.mode,
        maximum_frames=arguments.maximum_frames,
        maximum_window_millis=arguments.maximum_window_millis,
    )
    match probe_video(arguments.source):
        case VideoProperties() as properties:
            windows = plan_source_windows(
                properties.duration_millis,
                maximum_window_millis=settings.maximum_window_millis,
            )
        case UnreadableVideo():
            raise SystemExit("Source video is unreadable")
        case UnsupportedVideo():
            raise SystemExit("Source video exceeds the inspection limits")
    arguments.output.mkdir(parents=True, exist_ok=False)
    for window_index, window in enumerate(windows):
        samples = sample_source_frames(
            arguments.source,
            arguments.output / f"window-{window_index:04d}",
            window=window,
            settings=settings,
        )
        print(
            json.dumps(
                {
                    "start_millis": window.start_millis,
                    "end_millis": window.end_millis,
                    "actual_mode": samples.actual_mode,
                    "fallback_reason": samples.fallback_reason,
                    "frames": [
                        {
                            "path": str(frame.path),
                            "timestamp_seconds": float(frame.timestamp_seconds),
                        }
                        for frame in samples.frames
                    ],
                }
            )
        )


if __name__ == "__main__":
    main()
