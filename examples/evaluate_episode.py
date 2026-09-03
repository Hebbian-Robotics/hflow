"""Evaluate a real egocentric episode with local and hosted HFlow checks.

The normal ``hflow.App`` baseline stays enabled. Two explicitly registered
hosted checks add Build AI's hand-visibility and active-manipulation judgments.

Run from the repository root::

    uv run python examples/evaluate_episode.py episode.mcap [--camera TOPIC]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import hflow

DEFAULT_DATA_ROOT = Path("data/episode-evaluation")


def build_application(
    *,
    data_root: Path,
    camera: str | None,
    frame_time_seconds: float,
) -> hflow.App:
    application = hflow.App("episode-evaluation", data_root=data_root)
    hosted_execution = hflow.build_ai_vlm_checks.HFlowHostedExecution()
    hflow.build_ai_vlm_checks.register_hand_visibility(
        application,
        execution=hosted_execution,
        camera=camera,
        frame_time_seconds=frame_time_seconds,
    )
    hflow.build_ai_vlm_checks.register_active_manipulation(
        application,
        execution=hosted_execution,
        camera=camera,
        frame_time_seconds=frame_time_seconds,
    )
    return application


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path, help="MCAP episode with egocentric video")
    parser.add_argument(
        "--camera",
        help="camera topic to evaluate; required when the episode has multiple cameras",
    )
    parser.add_argument(
        "--frame-time-seconds",
        type=float,
        default=0.0,
        help="seconds from the start of the camera stream to evaluate (default: 0)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"HFlow output root (default: {DEFAULT_DATA_ROOT})",
    )
    return parser


def main() -> None:
    arguments = argument_parser().parse_args()
    application = build_application(
        data_root=arguments.data_root,
        camera=arguments.camera,
        frame_time_seconds=arguments.frame_time_seconds,
    )
    report = application.test(arguments.episode)
    if report.has_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
