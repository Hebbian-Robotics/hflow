"""Evaluate a real egocentric episode with local and hosted HFlow checks.

The normal ``hflow.App`` baseline stays enabled. Two explicitly registered
hosted checks add Build AI's hand-visibility and active-manipulation judgments.
Contact us to get access to the hosted API.

Run from the repository root::

    uv run python examples/evaluate_episode.py episode.mcap \\
        --hosted-base-url URL [--camera TOPIC]
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import hflow

DEFAULT_DATA_ROOT = Path("data/episode-evaluation")


def build_application(
    *,
    data_root: Path,
    hosted_base_url: str,
    camera: str | None,
    frame_time_seconds: float,
) -> hflow.App:
    application = hflow.App("episode-evaluation", data_root=data_root)
    hosted_execution = hflow.build_ai_vlm_checks.HFlowHostedExecution(base_url=hosted_base_url)
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
        "--hosted-base-url",
        required=True,
        help="HFlow hosted API URL; contact us to get access",
    )
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
        hosted_base_url=arguments.hosted_base_url,
        camera=arguments.camera,
        frame_time_seconds=arguments.frame_time_seconds,
    )
    report = asyncio.run(application.test(arguments.episode))
    if report.has_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
