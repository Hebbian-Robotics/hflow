"""Raw FFmpeg blur scores with explicit finite-score coverage, not quality labels."""

from __future__ import annotations

import math
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from hflow.ffmpeg import ffmpeg_path


@dataclass(frozen=True, slots=True)
class BlurSummary:
    frame_count: int
    scored_frame_count: int
    mean_blur_score: float | None


def summarize_blur_scores(scores: Iterable[float]) -> BlurSummary:
    """Average finite raw frame scores; unavailable scores never become zero."""
    frame_count = 0
    scored_frame_count = 0
    mean_blur_score = 0.0
    for score in scores:
        frame_count += 1
        # blurdetect emits NaN when it cannot measure edges, e.g. a flat image.
        if not math.isfinite(score):
            continue
        if score < 0:
            raise RuntimeError("Blur analysis returned an invalid score")
        scored_frame_count += 1
        mean_blur_score += (score - mean_blur_score) / scored_frame_count
    return BlurSummary(
        frame_count=frame_count,
        scored_frame_count=scored_frame_count,
        mean_blur_score=mean_blur_score if scored_frame_count else None,
    )


def measure_video_blur(
    video_path: Path, *, timeout_seconds: float = 120.0, executable: Path | None = None
) -> BlurSummary:
    """Score a local video window at its supplied resolution and frame cadence.

    Uses FFmpeg's default blurdetect settings on the first video stream. The
    result is the frame-weighted mean of finite raw scores, not a percentage or
    probability. No resizing, frame sampling, classification, or model call is
    performed. Results contain raw scores and coverage, without affected-time percentages.
    """
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive and finite")
    try:
        with tempfile.TemporaryFile() as metadata_output:
            subprocess.run(
                [
                    str(executable or ffmpeg_path()),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-xerror",
                    "-protocol_whitelist",
                    "file",
                    "-noautorotate",
                    "-threads",
                    "1",
                    "-i",
                    str(video_path.resolve()),
                    "-map",
                    "0:v:0",
                    "-an",
                    "-sn",
                    "-dn",
                    "-filter_threads",
                    "1",
                    "-vf",
                    "blurdetect,metadata=mode=print:key=lavfi.blur:file=-",
                    "-fps_mode",
                    "passthrough",
                    "-f",
                    "null",
                    "-",
                ],
                stdin=subprocess.DEVNULL,
                stdout=metadata_output,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
                check=True,
            )
            metadata_output.seek(0)
            summary = summarize_blur_scores(
                float(line.removeprefix(b"lavfi.blur="))
                for line in metadata_output
                if line.startswith(b"lavfi.blur=")
            )
            metadata_output.seek(0)
            emitted_frame_count = sum(line.startswith(b"frame:") for line in metadata_output)
        if summary.frame_count != emitted_frame_count:
            raise RuntimeError("Blur analysis returned incomplete frame scores")
        return summary
    except Exception as error:
        raise RuntimeError("Blur analysis failed") from error
