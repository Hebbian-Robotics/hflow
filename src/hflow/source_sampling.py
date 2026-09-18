"""Bounded JPEG sampling from original video with measured presentation times."""

from __future__ import annotations

import math
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Literal

from hflow._field_guards import require_positive_float, require_positive_int
from hflow.ffmpeg._process import MediaCommandResult, MediaToolError, run_media_command
from hflow.source_windows import SourceWindow

SOURCE_FRAME_SAMPLING_VERSION = "source-frame-sampling-v1"
_TIMESTAMP_PATTERN = re.compile(
    rb"\[Parsed_showinfo_[^\]]+\].*\bn:\s*\d+\s+pts:\s*(-?\d+)\s+pts_time:"
)
_TIME_BASE_PATTERN = re.compile(rb"\[Parsed_showinfo_[^\]]+\].*config in time_base:\s*(\d+)/(\d+)")


class SourceSamplingMode(StrEnum):
    UNIFORM = "uniform"
    KEYFRAMES = "keyframes"
    KEYFRAMES_FIRST = "keyframes_first"


class KeyframeFallbackReason(StrEnum):
    TOO_FEW_KEYFRAMES = "fewer_than_two_keyframes"
    INSUFFICIENT_SPAN = "keyframe_span_below_half_window"


@dataclass(frozen=True)
class SourceFrameSampling:
    """Selection and resource limits shared by every extraction attempt.

    Uniform sampling uses the first frame in each temporal bin, whose length
    is the greater of ``minimum_interval_millis`` and window length divided
    by ``maximum_frames``. Keyframe sampling uses equal bins with no minimum
    interval. Keyframes-first falls back to uniform when fewer than two
    keyframes were selected or their span is less than half the window.

    All sizes are output bounds; callers own source download/decoder memory
    limits. The canvas preserves aspect ratio with black padding. Increasing
    ``maximum_window_millis`` explicitly permits longer source excerpts.
    """

    mode: SourceSamplingMode = SourceSamplingMode.UNIFORM
    maximum_frames: int = 16
    minimum_interval_millis: int = 1_000
    maximum_window_millis: int = 120_000
    width: int = 640
    height: int = 360
    timeout_seconds: float = 120.0
    maximum_frame_bytes: int = 2 * 1024 * 1024
    maximum_log_bytes: int = 2 * 1024 * 1024

    def __post_init__(self) -> None:
        if not isinstance(self.mode, SourceSamplingMode):
            raise ValueError("mode must be a SourceSamplingMode")
        for field_name in (
            "maximum_frames",
            "minimum_interval_millis",
            "maximum_window_millis",
            "width",
            "height",
            "maximum_frame_bytes",
            "maximum_log_bytes",
        ):
            require_positive_int(getattr(self, field_name), field_name)
        require_positive_float(self.timeout_seconds, "timeout_seconds")
        if self.width % 2 or self.height % 2:
            raise ValueError("width and height must be even for JPEG chroma subsampling")


@dataclass(frozen=True)
class SampledSourceFrame:
    """A JPEG and its actual presentation time from the source playback origin.

    The origin is the container start time, not the requested window start.
    ``timestamp_seconds`` retains FFmpeg's rational time base without rounding.
    It is neither an MCAP log time nor an invented uniform sampling tick.
    """

    path: Path
    timestamp_seconds: Fraction


@dataclass(frozen=True)
class SourceFrameSamples:
    window: SourceWindow
    frames: tuple[SampledSourceFrame, ...]
    requested_mode: SourceSamplingMode
    actual_mode: Literal[SourceSamplingMode.UNIFORM, SourceSamplingMode.KEYFRAMES]
    fallback_reason: KeyframeFallbackReason | None


class SourceSamplingError(RuntimeError):
    """Extraction failed, exceeded a resource limit, or could not prove frame timing."""


def _remaining_seconds(deadline: float) -> float:
    remaining_seconds = deadline - time.monotonic()
    if remaining_seconds <= 0:
        raise SourceSamplingError("source frame extraction exceeded its time limit")
    return remaining_seconds


def _run_sampling_command(
    arguments: list[str], *, deadline: float, maximum_log_bytes: int
) -> MediaCommandResult:
    try:
        completed = run_media_command(
            arguments,
            timeout_seconds=_remaining_seconds(deadline),
            maximum_output_bytes=maximum_log_bytes,
        )
    except MediaToolError as error:
        raise SourceSamplingError(str(error)) from None
    if completed.returncode != 0:
        raise SourceSamplingError("source frame extraction failed")
    _remaining_seconds(deadline)
    return completed


def _source_time_base(source_path: Path, executable: Path, *, deadline: float) -> Fraction:
    completed = _run_sampling_command(
        [
            str(executable),
            "-v",
            "error",
            "-protocol_whitelist",
            "file",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=time_base",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(source_path),
        ],
        deadline=deadline,
        maximum_log_bytes=65_536,
    )
    try:
        time_base = Fraction(completed.stdout.decode("ascii").strip())
    except (ValueError, ZeroDivisionError):
        raise SourceSamplingError("source video has no valid presentation time base") from None
    if time_base <= 0:
        raise SourceSamplingError("source video has no valid presentation time base")
    return time_base


def _extract_frames(
    source_path: Path,
    output_directory: Path,
    window: SourceWindow,
    settings: SourceFrameSampling,
    mode: Literal[SourceSamplingMode.UNIFORM, SourceSamplingMode.KEYFRAMES],
    *,
    executable: Path,
    deadline: float,
    time_base: Fraction,
) -> tuple[SampledSourceFrame, ...]:
    duration_seconds = window.duration_millis / 1000
    start_seconds = window.start_millis / 1000
    sampling_interval_millis = Fraction(window.duration_millis, settings.maximum_frames)
    if mode is SourceSamplingMode.UNIFORM:
        sampling_interval_millis = max(
            Fraction(settings.minimum_interval_millis), sampling_interval_millis
        )
    # Reduce to integer PTS arithmetic before dividing into bins. Computing
    # floor((t - start) / interval) in seconds can put a boundary frame in the
    # previous bin (e.g. 4.3 - 4.0 at 10 fps), silently dropping that frame.
    bin_scale = time_base * 1000 / sampling_interval_millis
    bin_offset = Fraction(window.start_millis) / sampling_interval_millis
    common_denominator = math.lcm(bin_scale.denominator, bin_offset.denominator)
    timestamp_multiplier = bin_scale.numerator * (common_denominator // bin_scale.denominator)
    start_offset = bin_offset.numerator * (common_denominator // bin_offset.denominator)
    end_tick = Fraction(window.end_millis, 1000) / time_base
    # FFmpeg expressions use doubles. Refuse timelines whose integer products
    # would no longer be exact instead of silently weakening the bin contract.
    if (
        math.ceil(end_tick) * timestamp_multiplier + start_offset > 2**53
        or common_denominator * settings.maximum_frames * 4 > 2**53
    ):
        raise SourceSamplingError("source timeline exceeds exact frame-selection bounds")
    first_tick = math.ceil(Fraction(window.start_millis, 1000) / time_base)
    end_tick_exclusive = math.ceil(end_tick)
    current_bin = f"floor((pts*{timestamp_multiplier}-{start_offset})/{common_denominator})"
    previous_bin = (
        f"floor((prev_selected_pts*{timestamp_multiplier}-{start_offset})/{common_denominator})"
    )
    # Select original frames, rather than using fps (which can duplicate frames
    # and replace PTS). The bins are relative to the requested window start.
    selection_filter = (
        f"select=gte(pts\\,{first_tick})*lt(pts\\,{end_tick_exclusive})*"
        f"lt(selected_n\\,{settings.maximum_frames})*"
        "(isnan(prev_selected_t)+"
        f"gt({current_bin}\\,{previous_bin}))"
    )
    video_filter = ",".join(
        (
            selection_filter,
            f"scale={settings.width}:{settings.height}:force_original_aspect_ratio=decrease:flags=lanczos",
            f"pad={settings.width}:{settings.height}:(ow-iw)/2:(oh-ih)/2:black",
            "showinfo",
        )
    )
    arguments = [
        str(executable),
        "-hide_banner",
        "-loglevel",
        "info",
        "-nostats",
        "-nostdin",
        "-xerror",
        "-protocol_whitelist",
        "file",
        "-threads",
        "2",
    ]
    if mode is SourceSamplingMode.KEYFRAMES:
        arguments.extend(("-skip_frame", "nokey"))
    # Preserve the original playback PTS through a seek. Adding a seek offset
    # back to rebased PTS would introduce rounding at non-tick-aligned starts.
    # Selection excludes preceding-GOP frames and the exclusive end directly.
    arguments.extend(
        (
            "-copyts",
            "-start_at_zero",
            "-noaccurate_seek",
            "-ss",
            f"{start_seconds:.3f}",
            "-t",
            f"{duration_seconds:.3f}",
            "-i",
            str(source_path),
            "-map",
            "0:v:0",
            "-an",
            "-map_metadata",
            "-1",
            "-vf",
            video_filter,
            "-filter_threads",
            "1",
            "-frames:v",
            str(settings.maximum_frames),
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "yuvj420p",
            "-c:v",
            "mjpeg",
            "-q:v",
            "5",
            "-threads",
            "2",
            "-f",
            "image2",
            str(output_directory / "frame_%06d.jpg"),
        )
    )
    diagnostics = _run_sampling_command(
        arguments, deadline=deadline, maximum_log_bytes=settings.maximum_log_bytes
    ).stderr
    time_bases = {
        (int(match.group(1)), int(match.group(2)))
        for match in _TIME_BASE_PATTERN.finditer(diagnostics)
    }
    if len(time_bases) != 1:
        raise SourceSamplingError("source frame extraction did not report one time base")
    numerator, denominator = time_bases.pop()
    if numerator <= 0 or denominator <= 0 or Fraction(numerator, denominator) != time_base:
        raise SourceSamplingError("source frame extraction reported an invalid time base")
    timestamps = tuple(
        int(match.group(1)) * Fraction(numerator, denominator)
        for match in _TIMESTAMP_PATTERN.finditer(diagnostics)
    )
    frame_paths = tuple(sorted(output_directory.glob("frame_*.jpg")))
    if len(frame_paths) != len(timestamps) or len(frame_paths) > settings.maximum_frames:
        raise SourceSamplingError("source frame extraction produced inconsistent samples")
    if any(
        not Fraction(window.start_millis, 1000) <= timestamp < Fraction(window.end_millis, 1000)
        for timestamp in timestamps
    ) or any(later <= earlier for earlier, later in pairwise(timestamps)):
        raise SourceSamplingError("source frame extraction produced invalid timestamps")
    if any(not 0 < path.stat().st_size <= settings.maximum_frame_bytes for path in frame_paths):
        raise SourceSamplingError("source frame extraction exceeded its frame byte limit")
    return tuple(
        SampledSourceFrame(path, timestamp)
        for path, timestamp in zip(frame_paths, timestamps, strict=True)
    )


def sample_source_frames(
    source_path: Path,
    output_directory: Path,
    *,
    window: SourceWindow,
    settings: SourceFrameSampling = SourceFrameSampling(),
) -> SourceFrameSamples:
    """Extract bounded original frames into a new, caller-owned directory.

    The half-open window is relative to the source's playback start. Return
    an empty frame tuple when no eligible frames exist; this is not a model
    observation. A fallback shares the first attempt's deadline. Failures
    remove newly created output; an existing destination is never overwritten.
    The deadline starts after resolving HFlow's FFmpeg binary (which may
    download a managed build). No source file is modified or re-encoded.
    """
    if not isinstance(window, SourceWindow):
        raise ValueError("window must be a SourceWindow")
    if not isinstance(settings, SourceFrameSampling):
        raise ValueError("settings must be a SourceFrameSampling")
    if window.duration_millis > settings.maximum_window_millis:
        raise ValueError("window exceeds maximum_window_millis")
    try:
        finite_end = math.isfinite(window.end_millis / 1000)
    except OverflowError:
        finite_end = False
    if not finite_end:
        raise ValueError("window exceeds the supported playback timeline")
    if not source_path.is_file():
        raise FileNotFoundError("sampling source is not a local file")
    from hflow.ffmpeg import ffmpeg_path, ffprobe_path

    executable = ffmpeg_path()
    probe_executable = ffprobe_path()
    deadline = time.monotonic() + settings.timeout_seconds
    output_directory.mkdir(parents=True, exist_ok=False)
    actual_mode: Literal[SourceSamplingMode.UNIFORM, SourceSamplingMode.KEYFRAMES] = (
        SourceSamplingMode.UNIFORM
        if settings.mode is SourceSamplingMode.UNIFORM
        else SourceSamplingMode.KEYFRAMES
    )
    fallback_reason = None
    try:
        time_base = _source_time_base(source_path.resolve(), probe_executable, deadline=deadline)
        with tempfile.TemporaryDirectory(
            dir=output_directory, prefix="frames-"
        ) as temporary_directory:
            staging_directory = Path(temporary_directory)
            frames = _extract_frames(
                source_path.resolve(),
                staging_directory,
                window,
                settings,
                actual_mode,
                executable=executable,
                deadline=deadline,
                time_base=time_base,
            )
            if settings.mode is SourceSamplingMode.KEYFRAMES_FIRST:
                if len(frames) < 2:
                    fallback_reason = KeyframeFallbackReason.TOO_FEW_KEYFRAMES
                elif frames[-1].timestamp_seconds - frames[0].timestamp_seconds < Fraction(
                    window.duration_millis, 2000
                ):
                    fallback_reason = KeyframeFallbackReason.INSUFFICIENT_SPAN
                if fallback_reason is not None:
                    for frame in frames:
                        frame.path.unlink()
                    actual_mode = SourceSamplingMode.UNIFORM
                    frames = _extract_frames(
                        source_path.resolve(),
                        staging_directory,
                        window,
                        settings,
                        actual_mode,
                        executable=executable,
                        deadline=deadline,
                        time_base=time_base,
                    )
            published_frames = tuple(
                SampledSourceFrame(
                    frame.path.rename(output_directory / frame.path.name), frame.timestamp_seconds
                )
                for frame in frames
            )
        _remaining_seconds(deadline)
        return SourceFrameSamples(
            window, published_frames, settings.mode, actual_mode, fallback_reason
        )
    except BaseException:
        shutil.rmtree(output_directory)
        raise
