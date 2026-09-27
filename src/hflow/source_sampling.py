"""Bounded JPEG sampling from original video with measured presentation times."""

from __future__ import annotations

import json
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
from hflow.byte_range_source import LoopbackVideoSource, MediaInput, media_input
from hflow.ffmpeg._process import MediaCommandResult, MediaToolError, run_media_command
from hflow.source_windows import SourceWindow

SOURCE_FRAME_SAMPLING_VERSION = "source-frame-sampling-v1"
_TIMESTAMP_PATTERN = re.compile(
    rb"\[Parsed_showinfo_[^\]]+\].*\bn:\s*\d+\s+pts:\s*(-?\d+)\s+pts_time:"
)
_TIME_BASE_PATTERN = re.compile(rb"\[Parsed_showinfo_[^\]]+\].*config in time_base:\s*(\d+)/(\d+)")
# The first keyframe probe after a bin start reads this much source time, and
# doubles until it finds a keyframe or reaches the window end.
_KEYFRAME_PROBE_SECONDS = Fraction(1, 2)


class SourceSamplingMode(StrEnum):
    UNIFORM = "uniform"
    KEYFRAMES = "keyframes"
    KEYFRAMES_FIRST = "keyframes_first"
    NEAREST_KEYFRAMES = "nearest_keyframes"


class SourceFrameResize(StrEnum):
    PAD = "pad"
    FIT = "fit"


class KeyframeFallbackReason(StrEnum):
    TOO_FEW_KEYFRAMES = "fewer_than_two_keyframes"
    INSUFFICIENT_SPAN = "keyframe_span_below_half_window"


@dataclass(frozen=True)
class SourceFrameSampling:
    """Selection and resource limits shared by every extraction attempt.

    Uniform sampling uses the first frame in each temporal bin, whose length
    is the greater of ``minimum_interval_millis`` and window length divided
    by ``maximum_frames``. Keyframe sampling uses the first keyframe in each of
    ``maximum_frames`` equal bins, found by seeking to each bin start, so it
    reads only the source bytes near the selected keyframes. Keyframes-first
    falls back to uniform when fewer than two keyframes were selected or their
    span is less than half the window; uniform reads the whole window.

    All sizes are output bounds; callers own source download/decoder memory
    limits. The default canvas preserves aspect ratio with black padding; FIT omits
    padding. NEAREST_KEYFRAMES selects unique keyframes closest to the configured
    relative positions, breaking ties toward the earlier frame without fallback. Increasing
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
    maximum_probe_bytes: int = 8 * 1024 * 1024
    keyframe_positions: tuple[float, ...] = ()
    resize: SourceFrameResize = SourceFrameResize.PAD
    scaling_algorithm: Literal["lanczos", "bicubic"] = "lanczos"
    jpeg_quality: int = 5

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
            "maximum_probe_bytes",
        ):
            require_positive_int(getattr(self, field_name), field_name)
        require_positive_float(self.timeout_seconds, "timeout_seconds")
        if not isinstance(self.resize, SourceFrameResize):
            raise ValueError("resize must be a SourceFrameResize")
        if self.scaling_algorithm not in ("lanczos", "bicubic"):
            raise ValueError("unsupported scaling_algorithm")
        if type(self.jpeg_quality) is not int or not 1 <= self.jpeg_quality <= 31:
            raise ValueError("jpeg_quality must be an integer between 1 and 31")
        if not isinstance(self.keyframe_positions, tuple) or any(
            isinstance(position, bool)
            or not isinstance(position, int | float)
            or not math.isfinite(position)
            or not 0 <= position <= 1
            for position in self.keyframe_positions
        ):
            raise ValueError("keyframe_positions must be finite relative positions between 0 and 1")
        if self.mode is SourceSamplingMode.NEAREST_KEYFRAMES:
            if not 0 < len(self.keyframe_positions) <= self.maximum_frames:
                raise ValueError("nearest keyframes require bounded keyframe_positions")
        elif self.keyframe_positions:
            raise ValueError("keyframe_positions require nearest-keyframe sampling")
        if self.width % 2 or self.height % 2:
            raise ValueError("width and height must be even for JPEG chroma subsampling")


@dataclass(frozen=True)
class SampledSourceFrame:
    """A JPEG and its actual presentation time from the source playback origin.

    The origin is the container start time, not the requested window start,
    taken as FFmpeg rescales it onto the video time base: the nearest tick,
    ties away from zero. Every sampling mode shares that origin.
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
    actual_mode: Literal[
        SourceSamplingMode.UNIFORM,
        SourceSamplingMode.KEYFRAMES,
        SourceSamplingMode.NEAREST_KEYFRAMES,
    ]
    fallback_reason: KeyframeFallbackReason | None


class SourceSamplingError(RuntimeError):
    """Extraction failed, exceeded a resource limit, or could not prove frame timing."""


def _remaining_seconds(deadline: float) -> float:
    remaining_seconds = deadline - time.monotonic()
    if remaining_seconds <= 0:
        raise SourceSamplingError("source frame extraction exceeded its time limit")
    return remaining_seconds


def _run_sampling_command(
    arguments: list[str], *, deadline: float, maximum_log_bytes: int, source_input: MediaInput
) -> MediaCommandResult:
    try:
        completed = run_media_command(
            arguments,
            timeout_seconds=_remaining_seconds(deadline),
            maximum_output_bytes=maximum_log_bytes,
            environment=source_input.environment,
        )
    except MediaToolError as error:
        source_input.raise_for_reader_failure()
        raise SourceSamplingError(str(error)) from None
    source_input.raise_for_reader_failure()
    if completed.returncode != 0:
        raise SourceSamplingError("source frame extraction failed")
    _remaining_seconds(deadline)
    return completed


def _source_time_base(source_input: MediaInput, executable: Path, *, deadline: float) -> Fraction:
    completed = _run_sampling_command(
        [
            str(executable),
            "-v",
            "error",
            "-protocol_whitelist",
            source_input.protocols,
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=time_base",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            source_input.location,
        ],
        deadline=deadline,
        maximum_log_bytes=65_536,
        source_input=source_input,
    )
    try:
        # MPEG-TS may repeat the selected stream inside a program section.
        # Repeated identical values still describe one unambiguous time base.
        time_bases = {
            Fraction(value) for value in completed.stdout.decode("ascii").splitlines() if value
        }
        if len(time_bases) != 1:
            raise ValueError("ambiguous time base")
        time_base = time_bases.pop()
    except (ValueError, ZeroDivisionError):
        raise SourceSamplingError("source video has no valid presentation time base") from None
    if time_base <= 0:
        raise SourceSamplingError("source video has no valid presentation time base")
    return time_base


def _aligned_playback_origin(
    source_input: MediaInput, executable: Path, *, deadline: float, time_base: Fraction
) -> Fraction:
    origin_result = _run_sampling_command(
        [
            str(executable),
            "-v",
            "error",
            "-protocol_whitelist",
            source_input.protocols,
            "-show_entries",
            "format=start_time",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            source_input.location,
        ],
        deadline=deadline,
        maximum_log_bytes=65_536,
        source_input=source_input,
    )
    try:
        origin = Fraction(origin_result.stdout.decode("ascii").strip())
    except (ValueError, ZeroDivisionError) as error:
        raise SourceSamplingError("source video has no valid playback origin") from error
    # Container start times use microseconds and need not align to video ticks.
    # Match FFmpeg's timestamp offset rescaling: nearest, with ties away from zero.
    origin_ticks = origin / time_base
    rounded_origin_ticks = (
        math.floor(origin_ticks + Fraction(1, 2))
        if origin_ticks >= 0
        else math.ceil(origin_ticks - Fraction(1, 2))
    )
    return rounded_origin_ticks * time_base


def _keyframe_times_in_interval(
    source_input: MediaInput,
    read_interval: str,
    *,
    executable: Path,
    deadline: float,
    time_base: Fraction,
    aligned_origin: Fraction,
    maximum_probe_bytes: int,
) -> set[Fraction]:
    """Keyframe times from the playback origin, for packets ffprobe reads in ``read_interval``."""
    packet_result = _run_sampling_command(
        [
            str(executable),
            "-v",
            "error",
            "-protocol_whitelist",
            source_input.protocols,
            "-select_streams",
            "v:0",
            "-read_intervals",
            read_interval,
            "-show_packets",
            "-show_entries",
            "packet=pts,flags",
            "-of",
            "json",
            source_input.location,
        ],
        deadline=deadline,
        maximum_log_bytes=maximum_probe_bytes,
        source_input=source_input,
    )
    try:
        document = json.loads(packet_result.stdout)
        if not isinstance(document, dict) or not isinstance(document.get("packets"), list):
            raise ValueError("invalid packet document")
        keyframe_times: set[Fraction] = set()
        for packet in document["packets"]:
            if not isinstance(packet, dict) or not isinstance(packet.get("flags"), str):
                raise ValueError("invalid packet")
            if "K" not in packet["flags"]:
                continue
            if type(packet.get("pts")) is not int:
                raise ValueError("keyframe has no presentation timestamp")
            keyframe_times.add(packet["pts"] * time_base - aligned_origin)
    except (ValueError, TypeError, KeyError) as error:
        raise SourceSamplingError("source keyframes have invalid presentation times") from error
    return keyframe_times


def _nearest_keyframe_times(
    source_input: MediaInput,
    window: SourceWindow,
    settings: SourceFrameSampling,
    *,
    executable: Path,
    deadline: float,
    time_base: Fraction,
) -> tuple[Fraction, ...]:
    aligned_origin = _aligned_playback_origin(
        source_input, executable, deadline=deadline, time_base=time_base
    )
    start = Fraction(window.start_millis, 1000)
    end = Fraction(window.end_millis, 1000)
    available_times = {
        timestamp
        for timestamp in _keyframe_times_in_interval(
            source_input,
            f"%{float(aligned_origin + end):.9f}",
            executable=executable,
            deadline=deadline,
            time_base=time_base,
            aligned_origin=aligned_origin,
            maximum_probe_bytes=settings.maximum_probe_bytes,
        )
        if start <= timestamp < end
    }
    if not available_times:
        return ()
    return tuple(
        sorted(
            {
                min(
                    available_times,
                    key=lambda timestamp: (
                        abs(timestamp - (start + (end - start) * Fraction(str(position)))),
                        timestamp,
                    ),
                )
                for position in settings.keyframe_positions
            }
        )
    )


def _extraction_arguments(
    source_input: MediaInput,
    output_directory: Path,
    settings: SourceFrameSampling,
    *,
    executable: Path,
    video_filter: str,
    seek_seconds: str,
    duration_seconds: str,
    maximum_frames: int,
    skip_non_keyframes: bool,
    first_frame_number: int = 1,
) -> list[str]:
    arguments = [
        str(executable),
        "-hide_banner",
        "-loglevel",
        "info",
        "-nostats",
        "-nostdin",
        "-xerror",
        "-protocol_whitelist",
        source_input.protocols,
        "-threads",
        "2",
    ]
    if skip_non_keyframes:
        arguments.extend(("-skip_frame", "nokey"))
    # Preserve the original playback PTS through a seek. Adding a seek offset
    # back to rebased PTS would introduce rounding at non-tick-aligned starts.
    # Accurate seeking discards preceding-GOP frames before the input duration
    # limit is applied. Disabling it can exhaust a short window before its start.
    arguments.extend(
        (
            "-copyts",
            "-start_at_zero",
            "-ss",
            seek_seconds,
            "-t",
            duration_seconds,
            "-i",
            source_input.location,
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
            str(maximum_frames),
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "yuvj420p",
            "-c:v",
            "mjpeg",
            "-q:v",
            str(settings.jpeg_quality),
            "-threads",
            "2",
            "-f",
            "image2",
            "-start_number",
            str(first_frame_number),
            str(output_directory / "frame_%06d.jpg"),
        )
    )
    return arguments


def _resize_filters(settings: SourceFrameSampling) -> list[str]:
    resize_filters = [
        f"scale={settings.width}:{settings.height}:force_original_aspect_ratio=decrease:flags={settings.scaling_algorithm}",
    ]
    if settings.resize is SourceFrameResize.PAD:
        resize_filters.append(f"pad={settings.width}:{settings.height}:(ow-iw)/2:(oh-ih)/2:black")
    return resize_filters


def _run_extraction(
    arguments: list[str],
    settings: SourceFrameSampling,
    *,
    deadline: float,
    time_base: Fraction,
    source_input: MediaInput,
) -> tuple[Fraction, ...]:
    """Run one extraction and return the presentation times FFmpeg reported."""
    diagnostics = _run_sampling_command(
        arguments,
        deadline=deadline,
        maximum_log_bytes=settings.maximum_log_bytes,
        source_input=source_input,
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
    return tuple(
        int(match.group(1)) * Fraction(numerator, denominator)
        for match in _TIMESTAMP_PATTERN.finditer(diagnostics)
    )


def _validated_frames(
    output_directory: Path,
    timestamps: tuple[Fraction, ...],
    window: SourceWindow,
    settings: SourceFrameSampling,
) -> tuple[SampledSourceFrame, ...]:
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


def _extract_frames(
    source_input: MediaInput,
    output_directory: Path,
    window: SourceWindow,
    settings: SourceFrameSampling,
    mode: Literal[SourceSamplingMode.UNIFORM, SourceSamplingMode.NEAREST_KEYFRAMES],
    *,
    executable: Path,
    deadline: float,
    time_base: Fraction,
    probe_executable: Path,
) -> tuple[SampledSourceFrame, ...]:
    """Extract uniform or nearest-keyframe samples in one pass over the window."""
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
    selected_times: tuple[Fraction, ...] = ()
    if mode is SourceSamplingMode.NEAREST_KEYFRAMES:
        selected_times = _nearest_keyframe_times(
            source_input,
            window,
            settings,
            executable=probe_executable,
            deadline=deadline,
            time_base=time_base,
        )
        if not selected_times:
            return ()
        selection_filter = "select=" + "+".join(
            f"eq(pts\\,{int(timestamp / time_base)})" for timestamp in selected_times
        )
    video_filter = ",".join((selection_filter, *_resize_filters(settings), "showinfo"))
    timestamps = _run_extraction(
        _extraction_arguments(
            source_input,
            output_directory,
            settings,
            executable=executable,
            video_filter=video_filter,
            seek_seconds=f"{start_seconds:.3f}",
            duration_seconds=f"{duration_seconds:.3f}",
            maximum_frames=settings.maximum_frames,
            skip_non_keyframes=mode is SourceSamplingMode.NEAREST_KEYFRAMES,
        ),
        settings,
        deadline=deadline,
        time_base=time_base,
        source_input=source_input,
    )
    if mode is SourceSamplingMode.NEAREST_KEYFRAMES and timestamps != selected_times:
        raise SourceSamplingError("extraction did not reproduce the selected keyframes")
    return _validated_frames(output_directory, timestamps, window, settings)


def _first_keyframe_in_each_bin(
    source_input: MediaInput,
    window: SourceWindow,
    settings: SourceFrameSampling,
    *,
    executable: Path,
    deadline: float,
    time_base: Fraction,
    aligned_origin: Fraction,
) -> tuple[Fraction, ...]:
    """Select the first keyframe of each of ``maximum_frames`` equal bins, skipping empty bins.

    Each bin is probed from its start, so ffprobe reads packets from the
    keyframe before the bin start to the first keyframe inside it rather
    than the whole window.
    """
    start = Fraction(window.start_millis, 1000)
    end = Fraction(window.end_millis, 1000)
    bin_seconds = Fraction(window.duration_millis, 1000 * settings.maximum_frames)
    selected: list[Fraction] = []
    bin_index = 0
    while bin_index < settings.maximum_frames:
        bin_start = start + bin_index * bin_seconds
        probe_seconds = _KEYFRAME_PROBE_SECONDS
        while True:
            keyframe_times = [
                timestamp
                for timestamp in _keyframe_times_in_interval(
                    source_input,
                    # An absolute end: an offset end would count from the keyframe the
                    # seek lands on, before the bin start, and stop short of the probe.
                    f"{float(aligned_origin + bin_start):.9f}"
                    f"%{float(aligned_origin + bin_start + probe_seconds):.9f}",
                    executable=executable,
                    deadline=deadline,
                    time_base=time_base,
                    aligned_origin=aligned_origin,
                    maximum_probe_bytes=settings.maximum_probe_bytes,
                )
                if bin_start <= timestamp < end
            ]
            if keyframe_times or bin_start + probe_seconds >= end:
                break
            probe_seconds *= 2
        if not keyframe_times:
            break
        first_keyframe_time = min(keyframe_times)
        selected.append(first_keyframe_time)
        bin_index = math.floor((first_keyframe_time - start) / bin_seconds) + 1
    return tuple(selected)


def _extract_keyframes_by_seeking(
    source_input: MediaInput,
    output_directory: Path,
    window: SourceWindow,
    settings: SourceFrameSampling,
    *,
    executable: Path,
    deadline: float,
    time_base: Fraction,
    probe_executable: Path,
) -> tuple[SampledSourceFrame, ...]:
    """Extract the first keyframe of each bin, seeking to each one.

    The decoder is not told to skip non-keyframes: with nothing to output it
    reads far past the keyframe before stopping. Selection by exact PTS keeps
    any frame decoded after the keyframe out of the result.
    """
    aligned_origin = _aligned_playback_origin(
        source_input, probe_executable, deadline=deadline, time_base=time_base
    )
    selected_times = _first_keyframe_in_each_bin(
        source_input,
        window,
        settings,
        executable=probe_executable,
        deadline=deadline,
        time_base=time_base,
        aligned_origin=aligned_origin,
    )
    end = Fraction(window.end_millis, 1000)
    timestamps: list[Fraction] = []
    for frame_number, keyframe_time in enumerate(selected_times, start=1):
        # Seek to the keyframe, floored to FFmpeg's microsecond precision.
        seek = Fraction(math.floor(keyframe_time * 1_000_000), 1_000_000)
        extracted = _run_extraction(
            _extraction_arguments(
                source_input,
                output_directory,
                settings,
                executable=executable,
                video_filter=",".join(
                    (
                        f"select=eq(pts\\,{int(keyframe_time / time_base)})",
                        *_resize_filters(settings),
                        "showinfo",
                    )
                ),
                seek_seconds=f"{float(seek):.6f}",
                duration_seconds=f"{math.ceil((end - seek) * 1_000_000) / 1_000_000:.6f}",
                maximum_frames=1,
                skip_non_keyframes=False,
                first_frame_number=frame_number,
            ),
            settings,
            deadline=deadline,
            time_base=time_base,
            source_input=source_input,
        )
        if extracted != (keyframe_time,):
            raise SourceSamplingError("extraction did not reproduce the selected keyframes")
        timestamps.append(keyframe_time)
    return _validated_frames(output_directory, tuple(timestamps), window, settings)


def sample_source_frames(
    source_path: Path | LoopbackVideoSource,
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
    ``source_path`` may be a :class:`~hflow.byte_range_source.LoopbackVideoSource`;
    its reader failures are re-raised as the reader's own exception.
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
    if isinstance(source_path, Path) and not source_path.is_file():
        raise FileNotFoundError("sampling source is not a local file")
    source_input = media_input(source_path, local_protocols="file")
    from hflow.ffmpeg import ffmpeg_path, ffprobe_path

    executable = ffmpeg_path()
    probe_executable = ffprobe_path()
    deadline = time.monotonic() + settings.timeout_seconds
    output_directory.mkdir(parents=True, exist_ok=False)
    actual_mode: Literal[
        SourceSamplingMode.UNIFORM,
        SourceSamplingMode.KEYFRAMES,
        SourceSamplingMode.NEAREST_KEYFRAMES,
    ] = (
        SourceSamplingMode.UNIFORM
        if settings.mode is SourceSamplingMode.UNIFORM
        else SourceSamplingMode.NEAREST_KEYFRAMES
        if settings.mode is SourceSamplingMode.NEAREST_KEYFRAMES
        else SourceSamplingMode.KEYFRAMES
    )
    fallback_reason = None
    try:
        time_base = _source_time_base(source_input, probe_executable, deadline=deadline)
        with tempfile.TemporaryDirectory(
            dir=output_directory, prefix="frames-"
        ) as temporary_directory:
            staging_directory = Path(temporary_directory)
            if actual_mode is SourceSamplingMode.KEYFRAMES:
                frames = _extract_keyframes_by_seeking(
                    source_input,
                    staging_directory,
                    window,
                    settings,
                    executable=executable,
                    deadline=deadline,
                    time_base=time_base,
                    probe_executable=probe_executable,
                )
            else:
                frames = _extract_frames(
                    source_input,
                    staging_directory,
                    window,
                    settings,
                    actual_mode,
                    executable=executable,
                    deadline=deadline,
                    time_base=time_base,
                    probe_executable=probe_executable,
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
                        source_input,
                        staging_directory,
                        window,
                        settings,
                        actual_mode,
                        executable=executable,
                        deadline=deadline,
                        time_base=time_base,
                        probe_executable=probe_executable,
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
