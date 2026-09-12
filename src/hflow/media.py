"""Bounded local video inspection and fixed-rate window preparation.

Unreadable and unsupported media are expected outcomes. Process, timeout, and
filesystem failures raise MediaToolError or OSError and must not be counted as
unreadable recordings. No source file is changed and no network input is read.
"""

import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from pathlib import Path

from hflow.ffmpeg import ffmpeg_path, ffprobe_path
from hflow.ffmpeg._process import MediaToolError, media_input_was_rejected, run_media_command


@dataclass(frozen=True)
class VideoLimits:
    maximum_frame_pixels: int = 33554432
    maximum_frames_per_second: float = 240.0
    minimum_duration_seconds: float = 0.001
    maximum_duration_seconds: float = 86400.0
    timeout_seconds: float = 600.0
    maximum_probe_bytes: int = 65536

    def __post_init__(self) -> None:
        for value in (self.maximum_frame_pixels, self.maximum_probe_bytes):
            if type(value) is not int or value <= 0:
                raise ValueError("video size limits must be positive integers")
        for value in (
            self.maximum_frames_per_second,
            self.minimum_duration_seconds,
            self.maximum_duration_seconds,
            self.timeout_seconds,
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("video limits must be finite and positive")
        if self.minimum_duration_seconds > self.maximum_duration_seconds:
            raise ValueError("video duration limits are reversed")


@dataclass(frozen=True)
class VideoProperties:
    width: int
    height: int
    frames_per_second: Decimal
    duration_seconds: Decimal

    def __post_init__(self) -> None:
        if (
            type(self.width) is not int
            or type(self.height) is not int
            or self.width <= 0
            or self.height <= 0
        ):
            raise ValueError("video dimensions must be positive integers")
        for value in (self.frames_per_second, self.duration_seconds):
            if not value.is_finite() or value <= 0:
                raise ValueError("video properties must be finite and positive")

    @property
    def duration_millis(self) -> int:
        return int((self.duration_seconds * 1000).to_integral_value(rounding=ROUND_CEILING))


@dataclass(frozen=True)
class UnreadableVideo:
    """The source does not contain a complete readable video stream."""


@dataclass(frozen=True)
class UnsupportedVideo:
    """The source has properties outside the caller's supported profile."""


VideoInspection = VideoProperties | UnreadableVideo | UnsupportedVideo


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError("duplicate probe field")
        document[key] = value
    return document


def _duration_from_stream(stream: dict[str, object], document: dict[str, object]) -> Decimal:
    duration = stream.get("duration")
    if duration is not None and duration != "N/A":
        if not isinstance(duration, str):
            raise ValueError("invalid duration")
        return Decimal(duration)
    tags = stream.get("tags", {})
    if not isinstance(tags, dict):
        raise ValueError("invalid duration tags")
    duration_tag = tags.get("DURATION")
    if duration_tag is not None:
        if not isinstance(duration_tag, str) or not re.fullmatch(
            r"[0-9]{2}:[0-5][0-9]:[0-5][0-9](?:\.[0-9]{1,9})?", duration_tag
        ):
            raise ValueError("invalid duration tag")
        hours, minutes, seconds = duration_tag.split(":")
        return Decimal(hours) * 3600 + Decimal(minutes) * 60 + Decimal(seconds)
    # Container duration is unambiguous only when there is a single stream.
    video_format = document.get("format")
    if (
        not isinstance(video_format, dict)
        or type(video_format.get("nb_streams")) is not int
        or video_format["nb_streams"] != 1
    ):
        raise ValueError("ambiguous duration")
    format_duration = video_format.get("duration")
    if not isinstance(format_duration, str):
        raise ValueError("invalid container duration")
    return Decimal(format_duration)


def probe_video(
    source: Path, *, limits: VideoLimits = VideoLimits(), executable: Path | None = None
) -> VideoInspection:
    """Parse video properties once, retaining exact decimal duration for planning."""
    source = source.resolve(strict=True)
    completed = run_media_command(
        [
            str(executable or ffprobe_path()),
            "-v",
            "error",
            "-protocol_whitelist",
            "file,pipe",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,duration:stream_tags=DURATION:format=duration,nb_streams",
            "-of",
            "json",
            str(source),
        ],
        timeout_seconds=limits.timeout_seconds,
        maximum_output_bytes=limits.maximum_probe_bytes,
    )
    if media_input_was_rejected(completed):
        return UnreadableVideo()
    try:
        document = json.loads(completed.stdout, object_pairs_hook=_unique_object)
        if not isinstance(document, dict):
            return UnreadableVideo()
        streams = document["streams"]
        if not isinstance(streams, list) or len(streams) != 1 or not isinstance(streams[0], dict):
            return UnreadableVideo()
        stream = streams[0]
        rate = stream["avg_frame_rate"]
        if not isinstance(rate, str):
            return UnreadableVideo()
        numerator, denominator = rate.split("/")
        properties = VideoProperties(
            stream["width"],
            stream["height"],
            Decimal(numerator) / Decimal(denominator),
            _duration_from_stream(stream, document),
        )
    except (KeyError, TypeError, ValueError, InvalidOperation, ZeroDivisionError):
        return UnreadableVideo()
    if (
        properties.width * properties.height > limits.maximum_frame_pixels
        or properties.frames_per_second > Decimal(str(limits.maximum_frames_per_second))
        or not Decimal(str(limits.minimum_duration_seconds))
        <= properties.duration_seconds
        <= Decimal(str(limits.maximum_duration_seconds))
    ):
        return UnsupportedVideo()
    return properties


@dataclass(frozen=True)
class VideoWindow:
    start_seconds: float
    duration_seconds: float
    frames_per_second: float

    def __post_init__(self) -> None:
        for value in (self.start_seconds, self.duration_seconds, self.frames_per_second):
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise ValueError("window values must be finite numbers")
        if self.start_seconds < 0 or self.duration_seconds <= 0 or self.frames_per_second <= 0:
            raise ValueError("invalid video window")


@dataclass(frozen=True)
class PreparedVideoWindow:
    path: Path
    properties: VideoProperties


def prepare_video_window(
    source: Path, output: Path, window: VideoWindow, *, limits: VideoLimits = VideoLimits()
) -> PreparedVideoWindow | UnreadableVideo | UnsupportedVideo:
    """Publish an H.264 fixed-rate window without replacing an existing file.

    Sampling uses FFmpeg's fps filter and millisecond seek precision. This is
    deliberately separate from the importer's covering-frame sampling contract.
    """
    if output.exists() or output.is_symlink():
        raise FileExistsError("video window output already exists")
    inspection = probe_video(source, limits=limits)
    if not isinstance(inspection, VideoProperties):
        return inspection
    if (
        window.frames_per_second > limits.maximum_frames_per_second
        or window.duration_seconds > limits.maximum_duration_seconds
    ):
        return UnsupportedVideo()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".video-window-") as directory:
        staged = Path(directory) / "window.mp4"
        completed = run_media_command(
            [
                str(ffmpeg_path()),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-protocol_whitelist",
                "file,pipe",
                "-ss",
                f"{window.start_seconds:.3f}",
                "-t",
                f"{window.duration_seconds:.3f}",
                "-i",
                str(source.resolve()),
                "-map",
                "0:v:0",
                "-vf",
                f"fps={window.frames_per_second}",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                str(staged),
            ],
            timeout_seconds=limits.timeout_seconds,
        )
        if media_input_was_rejected(completed):
            return UnreadableVideo()
        prepared_properties = probe_video(staged, limits=limits)
        if not isinstance(prepared_properties, VideoProperties):
            return prepared_properties
        if prepared_properties.duration_seconds < Decimal(
            str(min(0.5, window.duration_seconds * 0.5))
        ):
            return UnreadableVideo()
        os.link(staged, output)
    return PreparedVideoWindow(output, prepared_properties)


__all__ = [
    "MediaToolError",
    "PreparedVideoWindow",
    "UnreadableVideo",
    "UnsupportedVideo",
    "VideoInspection",
    "VideoLimits",
    "VideoProperties",
    "VideoWindow",
    "prepare_video_window",
    "probe_video",
]
