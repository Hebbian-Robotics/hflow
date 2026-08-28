"""Strict, single-pass frame statistics for one video file."""

import math
import re
import statistics
import subprocess
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cache
from io import StringIO
from pathlib import Path
from typing import Protocol

from ._field_guards import require_float, require_int
from ._toolchain import VideoMeasurementToolchain

FRAME_STATISTICS_DEFINITION_VERSION = "video-frame-statistics/v1"


class _TextWriter(Protocol):
    def write(self, text: str, /) -> int: ...


class FrameStatisticsParseError(RuntimeError):
    """FFmpeg emitted incomplete or malformed per-frame metadata."""


class FrameStatisticsExecutionError(RuntimeError):
    """FFmpeg could not complete the frame-statistics measurement."""


class _NoInstrumentFramesError(FrameStatisticsParseError):
    """The stream ended cleanly before any frame metadata appeared."""


class UnsupportedVideoMeasurementToolchainError(RuntimeError):
    """The supplied FFmpeg build lacks a filter required by the measurement."""


class LumaRangeEvidence(StrEnum):
    """What decoded luma samples show about the nominal limited range."""

    NOMINAL_LIMITED_RANGE_COMPATIBLE = "nominal_limited_range_compatible"
    EXTENDS_BEYOND_NOMINAL_LIMITED_RANGE = "extends_beyond_nominal_limited_range"


@dataclass(frozen=True)
class VideoTimeInterval:
    """A closed-open interval on a video's seconds-from-start clock."""

    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        require_float(self.start_seconds, "start_seconds")
        require_float(self.end_seconds, "end_seconds")
        if not math.isfinite(self.start_seconds) or self.start_seconds < 0:
            raise ValueError(f"start_seconds must be finite and nonnegative: {self.start_seconds}")
        if not math.isfinite(self.end_seconds) or self.end_seconds < self.start_seconds:
            raise ValueError(
                "end_seconds must be finite and no earlier than start_seconds: "
                f"{self.end_seconds} < {self.start_seconds}"
            )


@dataclass(frozen=True)
class FrameStatisticsSettings:
    """The caller-owned thresholds that define the frame statistics."""

    black_frame_minimum_pixel_share_percent: int = 98
    black_pixel_luma_threshold: int = 17
    freeze_noise_tolerance_decibels: float = -60.0
    freeze_minimum_duration_seconds: float = 2.0
    overexposed_average_luma_threshold: float = 235.0

    def __post_init__(self) -> None:
        require_int(
            self.black_frame_minimum_pixel_share_percent,
            "black_frame_minimum_pixel_share_percent",
        )
        require_int(self.black_pixel_luma_threshold, "black_pixel_luma_threshold")
        require_float(self.freeze_noise_tolerance_decibels, "freeze_noise_tolerance_decibels")
        require_float(self.freeze_minimum_duration_seconds, "freeze_minimum_duration_seconds")
        require_float(self.overexposed_average_luma_threshold, "overexposed_average_luma_threshold")
        if not 0 <= self.black_frame_minimum_pixel_share_percent <= 100:
            raise ValueError("black_frame_minimum_pixel_share_percent must be between 0 and 100")
        if not 0 <= self.black_pixel_luma_threshold <= 255:
            raise ValueError("black_pixel_luma_threshold must be between 0 and 255")
        if not math.isfinite(self.freeze_noise_tolerance_decibels):
            raise ValueError("freeze_noise_tolerance_decibels must be finite")
        if (
            not math.isfinite(self.freeze_minimum_duration_seconds)
            or self.freeze_minimum_duration_seconds <= 0
        ):
            raise ValueError("freeze_minimum_duration_seconds must be finite and positive")
        if (
            not math.isfinite(self.overexposed_average_luma_threshold)
            or not 0 <= self.overexposed_average_luma_threshold <= 255
        ):
            raise ValueError("overexposed_average_luma_threshold must be between 0 and 255")


@dataclass(frozen=True)
class FrameStatisticsProvenance:
    """The definition, settings, and binary that produced a result."""

    measurement_definition_version: str
    ffmpeg_version: str
    filter_graph: str
    settings: FrameStatisticsSettings


@dataclass(frozen=True)
class VideoFrameStatistics:
    """Format-independent measurements aggregated over decoded video frames."""

    decoded_frame_count: int
    duration_seconds: float
    black_frame_count: int
    black_frame_percent: float
    overexposed_frame_count: int
    overexposed_frame_percent: float
    freeze_intervals: tuple[VideoTimeInterval, ...]
    freeze_total_seconds: float
    average_luma_mean: float
    average_luma_minimum: float
    average_luma_maximum: float
    black_pixel_share_mean: float
    black_pixel_share_maximum: float
    minimum_luma: float
    maximum_luma: float
    luma_range_evidence: LumaRangeEvidence
    tenth_percentile_luma_mean: float
    ninetieth_percentile_luma_mean: float
    clipped_highlight_frame_percent: float
    crushed_shadow_frame_percent: float
    frame_difference_mean: float
    frame_difference_maximum: float
    temporal_outlier_mean: float
    temporal_outlier_maximum: float
    out_of_legal_range_mean: float
    out_of_legal_range_maximum: float
    provenance: FrameStatisticsProvenance


@dataclass(frozen=True)
class _AggregatedFrameStatistics:
    decoded_frame_count: int
    duration_seconds: float
    black_frame_count: int
    black_frame_percent: float
    overexposed_frame_count: int
    overexposed_frame_percent: float
    freeze_intervals: tuple[VideoTimeInterval, ...]
    freeze_total_seconds: float
    average_luma_mean: float
    average_luma_minimum: float
    average_luma_maximum: float
    black_pixel_share_mean: float
    black_pixel_share_maximum: float
    minimum_luma: float
    maximum_luma: float
    luma_range_evidence: LumaRangeEvidence
    tenth_percentile_luma_mean: float
    ninetieth_percentile_luma_mean: float
    clipped_highlight_frame_percent: float
    crushed_shadow_frame_percent: float
    frame_difference_mean: float
    frame_difference_maximum: float
    temporal_outlier_mean: float
    temporal_outlier_maximum: float
    out_of_legal_range_mean: float
    out_of_legal_range_maximum: float


LIMITED_RANGE_LUMA_FLOOR = 16
LIMITED_RANGE_LUMA_CEILING = 235
_LIMITED_RANGE_CLIPPED_HIGHLIGHT_GATE = 246.0
_EXTENDED_RANGE_CLIPPED_HIGHLIGHT_GATE = 254.0
_LIMITED_RANGE_CRUSHED_SHADOW_GATE = 5.0
_EXTENDED_RANGE_CRUSHED_SHADOW_GATE = 1.0
_MAXIMUM_EIGHT_BIT_LUMA = 255.0

_FRAME_HEADER_PATTERN = re.compile(
    r"^frame:(?P<frame_index>\d+)\s+pts:(?P<pts>-?\d+)\s+pts_time:(?P<pts_time>-?[0-9.eE+-]+)\s*$"
)
_METADATA_LINE_PATTERN = re.compile(r"^(?P<key>lavfi\.[A-Za-z0-9_.]+)=(?P<value>.*)$")
_FILTER_LIST_ENTRY_PATTERN = re.compile(r"^\s*[TSC.]+\s+(?P<filter_name>[A-Za-z0-9_]+)\s+")

_SIGNALSTATS_YAVG_KEY = "lavfi.signalstats.YAVG"
_BLACKFRAME_PBLACK_KEY = "lavfi.blackframe.pblack"
_FREEZE_START_KEY = "lavfi.freezedetect.freeze_start"
_FREEZE_END_KEY = "lavfi.freezedetect.freeze_end"
_SIGNALSTATS_YMIN_KEY = "lavfi.signalstats.YMIN"
_SIGNALSTATS_YLOW_KEY = "lavfi.signalstats.YLOW"
_SIGNALSTATS_YHIGH_KEY = "lavfi.signalstats.YHIGH"
_SIGNALSTATS_YMAX_KEY = "lavfi.signalstats.YMAX"
_SIGNALSTATS_YDIF_KEY = "lavfi.signalstats.YDIF"
_SIGNALSTATS_TOUT_KEY = "lavfi.signalstats.TOUT"
_SIGNALSTATS_BRNG_KEY = "lavfi.signalstats.BRNG"


@dataclass(frozen=True)
class _ParsedFrame:
    presentation_time_seconds: float
    metadata: dict[str, str]


@dataclass
class _RunningSummary:
    observed_value_count: int = 0
    value_total: float = 0.0
    minimum_value: float = math.inf
    maximum_value: float = -math.inf

    def add(self, value: float) -> None:
        self.observed_value_count += 1
        self.value_total += value
        self.minimum_value = min(self.minimum_value, value)
        self.maximum_value = max(self.maximum_value, value)

    def mean(self) -> float:
        if self.observed_value_count == 0:
            return 0.0
        return self.value_total / self.observed_value_count

    def minimum(self) -> float:
        return self.minimum_value if self.observed_value_count else 0.0

    def maximum(self) -> float:
        return self.maximum_value if self.observed_value_count else 0.0


@dataclass
class _FrameStatisticsAccumulator:
    settings: FrameStatisticsSettings
    decoded_frame_count: int = 0
    previous_presentation_time_seconds: float | None = None
    last_presentation_time_seconds: float = 0.0
    frame_intervals_seconds: list[float] = field(default_factory=list)
    black_frame_count: int = 0
    overexposed_frame_count: int = 0
    limited_range_clipped_highlight_count: int = 0
    extended_range_clipped_highlight_count: int = 0
    limited_range_crushed_shadow_count: int = 0
    extended_range_crushed_shadow_count: int = 0
    open_freeze_start_seconds: float | None = None
    freeze_intervals: list[VideoTimeInterval] = field(default_factory=list)
    average_luma: _RunningSummary = field(default_factory=_RunningSummary)
    black_pixel_share: _RunningSummary = field(default_factory=_RunningSummary)
    minimum_luma: _RunningSummary = field(default_factory=_RunningSummary)
    maximum_luma: _RunningSummary = field(default_factory=_RunningSummary)
    tenth_percentile_luma: _RunningSummary = field(default_factory=_RunningSummary)
    ninetieth_percentile_luma: _RunningSummary = field(default_factory=_RunningSummary)
    frame_difference: _RunningSummary = field(default_factory=_RunningSummary)
    temporal_outlier: _RunningSummary = field(default_factory=_RunningSummary)
    out_of_legal_range: _RunningSummary = field(default_factory=_RunningSummary)

    def add_frame(self, frame: _ParsedFrame) -> None:
        frame_index = self.decoded_frame_count
        average_luma = _required_luma_value(frame, frame_index, _SIGNALSTATS_YAVG_KEY)
        black_pixel_share = _required_finite_value(frame, frame_index, _BLACKFRAME_PBLACK_KEY)
        minimum_luma = _required_luma_value(frame, frame_index, _SIGNALSTATS_YMIN_KEY)
        tenth_percentile_luma = _required_luma_value(frame, frame_index, _SIGNALSTATS_YLOW_KEY)
        ninetieth_percentile_luma = _required_luma_value(frame, frame_index, _SIGNALSTATS_YHIGH_KEY)
        maximum_luma = _required_luma_value(frame, frame_index, _SIGNALSTATS_YMAX_KEY)
        frame_difference = _required_luma_value(frame, frame_index, _SIGNALSTATS_YDIF_KEY)
        temporal_outlier = _required_finite_value(frame, frame_index, _SIGNALSTATS_TOUT_KEY)
        out_of_legal_range = _required_finite_value(frame, frame_index, _SIGNALSTATS_BRNG_KEY)

        if self.previous_presentation_time_seconds is not None:
            self.frame_intervals_seconds.append(
                frame.presentation_time_seconds - self.previous_presentation_time_seconds
            )
            self.frame_difference.add(frame_difference)
        self.previous_presentation_time_seconds = frame.presentation_time_seconds
        self.last_presentation_time_seconds = frame.presentation_time_seconds

        self.average_luma.add(average_luma)
        self.black_pixel_share.add(black_pixel_share)
        self.minimum_luma.add(minimum_luma)
        self.maximum_luma.add(maximum_luma)
        self.tenth_percentile_luma.add(tenth_percentile_luma)
        self.ninetieth_percentile_luma.add(ninetieth_percentile_luma)
        self.temporal_outlier.add(temporal_outlier)
        self.out_of_legal_range.add(out_of_legal_range)

        self.black_frame_count += int(
            black_pixel_share >= self.settings.black_frame_minimum_pixel_share_percent
        )
        self.overexposed_frame_count += int(
            average_luma >= self.settings.overexposed_average_luma_threshold
        )
        self.limited_range_clipped_highlight_count += int(
            ninetieth_percentile_luma > _LIMITED_RANGE_CLIPPED_HIGHLIGHT_GATE
        )
        self.extended_range_clipped_highlight_count += int(
            ninetieth_percentile_luma > _EXTENDED_RANGE_CLIPPED_HIGHLIGHT_GATE
        )
        self.limited_range_crushed_shadow_count += int(
            tenth_percentile_luma < _LIMITED_RANGE_CRUSHED_SHADOW_GATE
        )
        self.extended_range_crushed_shadow_count += int(
            tenth_percentile_luma < _EXTENDED_RANGE_CRUSHED_SHADOW_GATE
        )
        self._add_freeze_metadata(frame, frame_index)
        self.decoded_frame_count += 1

    def _add_freeze_metadata(self, frame: _ParsedFrame, frame_index: int) -> None:
        freeze_start_text = frame.metadata.get(_FREEZE_START_KEY)
        if freeze_start_text is not None:
            if self.open_freeze_start_seconds is not None:
                raise FrameStatisticsParseError(
                    f"frame {frame_index}: freeze_start while a freeze is already open"
                )
            self.open_freeze_start_seconds = _parse_finite_float(
                freeze_start_text, f"freeze_start of frame {frame_index}"
            )
        freeze_end_text = frame.metadata.get(_FREEZE_END_KEY)
        if freeze_end_text is not None:
            if self.open_freeze_start_seconds is None:
                raise FrameStatisticsParseError(
                    f"frame {frame_index}: freeze_end without a matching freeze_start"
                )
            freeze_end_seconds = _parse_finite_float(
                freeze_end_text, f"freeze_end of frame {frame_index}"
            )
            self.freeze_intervals.append(
                _validated_freeze_interval(
                    self.open_freeze_start_seconds,
                    freeze_end_seconds,
                    context=f"frame {frame_index}",
                )
            )
            self.open_freeze_start_seconds = None

    def finish(self) -> _AggregatedFrameStatistics:
        if self.decoded_frame_count == 0:
            raise _NoInstrumentFramesError("instrument produced no frames")
        median_frame_interval_seconds = (
            statistics.median(self.frame_intervals_seconds) if self.frame_intervals_seconds else 0.0
        )
        duration_seconds = self.last_presentation_time_seconds + median_frame_interval_seconds
        if self.open_freeze_start_seconds is not None:
            self.freeze_intervals.append(
                _validated_freeze_interval(
                    self.open_freeze_start_seconds,
                    duration_seconds,
                    context="unterminated freeze at end of video",
                )
            )

        extends_beyond_limited_range = (
            self.minimum_luma.minimum() < LIMITED_RANGE_LUMA_FLOOR
            or self.maximum_luma.maximum() > LIMITED_RANGE_LUMA_CEILING
        )
        luma_range_evidence = (
            LumaRangeEvidence.EXTENDS_BEYOND_NOMINAL_LIMITED_RANGE
            if extends_beyond_limited_range
            else LumaRangeEvidence.NOMINAL_LIMITED_RANGE_COMPATIBLE
        )
        clipped_highlight_count = (
            self.extended_range_clipped_highlight_count
            if extends_beyond_limited_range
            else self.limited_range_clipped_highlight_count
        )
        crushed_shadow_count = (
            self.extended_range_crushed_shadow_count
            if extends_beyond_limited_range
            else self.limited_range_crushed_shadow_count
        )
        frame_count = self.decoded_frame_count
        return _AggregatedFrameStatistics(
            decoded_frame_count=frame_count,
            duration_seconds=duration_seconds,
            black_frame_count=self.black_frame_count,
            black_frame_percent=100.0 * self.black_frame_count / frame_count,
            overexposed_frame_count=self.overexposed_frame_count,
            overexposed_frame_percent=100.0 * self.overexposed_frame_count / frame_count,
            freeze_intervals=tuple(self.freeze_intervals),
            freeze_total_seconds=sum(
                (
                    interval.end_seconds - interval.start_seconds
                    for interval in self.freeze_intervals
                ),
                0.0,
            ),
            average_luma_mean=self.average_luma.mean(),
            average_luma_minimum=self.average_luma.minimum(),
            average_luma_maximum=self.average_luma.maximum(),
            black_pixel_share_mean=self.black_pixel_share.mean(),
            black_pixel_share_maximum=self.black_pixel_share.maximum(),
            minimum_luma=self.minimum_luma.minimum(),
            maximum_luma=self.maximum_luma.maximum(),
            luma_range_evidence=luma_range_evidence,
            tenth_percentile_luma_mean=self.tenth_percentile_luma.mean(),
            ninetieth_percentile_luma_mean=self.ninetieth_percentile_luma.mean(),
            clipped_highlight_frame_percent=100.0 * clipped_highlight_count / frame_count,
            crushed_shadow_frame_percent=100.0 * crushed_shadow_count / frame_count,
            frame_difference_mean=self.frame_difference.mean(),
            frame_difference_maximum=self.frame_difference.maximum(),
            temporal_outlier_mean=self.temporal_outlier.mean(),
            temporal_outlier_maximum=self.temporal_outlier.maximum(),
            out_of_legal_range_mean=self.out_of_legal_range.mean(),
            out_of_legal_range_maximum=self.out_of_legal_range.maximum(),
        )


def _parse_finite_float(value_text: str, context: str) -> float:
    try:
        value = float(value_text)
    except ValueError as error:
        raise FrameStatisticsParseError(f"unparsable {context}: {value_text!r}") from error
    if not math.isfinite(value):
        raise FrameStatisticsParseError(f"non-finite {context}: {value_text!r}")
    return value


def _validated_freeze_interval(
    start_seconds: float, end_seconds: float, *, context: str
) -> VideoTimeInterval:
    try:
        return VideoTimeInterval(
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )
    except ValueError as error:
        raise FrameStatisticsParseError(
            f"invalid freeze interval in {context}: {start_seconds} to {end_seconds}"
        ) from error


def _required_finite_value(frame: _ParsedFrame, frame_index: int, metadata_key: str) -> float:
    value_text = frame.metadata.get(metadata_key)
    if value_text is None:
        raise FrameStatisticsParseError(
            f"frame {frame_index} (pts_time {frame.presentation_time_seconds}) is missing "
            f"{metadata_key} -- truncated output, or the filter graph did not request it"
        )
    return _parse_finite_float(value_text, f"{metadata_key} of frame {frame_index}")


def _required_luma_value(frame: _ParsedFrame, frame_index: int, metadata_key: str) -> float:
    value = _required_finite_value(frame, frame_index, metadata_key)
    if not 0.0 <= value <= _MAXIMUM_EIGHT_BIT_LUMA:
        raise FrameStatisticsParseError(
            f"{metadata_key} of frame {frame_index} is {value}, outside the 8-bit "
            "range 0-255: the format pin ahead of the measuring filters was lost"
        )
    return value


def _parse_metadata_print_frames(output_lines: Iterable[str]) -> Iterator[_ParsedFrame]:
    """Parse FFmpeg metadata incrementally, retaining at most one frame."""
    current_frame: _ParsedFrame | None = None
    for line_number, line_with_ending in enumerate(output_lines, start=1):
        line = line_with_ending.rstrip("\r\n")
        if not line.strip():
            continue
        header_match = _FRAME_HEADER_PATTERN.match(line)
        if header_match is not None:
            if current_frame is not None:
                yield current_frame
            current_frame = _ParsedFrame(
                presentation_time_seconds=_parse_finite_float(
                    header_match.group("pts_time"),
                    f"pts_time on line {line_number}",
                ),
                metadata={},
            )
            continue
        metadata_match = _METADATA_LINE_PATTERN.match(line)
        if metadata_match is None:
            raise FrameStatisticsParseError(
                f"unparsable instrument output on line {line_number}: {line!r}"
            )
        if current_frame is None:
            raise FrameStatisticsParseError(
                f"metadata line {line_number} before any frame header: {line!r}"
            )
        current_frame.metadata[metadata_match.group("key")] = metadata_match.group("value")
    if current_frame is not None:
        yield current_frame


def _aggregate_frame_statistics_lines(
    output_lines: Iterable[str], settings: FrameStatisticsSettings
) -> _AggregatedFrameStatistics:
    accumulator = _FrameStatisticsAccumulator(settings=settings)
    for frame in _parse_metadata_print_frames(output_lines):
        accumulator.add_frame(frame)
    return accumulator.finish()


def _aggregate_frame_statistics_output(
    output_text: str,
    *,
    settings: FrameStatisticsSettings | None = None,
) -> _AggregatedFrameStatistics:
    """Aggregate synthetic instrument output without invoking FFmpeg."""
    return _aggregate_frame_statistics_lines(
        StringIO(output_text), settings or FrameStatisticsSettings()
    )


def _aggregate_cached_frame_statistics(
    cache_path: Path, settings: FrameStatisticsSettings
) -> _AggregatedFrameStatistics | None:
    """Read a valid cached instrument stream or remove a corrupt cache."""
    try:
        with cache_path.open(encoding="utf-8") as cached_output:
            return _aggregate_frame_statistics_lines(cached_output, settings)
    except FileNotFoundError:
        return None
    except (UnicodeDecodeError, FrameStatisticsParseError):
        cache_path.unlink(missing_ok=True)
        return None


def _copy_instrument_output_lines(
    output_lines: Iterable[str], cache_output: _TextWriter
) -> Iterator[str]:
    """Write streamed metadata to a cache candidate while it is parsed."""
    for output_line in output_lines:
        cache_output.write(output_line)
        yield output_line


@contextmanager
def _temporary_instrument_cache_output(
    cache_path: Path | None,
) -> Iterator[tuple[_TextWriter | None, Path | None]]:
    """Open a unique cache candidate and close it before atomic publication."""
    if cache_path is None:
        yield None, None
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=cache_path.parent,
        prefix=f".{cache_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as cache_output:
        yield cache_output, Path(cache_output.name)


def frame_statistics_filter_graph(settings: FrameStatisticsSettings) -> str:
    """Return the effective single-pass FFmpeg measurement graph."""
    return (
        "format=pix_fmts=yuv420p,"
        f"blackframe=amount=0:threshold={settings.black_pixel_luma_threshold},"
        "freezedetect="
        f"n={settings.freeze_noise_tolerance_decibels}dB:"
        f"d={settings.freeze_minimum_duration_seconds},"
        "signalstats=stat=tout+brng,metadata=mode=print:file=-"
    )


@cache
def _validate_frame_statistics_filters(
    toolchain: VideoMeasurementToolchain,
) -> None:
    completed_process = subprocess.run(
        [str(toolchain.ffmpeg_executable), "-hide_banner", "-filters"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed_process.returncode != 0:
        raise UnsupportedVideoMeasurementToolchainError(
            "could not inspect FFmpeg filters: " + completed_process.stderr.strip()
        )
    available_filter_names = {
        match.group("filter_name")
        for line in completed_process.stdout.splitlines()
        if (match := _FILTER_LIST_ENTRY_PATTERN.match(line)) is not None
    }
    required_filter_names = {
        "blackframe",
        "format",
        "freezedetect",
        "metadata",
        "signalstats",
    }
    missing_filter_names = required_filter_names - available_filter_names
    if missing_filter_names:
        raise UnsupportedVideoMeasurementToolchainError(
            "FFmpeg is missing required video measurement filters: "
            + ", ".join(sorted(missing_filter_names))
        )


def _attach_provenance(
    aggregate: _AggregatedFrameStatistics,
    provenance: FrameStatisticsProvenance,
) -> VideoFrameStatistics:
    return VideoFrameStatistics(
        decoded_frame_count=aggregate.decoded_frame_count,
        duration_seconds=aggregate.duration_seconds,
        black_frame_count=aggregate.black_frame_count,
        black_frame_percent=aggregate.black_frame_percent,
        overexposed_frame_count=aggregate.overexposed_frame_count,
        overexposed_frame_percent=aggregate.overexposed_frame_percent,
        freeze_intervals=aggregate.freeze_intervals,
        freeze_total_seconds=aggregate.freeze_total_seconds,
        average_luma_mean=aggregate.average_luma_mean,
        average_luma_minimum=aggregate.average_luma_minimum,
        average_luma_maximum=aggregate.average_luma_maximum,
        black_pixel_share_mean=aggregate.black_pixel_share_mean,
        black_pixel_share_maximum=aggregate.black_pixel_share_maximum,
        minimum_luma=aggregate.minimum_luma,
        maximum_luma=aggregate.maximum_luma,
        luma_range_evidence=aggregate.luma_range_evidence,
        tenth_percentile_luma_mean=aggregate.tenth_percentile_luma_mean,
        ninetieth_percentile_luma_mean=aggregate.ninetieth_percentile_luma_mean,
        clipped_highlight_frame_percent=aggregate.clipped_highlight_frame_percent,
        crushed_shadow_frame_percent=aggregate.crushed_shadow_frame_percent,
        frame_difference_mean=aggregate.frame_difference_mean,
        frame_difference_maximum=aggregate.frame_difference_maximum,
        temporal_outlier_mean=aggregate.temporal_outlier_mean,
        temporal_outlier_maximum=aggregate.temporal_outlier_maximum,
        out_of_legal_range_mean=aggregate.out_of_legal_range_mean,
        out_of_legal_range_maximum=aggregate.out_of_legal_range_maximum,
        provenance=provenance,
    )


def measure_video_frame_statistics(
    video: Path,
    *,
    toolchain: VideoMeasurementToolchain,
    settings: FrameStatisticsSettings | None = None,
    instrument_output_cache_path: Path | None = None,
) -> VideoFrameStatistics:
    """Decode ``video`` once and aggregate strict per-frame measurements."""
    effective_settings = settings or FrameStatisticsSettings()
    filter_graph = frame_statistics_filter_graph(effective_settings)
    provenance = FrameStatisticsProvenance(
        measurement_definition_version=FRAME_STATISTICS_DEFINITION_VERSION,
        ffmpeg_version=toolchain.ffmpeg_version,
        filter_graph=filter_graph,
        settings=effective_settings,
    )
    if instrument_output_cache_path is not None:
        cached_aggregate = _aggregate_cached_frame_statistics(
            instrument_output_cache_path, effective_settings
        )
        if cached_aggregate is not None:
            return _attach_provenance(cached_aggregate, provenance)

    _validate_frame_statistics_filters(toolchain)
    command = [
        str(toolchain.ffmpeg_executable),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostats",
        "-i",
        str(video),
        "-vf",
        filter_graph,
        "-f",
        "null",
        "-",
    ]
    temporary_cache_path: Path | None = None
    try:
        with _temporary_instrument_cache_output(instrument_output_cache_path) as (
            cache_output,
            candidate_cache_path,
        ):
            temporary_cache_path = candidate_cache_path
            with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as standard_error_file:
                instrument_process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=standard_error_file,
                    text=True,
                )
                assert instrument_process.stdout is not None
                aggregation_error: FrameStatisticsParseError | None = None
                aggregate: _AggregatedFrameStatistics | None = None
                terminated_for_parse_error = False
                instrument_output_lines: Iterable[str] = instrument_process.stdout
                if cache_output is not None:
                    instrument_output_lines = _copy_instrument_output_lines(
                        instrument_output_lines, cache_output
                    )
                try:
                    aggregate = _aggregate_frame_statistics_lines(
                        instrument_output_lines, effective_settings
                    )
                except FrameStatisticsParseError as error:
                    aggregation_error = error
                    if (
                        not isinstance(error, _NoInstrumentFramesError)
                        and instrument_process.poll() is None
                    ):
                        terminated_for_parse_error = True
                        instrument_process.terminate()
                finally:
                    instrument_process.stdout.close()
                return_code = instrument_process.wait()
                standard_error_file.seek(0)
                standard_error = standard_error_file.read()

        if aggregation_error is not None and terminated_for_parse_error:
            raise aggregation_error
        if return_code != 0:
            standard_error_tail = "\n".join(standard_error.strip().splitlines()[-5:])
            raise FrameStatisticsExecutionError(
                f"ffmpeg frame-statistics pass failed for {video}: {standard_error_tail}"
            )
        if aggregation_error is not None:
            raise aggregation_error
        assert aggregate is not None

        if temporary_cache_path is not None and instrument_output_cache_path is not None:
            temporary_cache_path.replace(instrument_output_cache_path)
            temporary_cache_path = None
        return _attach_provenance(aggregate, provenance)
    finally:
        if temporary_cache_path is not None:
            temporary_cache_path.unlink(missing_ok=True)
