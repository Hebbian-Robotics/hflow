"""The single-pass ffmpeg frame instrument.

One decode pass, one filter graph, one shared frame denominator::

    blackframe=amount={...}:threshold={...},freezedetect=n={...}:d={...},
    signalstats,metadata=mode=print:file=-

run as ``ffmpeg -i video -vf <graph> -f null -`` against the pinned binary,
parsing the ``metadata=print`` output. Parsing is HARD-VALIDATED: a frame
missing ``lavfi.signalstats.YAVG``, a NaN, or an unparsable line is an error
(``InstrumentParseError``), never a silent gap -- a measuring instrument that
sometimes skips frames is worse than one that fails loudly.

Covers Dyna's "camera blackout" and "bad frames" named issues. Deliberately
no blurdetect: it inverts on motion smear (fast, good manipulation looks
"blurry"). Thresholds are exposed and user-owned.
"""

import itertools
import math
import re
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path

from hflow.ffmpeg._binary import ffmpeg_path


class InstrumentParseError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrameStats:
    """Aggregated per-frame statistics from one instrument pass."""

    frame_count: int
    duration_s: float
    black_frame_count: int
    black_frame_pct: float  # 0.0-100.0, denominator = frame_count
    overexposed_frame_count: int
    overexposed_frame_pct: float  # 0.0-100.0, denominator = frame_count
    freeze_intervals: list[tuple[float, float]]  # (start_s, end_s) pairs
    freeze_total_s: float
    luma_avg_mean: float
    luma_avg_min: float
    luma_avg_max: float


# ``metadata=mode=print:file=-`` emits one header per frame followed by
# ``key=value`` lines for that frame's attached metadata.
_FRAME_HEADER_PATTERN = re.compile(
    r"^frame:(?P<frame_index>\d+)\s+pts:(?P<pts>-?\d+)\s+pts_time:(?P<pts_time>-?[0-9.eE+-]+)\s*$"
)
_METADATA_LINE_PATTERN = re.compile(r"^(?P<key>lavfi\.[A-Za-z0-9_.]+)=(?P<value>.*)$")

_SIGNALSTATS_YAVG_KEY = "lavfi.signalstats.YAVG"
_BLACKFRAME_PBLACK_KEY = "lavfi.blackframe.pblack"
_FREEZE_START_KEY = "lavfi.freezedetect.freeze_start"
_FREEZE_END_KEY = "lavfi.freezedetect.freeze_end"


@dataclass(frozen=True)
class _ParsedFrame:
    pts_time_s: float
    metadata: dict[str, str]


def _parse_finite_float(value_text: str, context: str) -> float:
    try:
        value = float(value_text)
    except ValueError as error:
        raise InstrumentParseError(f"unparsable {context}: {value_text!r}") from error
    if math.isnan(value) or math.isinf(value):
        raise InstrumentParseError(f"non-finite {context}: {value_text!r}")
    return value


def _parse_metadata_print_frames(output_text: str) -> list[_ParsedFrame]:
    """Strictly parse ``metadata=mode=print`` text into per-frame records."""
    frames: list[_ParsedFrame] = []
    current_metadata: dict[str, str] | None = None
    for line_number, line in enumerate(output_text.splitlines(), start=1):
        if not line.strip():
            continue
        header_match = _FRAME_HEADER_PATTERN.match(line)
        if header_match is not None:
            pts_time_s = _parse_finite_float(
                header_match.group("pts_time"), f"pts_time on line {line_number}"
            )
            current_metadata = {}
            frames.append(_ParsedFrame(pts_time_s=pts_time_s, metadata=current_metadata))
            continue
        metadata_match = _METADATA_LINE_PATTERN.match(line)
        if metadata_match is None:
            raise InstrumentParseError(
                f"unparsable instrument output on line {line_number}: {line!r}"
            )
        if current_metadata is None:
            raise InstrumentParseError(
                f"metadata line {line_number} before any frame header: {line!r}"
            )
        current_metadata[metadata_match.group("key")] = metadata_match.group("value")
    return frames


def _frame_luma_avg(frame: _ParsedFrame, frame_index: int) -> float:
    yavg_text = frame.metadata.get(_SIGNALSTATS_YAVG_KEY)
    if yavg_text is None:
        raise InstrumentParseError(
            f"frame {frame_index} (pts_time {frame.pts_time_s}) is missing "
            f"{_SIGNALSTATS_YAVG_KEY} -- truncated or non-signalstats output"
        )
    return _parse_finite_float(yavg_text, f"{_SIGNALSTATS_YAVG_KEY} of frame {frame_index}")


def _collect_freeze_intervals(
    frames: list[_ParsedFrame], video_duration_s: float
) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    open_freeze_start_s: float | None = None
    for frame_index, frame in enumerate(frames):
        start_text = frame.metadata.get(_FREEZE_START_KEY)
        if start_text is not None:
            if open_freeze_start_s is not None:
                raise InstrumentParseError(
                    f"frame {frame_index}: freeze_start while a freeze is already open"
                )
            open_freeze_start_s = _parse_finite_float(
                start_text, f"freeze_start of frame {frame_index}"
            )
        end_text = frame.metadata.get(_FREEZE_END_KEY)
        if end_text is not None:
            if open_freeze_start_s is None:
                raise InstrumentParseError(
                    f"frame {frame_index}: freeze_end without a matching freeze_start"
                )
            freeze_end_s = _parse_finite_float(end_text, f"freeze_end of frame {frame_index}")
            intervals.append((open_freeze_start_s, freeze_end_s))
            open_freeze_start_s = None
    # An unterminated freeze runs to end of video: close it at the duration.
    if open_freeze_start_s is not None:
        intervals.append((open_freeze_start_s, video_duration_s))
    return intervals


def _stats_from_instrument_output(
    output_text: str, *, bright_luma_threshold: float = 235.0
) -> FrameStats:
    """Pure aggregation of the instrument's stdout (testable on synthetic text)."""
    frames = _parse_metadata_print_frames(output_text)
    if not frames:
        raise InstrumentParseError("instrument produced no frames")
    luma_avg_values = [
        _frame_luma_avg(frame, frame_index) for frame_index, frame in enumerate(frames)
    ]
    frame_count = len(frames)
    pts_times_s = [frame.pts_time_s for frame in frames]
    if frame_count >= 2:
        median_frame_interval_s = statistics.median(
            later - earlier for earlier, later in itertools.pairwise(pts_times_s)
        )
    else:
        median_frame_interval_s = 0.0
    duration_s = pts_times_s[-1] + median_frame_interval_s

    black_frame_count = 0
    for frame_index, frame in enumerate(frames):
        pblack_text = frame.metadata.get(_BLACKFRAME_PBLACK_KEY)
        if pblack_text is None:
            continue
        # blackframe only attaches pblack to frames it flagged; still validate.
        _parse_finite_float(pblack_text, f"{_BLACKFRAME_PBLACK_KEY} of frame {frame_index}")
        black_frame_count += 1

    overexposed_frame_count = sum(1 for v in luma_avg_values if v >= bright_luma_threshold)

    freeze_intervals = _collect_freeze_intervals(frames, duration_s)
    return FrameStats(
        frame_count=frame_count,
        duration_s=duration_s,
        black_frame_count=black_frame_count,
        black_frame_pct=100.0 * black_frame_count / frame_count,
        overexposed_frame_count=overexposed_frame_count,
        overexposed_frame_pct=100.0 * overexposed_frame_count / frame_count,
        freeze_intervals=freeze_intervals,
        freeze_total_s=sum(end_s - start_s for start_s, end_s in freeze_intervals),
        luma_avg_mean=statistics.fmean(luma_avg_values),
        luma_avg_min=min(luma_avg_values),
        luma_avg_max=max(luma_avg_values),
    )


def frame_stats(
    video: Path,
    *,
    black_frame_amount_pct: int = 98,
    black_pixel_threshold: int = 32,
    freeze_noise_db: float = -60.0,
    freeze_min_duration_s: float = 2.0,
    bright_luma_threshold: float = 235.0,
) -> FrameStats:
    """Run the instrument over ``video`` and aggregate.

    - ``blackframe=amount:threshold`` flags frames whose pixels are
      near-black; ``black_frame_pct`` uses the signalstats frame count as the
      shared denominator.
    - ``freezedetect=n={noise}dB:d={duration}`` yields freeze intervals; an
      unterminated freeze at EOF closes at the video duration.
    - ``signalstats`` yields per-frame YAVG, aggregated to mean/min/max.
    - Frames with average luma at or above ``bright_luma_threshold`` are
      counted as overexposed; ``overexposed_frame_pct`` uses the same
      frame-count denominator as ``black_frame_pct``.
    """
    filter_graph = (
        f"blackframe=amount={black_frame_amount_pct}:threshold={black_pixel_threshold},"
        f"freezedetect=n={freeze_noise_db}dB:d={freeze_min_duration_s},"
        "signalstats,metadata=mode=print:file=-"
    )
    command = [
        str(ffmpeg_path()),
        "-hide_banner",
        "-nostats",
        "-i",
        str(video),
        "-vf",
        filter_graph,
        "-f",
        "null",
        "-",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        stderr_tail = "\n".join(completed.stderr.strip().splitlines()[-5:])
        raise RuntimeError(f"ffmpeg instrument pass failed for {video}: {stderr_tail}")
    return _stats_from_instrument_output(
        completed.stdout, bright_luma_threshold=bright_luma_threshold
    )
