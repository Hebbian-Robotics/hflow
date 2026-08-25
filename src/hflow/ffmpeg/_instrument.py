"""The single-pass ffmpeg frame instrument.

One decode pass, one filter graph, one shared frame denominator::

    blackframe=amount={...}:threshold={...},freezedetect=n={...}:d={...},
    signalstats,metadata=mode=print:file=-

run as ``ffmpeg -i video -vf <graph> -f null -`` against the pinned binary,
parsing the ``metadata=print`` output. Parsing is HARD-VALIDATED: a frame
missing ``lavfi.signalstats.YAVG``, a NaN, or an unparsable line is an error
(``InstrumentParseError``), never a silent gap -- a measuring instrument that
sometimes skips frames is worse than one that fails loudly.

Covers the "camera blackout" and "bad frames" issues named in Dyna's
article. Deliberately no blurdetect: it inverts on motion smear (fast, good
manipulation looks "blurry"). Thresholds are exposed and user-owned.
"""

import itertools
import math
import re
import statistics
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

from hflow.ffmpeg._binary import ffmpeg_path, ffmpeg_version


class InstrumentParseError(RuntimeError):
    pass


# Nominal limited-range luma bounds. Luma leaving them is what distinguishes
# full-range footage, and the bounds are inclusive: 16 and 235 themselves are
# limited-compatible. Verified against this build's ``colordetect`` filter,
# which reports full range on exactly the same condition -- so the range is
# derived from luma the instrument already reports rather than scraped out of
# a filter's log line, which no version of ffmpeg promises to keep stable.
LIMITED_RANGE_LUMA_FLOOR = 16
LIMITED_RANGE_LUMA_CEILING = 235

# Exposure gates, selected by the coding range measured above. These are
# measurement definitions rather than acceptance bars: they say what "clipped"
# and "crushed" mean, and keeping them fixed is what makes the share
# comparable across corpora. The 90th/10th luma percentiles are the subject,
# not the extremes, so one hot pixel cannot manufacture a defect.
_CLIPPED_HIGHLIGHT_GATE_BY_RANGE = {False: 246.0, True: 254.0}
_CRUSHED_SHADOW_GATE_BY_RANGE = {False: 5.0, True: 1.0}


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
    # Share of each frame's pixels below the black threshold, over every frame
    # rather than only the frames flagged black -- so a half-covered lens is
    # visible instead of rounding to "not a blackout".
    black_pixel_share_mean: float  # 0.0-100.0
    black_pixel_share_max: float  # 0.0-100.0
    # Whole-scale luma extremes, and the coding range they imply.
    luma_min: float
    luma_max: float
    full_range_detected: bool
    # Robust per-frame luma percentiles, averaged over frames.
    luma_p10_mean: float
    luma_p90_mean: float
    # Range-gated exposure defects, as a share of frames.
    clipped_highlight_pct: float  # 0.0-100.0
    crushed_shadow_pct: float  # 0.0-100.0
    # Mean absolute luma difference between consecutive frames: near zero means
    # a still scene, which is a different fact from a frozen feed.
    frame_difference_mean: float
    frame_difference_max: float
    # Intra-frame impulse noise and dropout streaks (signalstats TOUT).
    temporal_outlier_mean: float
    temporal_outlier_max: float
    # Share of samples outside the nominal range (signalstats BRNG). Dominated
    # by range mismatch on full-range footage, so read it with
    # ``full_range_detected``.
    out_of_legal_range_mean: float
    out_of_legal_range_max: float
    # Which binary produced these readings. Builds disagree about absolute luma
    # by more than rounding, so a reading without its instrument is not
    # comparable to another. Empty when aggregating text rather than decoding.
    instrument_version: str = ""


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
_SIGNALSTATS_YMIN_KEY = "lavfi.signalstats.YMIN"
_SIGNALSTATS_YLOW_KEY = "lavfi.signalstats.YLOW"
_SIGNALSTATS_YHIGH_KEY = "lavfi.signalstats.YHIGH"
_SIGNALSTATS_YMAX_KEY = "lavfi.signalstats.YMAX"
_SIGNALSTATS_YDIF_KEY = "lavfi.signalstats.YDIF"
_SIGNALSTATS_TOUT_KEY = "lavfi.signalstats.TOUT"
_SIGNALSTATS_BRNG_KEY = "lavfi.signalstats.BRNG"

# An 8-bit pipeline reports luma on 0-255. Anything above that means the format
# pin ahead of the measuring filters was lost and a higher-depth source is being
# read on its own scale, where every threshold here is meaningless -- so fail
# loudly rather than record numbers four times too large.
_MAXIMUM_EIGHT_BIT_LUMA = 255.0


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


def _required_luma_series(frames: list[_ParsedFrame], key: str) -> list[float]:
    """One signalstats luma signal across every frame, hard-validated.

    A missing reading is an error rather than a gap: a frame silently dropped
    from a denominator turns "we could not measure this" into "we measured it
    and it was clean", which is the one answer a quality instrument must never
    invent. The 0-255 bound catches a lost format pin, where a higher-depth
    source would otherwise report every level four times too large.
    """
    values: list[float] = []
    for frame_index, frame in enumerate(frames):
        text = frame.metadata.get(key)
        if text is None:
            raise InstrumentParseError(
                f"frame {frame_index} (pts_time {frame.pts_time_s}) is missing {key} "
                "-- truncated output, or the filter graph did not request it"
            )
        value = _parse_finite_float(text, f"{key} of frame {frame_index}")
        if not 0.0 <= value <= _MAXIMUM_EIGHT_BIT_LUMA:
            raise InstrumentParseError(
                f"{key} of frame {frame_index} is {value}, outside the 8-bit range "
                "0-255: the format pin ahead of the measuring filters was lost"
            )
        values.append(value)
    return values


def _required_share_series(frames: list[_ParsedFrame], key: str) -> list[float]:
    """One signalstats 0-1 share signal across every frame, hard-validated."""
    values: list[float] = []
    for frame_index, frame in enumerate(frames):
        text = frame.metadata.get(key)
        if text is None:
            raise InstrumentParseError(
                f"frame {frame_index} (pts_time {frame.pts_time_s}) is missing {key} "
                "-- truncated output, or the filter graph did not request it"
            )
        values.append(_parse_finite_float(text, f"{key} of frame {frame_index}"))
    return values


def _stats_from_instrument_output(
    output_text: str,
    *,
    bright_luma_threshold: float = 235.0,
    black_frame_amount_pct: int = 98,
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

    # ``blackframe=amount=0`` reports every frame's share, so the flag is
    # applied here instead. pblack is an integer percent, matching the integer
    # ``amount`` the filter would have compared against.
    black_pixel_shares = _required_share_series(frames, _BLACKFRAME_PBLACK_KEY)
    black_frame_count = sum(1 for share in black_pixel_shares if share >= black_frame_amount_pct)
    overexposed_frame_count = sum(1 for v in luma_avg_values if v >= bright_luma_threshold)

    luma_minima = _required_luma_series(frames, _SIGNALSTATS_YMIN_KEY)
    luma_maxima = _required_luma_series(frames, _SIGNALSTATS_YMAX_KEY)
    luma_p10_values = _required_luma_series(frames, _SIGNALSTATS_YLOW_KEY)
    luma_p90_values = _required_luma_series(frames, _SIGNALSTATS_YHIGH_KEY)
    frame_differences = _required_luma_series(frames, _SIGNALSTATS_YDIF_KEY)
    temporal_outliers = _required_share_series(frames, _SIGNALSTATS_TOUT_KEY)
    out_of_legal_range = _required_share_series(frames, _SIGNALSTATS_BRNG_KEY)

    luma_min = min(luma_minima)
    luma_max = max(luma_maxima)
    # One verdict for the clip, not per frame: a stream is encoded one way, and
    # any frame leaving the nominal bounds proves which way.
    full_range_detected = (
        luma_min < LIMITED_RANGE_LUMA_FLOOR or luma_max > LIMITED_RANGE_LUMA_CEILING
    )
    clipped_gate = _CLIPPED_HIGHLIGHT_GATE_BY_RANGE[full_range_detected]
    crushed_gate = _CRUSHED_SHADOW_GATE_BY_RANGE[full_range_detected]
    clipped_frame_count = sum(1 for value in luma_p90_values if value > clipped_gate)
    crushed_frame_count = sum(1 for value in luma_p10_values if value < crushed_gate)

    # The first frame has no predecessor, so its YDIF is a sentinel zero.
    # Excluded by position, never by value: filtering zeros would delete the
    # real stillness this signal exists to measure.
    observed_differences = frame_differences[1:]

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
        black_pixel_share_mean=statistics.fmean(black_pixel_shares),
        black_pixel_share_max=max(black_pixel_shares),
        luma_min=luma_min,
        luma_max=luma_max,
        full_range_detected=full_range_detected,
        luma_p10_mean=statistics.fmean(luma_p10_values),
        luma_p90_mean=statistics.fmean(luma_p90_values),
        clipped_highlight_pct=100.0 * clipped_frame_count / frame_count,
        crushed_shadow_pct=100.0 * crushed_frame_count / frame_count,
        frame_difference_mean=statistics.fmean(observed_differences)
        if observed_differences
        else 0.0,
        frame_difference_max=max(observed_differences) if observed_differences else 0.0,
        temporal_outlier_mean=statistics.fmean(temporal_outliers),
        temporal_outlier_max=max(temporal_outliers),
        out_of_legal_range_mean=statistics.fmean(out_of_legal_range),
        out_of_legal_range_max=max(out_of_legal_range),
    )


def instrument_filter_graph(
    *,
    black_pixel_threshold: int,
    freeze_noise_db: float,
    freeze_min_duration_s: float,
) -> str:
    """The measuring graph, as one string so it can be recorded as evidence.

    ``format=pix_fmts=yuv420p`` comes first and is load-bearing: without it a
    higher-depth source reaches the measuring filters on its own scale, where
    every luma threshold below is off by a factor of four. It is a no-op for the
    8-bit H.264 canonical episodes carry.

    ``blackframe=amount=0`` asks for every frame's black-pixel share rather than
    only the frames a filter-side threshold would have flagged; the flag is
    applied during aggregation, which yields the same count plus the
    distribution behind it.
    """
    return (
        "format=pix_fmts=yuv420p,"
        f"blackframe=amount=0:threshold={black_pixel_threshold},"
        f"freezedetect=n={freeze_noise_db}dB:d={freeze_min_duration_s},"
        "signalstats=stat=tout+brng,metadata=mode=print:file=-"
    )


# The instrument's raw ``metadata=print`` stdout for one video, cached as a
# text file beside the video. The cache key is the video path itself -- the
# same ``Episode.video(topic)`` workdir-MP4 path is the same underlying
# footage, so any caller reaching for the instrument over that MP4 gets the
# same decode output. Aggregation thresholds (``bright_luma_threshold``,
# ``freeze_min_duration_s``, ``black_frame_amount_pct``) re-run from the
# cached text on every call, so a threshold change is free rather than
# invalidating the cache.
_INSTRUMENT_CACHE_SUFFIX = ".instrument.txt"


def _instrument_cache_path(video: Path) -> Path | None:
    """Where the cached instrument stdout for ``video`` lives, or None if
    caching is disabled for this path.

    A sibling of the video, named after its stem. Caching is disabled when
    the video is not an ``.mp4`` -- the cache file would otherwise live at
    the same path on case-insensitive filesystems, and writing a
    ``.instrument.txt`` next to a ``.h264`` source would not match the
    convention the workdir uses for the remux cache.
    """
    if video.suffix.lower() != ".mp4":
        return None
    return video.with_name(f"{video.stem}{_INSTRUMENT_CACHE_SUFFIX}")


def _write_instrument_cache(cache_path: Path, output_text: str) -> None:
    """Atomically replace the cache file with ``output_text``.

    Same shape as the MP4 remux write: a partial file at the final path
    would be read as a complete cache, so we always go through a ``.tmp``
    sibling and ``Path.replace`` into place. On any failure the temp file
    is removed so a half-written cache cannot masquerade as a complete one.
    """
    temporary = cache_path.with_name(cache_path.name + ".tmp")
    try:
        temporary.write_text(output_text, encoding="utf-8")
        temporary.replace(cache_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_instrument_cache(cache_path: Path) -> str | None:
    """Return the cached stdout, or None if the cache is missing.

    A parse failure deletes the file and returns None, so the next call
    re-decodes -- the same self-healing shape as a corrupt MP4 would
    trigger a re-remux. A read error other than missing (permission) or
    non-UTF-8 propagates: the caller will treat it the same as any other
    I/O error and the user will see a real message rather than a silently
    wrong instrument reading.
    """
    if not cache_path.is_file():
        return None
    try:
        return cache_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        cache_path.unlink(missing_ok=True)
        return None


def frame_stats(
    video: Path,
    *,
    black_frame_amount_pct: int = 98,
    black_pixel_threshold: int = 17,
    freeze_noise_db: float = -60.0,
    freeze_min_duration_s: float = 2.0,
    bright_luma_threshold: float = 235.0,
) -> FrameStats:
    """Run the instrument over ``video`` and aggregate.

    One decode pass, one filter graph, one shared frame denominator. The
    raw instrument stdout is cached in a ``.instrument.txt`` file beside
    ``video`` (the workdir MP4 ``Episode.video()`` produces), so a second
    call over the same video with the same footage skips the ffmpeg
    decode and re-aggregates with whatever threshold parameters the
    caller passed this time. A pipeline that registers both
    :func:`hflow.checks.camera_frame_stats` and
    :func:`hflow.checks.camera_signal_quality` therefore pays one decode
    per camera per episode rather than two; a wrapper that reaches for
    ``frame_stats`` directly gets the same benefit.

    - ``blackframe`` yields each frame's black-pixel share; frames at or above
      ``black_frame_amount_pct`` are counted black. ``black_pixel_threshold``
      defaults to 17, which includes video-range black (16) and excludes the
      ordinary dark detail that ffmpeg's own default of 32 reads as dark.
    - ``freezedetect=n={noise}dB:d={duration}`` yields freeze intervals; an
      unterminated freeze at EOF closes at the video duration.
    - ``signalstats`` yields per-frame luma extremes and percentiles, frame
      differences, impulse noise, and nominal-range excursions. The coding
      range is derived from the luma extremes, and selects the exposure gates.
    - Frames with average luma at or above ``bright_luma_threshold`` are
      counted as overexposed, on the same frame-count denominator.
    """
    cache_path = _instrument_cache_path(video)

    if cache_path is not None:
        cached_output = _read_instrument_cache(cache_path)
        if cached_output is not None:
            try:
                stats = _stats_from_instrument_output(
                    cached_output,
                    bright_luma_threshold=bright_luma_threshold,
                    black_frame_amount_pct=black_frame_amount_pct,
                )
            except InstrumentParseError:
                # Corrupt or truncated cache: drop it and re-decode. A
                # check that caught a wrong number from a stale cache
                # would be the one answer this instrument must never
                # invent.
                cache_path.unlink(missing_ok=True)
            else:
                # Stamped here rather than read by callers: a check that
                # reached for ``ffmpeg_version`` itself would capture an
                # lru_cache wrapper, which step identity cannot
                # content-hash, and registering that check would fail.
                return replace(stats, instrument_version=ffmpeg_version())

    graph = instrument_filter_graph(
        black_pixel_threshold=black_pixel_threshold,
        freeze_noise_db=freeze_noise_db,
        freeze_min_duration_s=freeze_min_duration_s,
    )
    command = [
        str(ffmpeg_path()),
        "-hide_banner",
        "-nostats",
        "-i",
        str(video),
        "-vf",
        graph,
        "-f",
        "null",
        "-",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        stderr_tail = "\n".join(completed.stderr.strip().splitlines()[-5:])
        raise RuntimeError(f"ffmpeg instrument pass failed for {video}: {stderr_tail}")
    # Write the raw stdout before aggregating so the next caller (or a
    # future process) can read the cache even if aggregation itself
    # raised -- a real ffmpeg output the parser rejected is still a
    # valid decode worth keeping for debugging.
    if cache_path is not None:
        _write_instrument_cache(cache_path, completed.stdout)
    stats = _stats_from_instrument_output(
        completed.stdout,
        bright_luma_threshold=bright_luma_threshold,
        black_frame_amount_pct=black_frame_amount_pct,
    )
    # Stamped here rather than read by callers: a check that reached for
    # ``ffmpeg_version`` itself would capture an lru_cache wrapper, which step
    # identity cannot content-hash, and registering that check would fail.
    return replace(stats, instrument_version=ffmpeg_version())
