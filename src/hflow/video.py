"""H.264 elementary-stream expertise: encode, split, and remux access units.

This module encodes the non-obvious knowledge behind the canonical video
convention (docs/ARCHITECTURE.md, "The episode container"):

- ``foxglove.CompressedVideo`` requires an Annex B byte stream, SPS/PPS
  attached to every keyframe, no B-frames, and one decodable access unit per
  message.
- x264 delivers exactly that with :func:`canonical_x264_parameters`, which
  repeats parameter sets, emits access-unit delimiters, disables B-frames and
  scene-cut keyframes, and fixes the GOP cadence.
- A raw ``.h264`` stream carries no timestamps; remuxing to MP4 requires
  declaring the frame rate to the demuxer (``-r`` before ``-i``).

Encoding and remuxing use the pinned ffmpeg (``hflow.ffmpeg``). Annex B
inspection and delimiter repair come from the shared ``hebbian_mcap.h264`` module.
"""

import itertools
import sqlite3
import statistics
import subprocess
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from contextlib import closing, suppress
from pathlib import Path

import numpy as np
from hebbian_mcap.h264 import (
    AccessUnit as AccessUnit,
)
from hebbian_mcap.h264 import (
    PictureCodingScan as PictureCodingScan,
)
from hebbian_mcap.h264 import (
    PictureCodingScanner as _PictureCodingScanner,
)
from hebbian_mcap.h264 import (
    count_h264_pictures as count_h264_pictures,
)
from hebbian_mcap.h264 import (
    ensure_access_unit_delimiter as ensure_access_unit_delimiter,
)
from hebbian_mcap.h264 import (
    scan_picture_coding_types as scan_picture_coding_types,
)
from hebbian_mcap.h264 import (
    split_annex_b_stream as split_annex_b_stream,
)

from hflow.ffmpeg import ffmpeg_path


class VideoEncodeError(RuntimeError):
    """ffmpeg failed, or the encoder violated a documented output guarantee."""


_STDERR_TAIL_CHARACTER_LIMIT = 2000


def canonical_x264_parameters(gop_frames: int) -> str:
    """Return canonical x264 parameters for a fixed ``gop_frames`` GOP.

    The settings enforce ``keyint=min-keyint``, ``scenecut=0``, ``bframes=0``,
    ``repeat-headers=1``, and ``aud=1`` as required by the canonical video
    convention.
    """
    return (
        f"keyint={gop_frames}:min-keyint={gop_frames}:scenecut=0:bframes=0:repeat-headers=1:aud=1"
    )


def estimate_fps_from_log_times(log_times_ns: Sequence[int], *, topic: str) -> float:
    """The frame rate implied by message log times: 1e9 / median delta.

    Raises ``ValueError`` on fewer than two timestamps or a non-positive
    median delta (duplicate or non-increasing log times, e.g. stereo pairs on
    one topic or batch-granularity stamping) -- inferring a rate from such a
    stream would silently corrupt downstream timing, so it fails loudly with
    the topic named.
    """
    if len(log_times_ns) < 2:
        raise ValueError(
            f"camera topic {topic!r} has {len(log_times_ns)} timestamps; "
            "at least 2 are needed to infer a frame rate"
        )
    deltas_ns = [later - earlier for earlier, later in itertools.pairwise(log_times_ns)]
    median_delta_ns = statistics.median(deltas_ns)
    if median_delta_ns <= 0:
        raise ValueError(
            f"camera topic {topic!r} has a non-positive median inter-frame interval "
            "(duplicate or non-increasing log times); cannot infer a frame rate"
        )
    return 1e9 / median_delta_ns


def estimate_fps_from_streaming_log_times(log_times_ns: Iterable[int], *, topic: str) -> float:
    """Infer the same exact median rate without retaining all timestamps in RAM.

    Keep intervals in a temporary disk index with a bounded SQLite page cache.
    Selecting its middle one or two entries needs no in-memory sort. Only
    timestamps are consumed; callers can release each raw video batch in turn.
    """
    with (
        tempfile.TemporaryDirectory(prefix="hflow-video-fps-") as directory,
        closing(sqlite3.connect(Path(directory) / "intervals.sqlite")) as connection,
    ):
        connection.execute("PRAGMA cache_size = -1024")
        connection.execute("CREATE TABLE intervals (delta INTEGER NOT NULL)")
        connection.execute("CREATE INDEX ordered_intervals ON intervals (delta)")
        timestamp_count = 0

        def intervals() -> Iterator[tuple[int]]:
            nonlocal timestamp_count
            previous: int | None = None
            for timestamp in log_times_ns:
                timestamp_count += 1
                if previous is not None:
                    yield (timestamp - previous,)
                previous = timestamp

        connection.executemany("INSERT INTO intervals VALUES (?)", intervals())
        if timestamp_count < 2:
            # Reuse the existing error contract for empty/single-frame streams.
            return estimate_fps_from_log_times([0] * timestamp_count, topic=topic)
        interval_count = timestamp_count - 1
        middle = connection.execute(
            "SELECT delta FROM intervals ORDER BY delta LIMIT ? OFFSET ?",
            (2 - interval_count % 2, (interval_count - 1) // 2),
        ).fetchall()
        median_delta_ns = statistics.median(row[0] for row in middle)
        if median_delta_ns <= 0:
            return estimate_fps_from_log_times([0, 0], topic=topic)
        return 1e9 / median_delta_ns


def source_log_times_for_sampled_frames(
    source_timestamps_ns: Sequence[int] | np.ndarray,
    *,
    source_fps: float,
    sample_fps: float,
    start_s: float,
    frame_count: int,
) -> list[int]:
    """Log times of the source messages behind ``frame_count`` sampled frames.

    Anything that resamples a camera stream -- JPEG extraction for a model
    call, raw frames for a measurement -- needs the same answer: which
    recorded message does sampled frame *i* come from? One owner, because the
    reasoning is subtle in two places.

    The remux is index-preserving at constant source fps, and ffmpeg's ``fps``
    filter emits the frame VISIBLE at each output tick (the last frame
    at-or-before it), so output time *t* maps to source index
    ``floor(t * source_fps)``. The epsilon absorbs float error in an estimated
    rate. Times past the end of the stream clamp to the final message rather
    than raising: ffmpeg may emit one tick beyond it, and a frame it did
    produce must still be attributable.
    """
    if len(source_timestamps_ns) == 0:
        raise ValueError("cannot map sampled frames onto an empty source stream")
    if source_fps <= 0 or sample_fps <= 0:
        raise ValueError(
            f"source_fps and sample_fps must both be positive, got {source_fps} and {sample_fps}"
        )
    last_source_index = len(source_timestamps_ns) - 1
    log_times_ns: list[int] = []
    for frame_index in range(frame_count):
        output_time_s = start_s + frame_index / sample_fps
        source_index = min(int(output_time_s * source_fps + 1e-6), last_source_index)
        log_times_ns.append(int(source_timestamps_ns[source_index]))
    return log_times_ns


def encode_images_to_h264(
    images: Sequence[bytes],
    *,
    fps: float,
    gop_frames: int,
    crf: int = 23,
    input_codec: str = "mjpeg",
) -> list[AccessUnit]:
    """Encode a sequence of compressed images into H.264 access units.

    Pipes ``images`` (all the same codec, e.g. JPEG) into a single ffmpeg
    process (``-f image2pipe``) encoding with libx264 at ``crf``, ``yuv420p``,
    and :func:`canonical_x264_parameters`, raw Annex B out (``-f h264``), then
    splits the stream with :func:`split_annex_b_stream`.

    Guarantees (enforced, raising ``VideoEncodeError`` otherwise):
    - ``len(result) == len(images)`` -- one access unit per input frame, in
      input order (no B-frames means decode order == presentation order).
    - ``result[i].is_keyframe`` is True exactly when ``i % gop_frames == 0``.
    - Every keyframe access unit has ``has_parameter_sets == True``.
    - No access unit contains a B picture (checked from the slice headers).
    """
    concatenated_image_bytes = b"".join(images)
    x264_params = canonical_x264_parameters(gop_frames)

    def run_encode(extra_output_flags: list[str]) -> bytes:
        command: list[str] = [
            str(ffmpeg_path()),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "image2pipe",
            "-framerate",
            str(fps),
            "-c:v",
            input_codec,
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-x264-params",
            x264_params,
            *extra_output_flags,
            "-f",
            "h264",
            "-",
        ]
        # input= (not manual pipes) so stdin write and stdout drain interleave
        # without deadlocking on full pipe buffers.
        completed = subprocess.run(command, input=concatenated_image_bytes, capture_output=True)
        if completed.returncode != 0:
            raise VideoEncodeError(
                f"ffmpeg encode failed (exit {completed.returncode}): "
                f"{_stderr_tail(completed.stderr)}"
            )
        return completed.stdout

    try:
        access_units = split_annex_b_stream(run_encode([]))
        if len(access_units) != len(images):
            # Some frame-rate conversions duplicate or drop frames to hit CFR;
            # passthrough forces one output frame per input frame.
            access_units = split_annex_b_stream(run_encode(["-fps_mode", "passthrough"]))
    except ValueError as split_error:
        raise VideoEncodeError(
            f"encoder produced a stream without access-unit delimiters: {split_error}"
        ) from split_error

    _enforce_encode_guarantees(
        access_units, expected_frame_count=len(images), gop_frames=gop_frames
    )
    return access_units


def write_access_units_to_mp4(
    units: Iterable[bytes],
    *,
    fps: float,
    output: Path,
) -> Path:
    """Losslessly remux H.264 access units into an MP4 file (no re-encode).

    Streams complete access units to ffmpeg stdin and remuxes with
    ``ffmpeg -r {fps} -f h264 -i - -c:v copy -movflags +faststart``. The
    resulting file plays in anything; frame timing is constant-rate ``fps``
    (callers needing exact per-frame log times use the message timestamps).

    Raises ``ValueError`` on a stream carrying B-frames: a ``-c:v copy`` remux
    cannot represent the reorder tail, and the trailing frames are silently
    undecodable from the muxed file (#250 measured 303 samples in, 301
    decoded). Canonical video requires ``bframes=0`` (docs/FORMAT.md, "The
    H.264 bitstream constraints").
    """
    executable = ffmpeg_path()
    # Write to a unique sibling temp path and replace atomically: callers cache
    # on bare file existence, so the final path must never hold a partial MP4.
    # A fixed ``<output>.tmp`` name lets concurrent remuxes truncate/unlink the
    # same file; a per-call path keeps each ffmpeg process isolated.
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
    ) as temp_file:
        temporary_output = Path(temp_file.name)
    command: list[str] = [
        str(executable),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        # -r must precede -i: it declares the input frame rate to the h264
        # demuxer (raw Annex B carries no timing); after -i it would resample.
        "-r",
        str(fps),
        "-f",
        "h264",
        "-i",
        "-",
        "-c:v",
        "copy",
        "-movflags",
        "+faststart",
        # The .tmp suffix defeats extension sniffing; name the muxer explicitly.
        "-f",
        "mp4",
        str(temporary_output),
    ]
    try:
        # File-backed stderr cannot fill a pipe while stdin is being written,
        # and diagnostics never accumulate in Python memory.
        with tempfile.TemporaryFile() as stderr:
            process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=stderr
            )
            assert process.stdin is not None
            try:
                scanner = _PictureCodingScanner()
                pipe_broken = False
                for unit in units:
                    scanner.update(unit)
                    if not pipe_broken:
                        try:
                            process.stdin.write(unit)
                            process.stdin.flush()
                        except BrokenPipeError:
                            # Complete validation even if ffmpeg exits early:
                            # malformed input/B-frame errors retain precedence.
                            pipe_broken = True
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pipe_broken = True
                coding_types = scanner.result()
                if coding_types.b_picture_count:
                    raise ValueError(
                        f"cannot remux to MP4 without dropping frames: the stream carries "
                        f"{coding_types.b_picture_count} B picture(s) across "
                        f"{coding_types.picture_count} (reorder depth "
                        f"{coding_types.reorder_depth}); a -c:v copy remux drops the reorder "
                        f"tail, putting the last {coding_types.trailing_b_pictures} frame(s) at "
                        "risk (measured 301 of 303 in #250). Canonical video requires "
                        "bframes=0; re-encode upstream -- see docs/FORMAT.md item 4"
                    )
                returncode = process.wait()
                if returncode != 0 or pipe_broken:
                    stderr.seek(0, 2)
                    stderr.seek(max(0, stderr.tell() - 4 * _STDERR_TAIL_CHARACTER_LIMIT))
                    raise VideoEncodeError(
                        f"ffmpeg remux failed (exit {returncode}): {_stderr_tail(stderr.read())}"
                    )
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()
                with suppress(BrokenPipeError):
                    process.stdin.close()
        if not temporary_output.is_file() or temporary_output.stat().st_size == 0:
            raise VideoEncodeError(f"ffmpeg remux exited 0 but produced no output at {output}")
        temporary_output.replace(output)
    finally:
        temporary_output.unlink(missing_ok=True)
    return output


def _enforce_encode_guarantees(
    access_units: list[AccessUnit],
    *,
    expected_frame_count: int,
    gop_frames: int,
) -> None:
    """Raise ``VideoEncodeError`` if any docstring guarantee is violated."""
    if len(access_units) != expected_frame_count:
        raise VideoEncodeError(
            f"expected {expected_frame_count} access units (one per input frame), "
            f"got {len(access_units)}"
        )
    for unit_index, access_unit in enumerate(access_units):
        keyframe_expected = unit_index % gop_frames == 0
        if access_unit.is_keyframe != keyframe_expected:
            raise VideoEncodeError(
                f"access unit {unit_index}: is_keyframe={access_unit.is_keyframe}, "
                f"expected {keyframe_expected} (gop_frames={gop_frames})"
            )
        if keyframe_expected and not access_unit.has_parameter_sets:
            raise VideoEncodeError(
                f"keyframe access unit {unit_index} is missing SPS/PPS despite repeat-headers=1"
            )
    coding_scan = scan_picture_coding_types(b"".join(unit.data for unit in access_units))
    if coding_scan.b_picture_count > 0:
        # The joined scan proves the stream carries a B picture; the per-unit
        # rescan on this error path names the first offending access unit.
        for unit_index, access_unit in enumerate(access_units):
            unit_scan = scan_picture_coding_types(access_unit.data)
            if unit_scan.b_picture_count > 0:
                raise VideoEncodeError(
                    f"access unit {unit_index}: {unit_scan.b_picture_count} B picture(s), "
                    "canonical video requires no B-frames"
                )


def _stderr_tail(stderr: bytes) -> str:
    decoded_stderr = stderr.decode("utf-8", errors="replace").strip()
    if not decoded_stderr:
        return "(no stderr)"
    return decoded_stderr[-_STDERR_TAIL_CHARACTER_LIMIT:]
