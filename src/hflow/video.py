"""H.264 elementary-stream expertise: encode, split, and remux access units.

This module encodes the non-obvious knowledge behind the canonical video
convention (docs/ARCHITECTURE.md, "The episode container"):

- ``foxglove.CompressedVideo`` requires an Annex B byte stream, SPS/PPS
  attached to every keyframe, no B-frames, and one decodable access unit per
  message.
- x264 delivers exactly that with ``repeat-headers=1`` (parameter sets before
  every IDR), ``aud=1`` (access-unit delimiters, giving an unambiguous split
  point), ``bframes=0`` and ``scenecut=0`` (frame i in equals access unit i
  out, keyframes exactly every ``gop_frames``).
- A raw ``.h264`` stream carries no timestamps; remuxing to MP4 requires
  declaring the frame rate to the demuxer (``-r`` before ``-i``).

Everything here shells out to the pinned ffmpeg (``hflow.ffmpeg``).
"""

import itertools
import statistics
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from hflow.ffmpeg import ffmpeg_path


class VideoEncodeError(RuntimeError):
    """ffmpeg failed, or the encoder violated a documented output guarantee."""


_ANNEX_B_START_CODE = b"\x00\x00\x01"
_STDERR_TAIL_CHARACTER_LIMIT = 2000

# NAL unit types (H.264 spec, table 7-1).
_NAL_TYPE_IDR_SLICE = 5
_NAL_TYPE_SPS = 7
_NAL_TYPE_PPS = 8
_NAL_TYPE_ACCESS_UNIT_DELIMITER = 9
# VCL NAL types (coded slice data); an access unit must be delimited before these.
_VCL_NAL_TYPES = frozenset({1, 2, 3, 4, 5})
# Annex B AUD with primary_pic_type 7 (any I/P/B picture), followed by the
# required RBSP stop bit. It describes the following access unit without
# changing any slice data.
_ACCESS_UNIT_DELIMITER_NAL = b"\x00\x00\x00\x01\x09\xf0"


@dataclass(frozen=True)
class AccessUnit:
    """One decodable H.264 access unit (one video frame), Annex B format."""

    data: bytes
    is_keyframe: bool  # contains an IDR NAL (type 5)
    has_parameter_sets: bool  # contains SPS (type 7) and PPS (type 8) NALs


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


def source_log_times_for_sampled_frames(
    source_timestamps_ns: Sequence[int],
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
    if not source_timestamps_ns:
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
    process (``-f image2pipe``) encoding with libx264 at ``crf``,
    ``yuv420p``, ``keyint=min-keyint=gop_frames``, ``scenecut=0``,
    ``bframes=0``, ``repeat-headers=1``, ``aud=1``, raw Annex B out
    (``-f h264``), then splits the stream with :func:`split_annex_b_stream`.

    Guarantees (enforced, raising ``VideoEncodeError`` otherwise):
    - ``len(result) == len(images)`` -- one access unit per input frame, in
      input order (no B-frames means decode order == presentation order).
    - ``result[i].is_keyframe`` is True exactly when ``i % gop_frames == 0``.
    - Every keyframe access unit has ``has_parameter_sets == True``.
    """
    concatenated_image_bytes = b"".join(images)
    x264_params = (
        f"keyint={gop_frames}:min-keyint={gop_frames}:scenecut=0:bframes=0:repeat-headers=1:aud=1"
    )

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


def _annex_b_nal_offsets_and_types(stream: bytes) -> list[tuple[int, int]]:
    """Return ``(start offset, NAL type)`` pairs from an Annex B stream."""
    # Every (start_offset_including_start_code, nal_type) in stream order.
    # This naive byte scan is sound: inside a NAL payload the encoder inserts
    # an emulation-prevention byte (0x03) after any 00 00 pair that would
    # otherwise be followed by 00/01/02/03, so the sequence 00 00 01 can only
    # ever appear as a genuine start code.
    nal_offsets_and_types: list[tuple[int, int]] = []
    search_offset = 0
    while True:
        code_offset = stream.find(_ANNEX_B_START_CODE, search_offset)
        if code_offset == -1:
            break
        type_byte_offset = code_offset + len(_ANNEX_B_START_CODE)
        if type_byte_offset >= len(stream):
            break
        # A preceding zero byte makes this the 4-byte form 00 00 00 01; the
        # zero belongs to the start code and thus to this NAL's slice.
        has_four_byte_start_code = code_offset > 0 and stream[code_offset - 1] == 0
        nal_start_offset = code_offset - 1 if has_four_byte_start_code else code_offset
        nal_offsets_and_types.append((nal_start_offset, stream[type_byte_offset] & 0x1F))
        search_offset = type_byte_offset
    return nal_offsets_and_types


def ensure_access_unit_delimiter(access_unit: bytes) -> bytes:
    """Prepend an H.264 AUD when one access unit has none.

    The caller owns the access-unit boundary (for example, one
    ``foxglove.CompressedVideo`` message is one frame). This function does not
    re-encode or inspect slice headers: every existing byte is retained after
    the six-byte delimiter. A payload without coded slice data is rejected
    instead of being made to look valid by adding an AUD to garbage.
    """
    nal_types = [nal_type for _, nal_type in _annex_b_nal_offsets_and_types(access_unit)]
    if _NAL_TYPE_ACCESS_UNIT_DELIMITER in nal_types:
        return access_unit
    if not any(nal_type in _VCL_NAL_TYPES for nal_type in nal_types):
        raise ValueError("cannot insert an access-unit delimiter: no VCL NAL found")
    return _ACCESS_UNIT_DELIMITER_NAL + access_unit


def split_annex_b_stream(stream: bytes) -> list[AccessUnit]:
    """Split a raw Annex B H.264 byte stream into access units.

    Requires the stream to contain access-unit delimiter NALs (type 9), as
    produced by ``aud=1``; splits on them. Each unit's ``is_keyframe`` /
    ``has_parameter_sets`` are derived from the NAL types present (5 = IDR,
    7 = SPS, 8 = PPS). NAL types are read from the byte after each start code
    (``00 00 01`` / ``00 00 00 01``); emulation-prevention bytes cannot
    produce false start codes mid-NAL for the type scan.
    """
    nal_offsets_and_types = _annex_b_nal_offsets_and_types(stream)

    aud_nal_indices = [
        nal_index
        for nal_index, (_, nal_type) in enumerate(nal_offsets_and_types)
        if nal_type == _NAL_TYPE_ACCESS_UNIT_DELIMITER
    ]
    if not aud_nal_indices:
        raise ValueError(
            "no access-unit delimiter (AUD, NAL type 9) found; splitting requires "
            "a stream encoded with aud=1"
        )
    first_aud_nal_index = aud_nal_indices[0]
    vcl_nal_precedes_first_aud = any(
        nal_type in _VCL_NAL_TYPES for _, nal_type in nal_offsets_and_types[:first_aud_nal_index]
    )
    if vcl_nal_precedes_first_aud:
        raise ValueError(
            "a VCL NAL precedes the first access-unit delimiter; splitting requires "
            "a stream encoded with aud=1"
        )

    # A new access unit begins at each AUD (the AUD stays part of its unit);
    # any leading non-VCL NALs attach to the first unit.
    unit_start_offsets = [nal_offsets_and_types[i][0] for i in aud_nal_indices]
    unit_start_offsets[0] = 0
    unit_end_offsets = [*unit_start_offsets[1:], len(stream)]

    access_units: list[AccessUnit] = []
    # Both the NAL list and the unit boundaries are ascending in offset, so a
    # single pointer walk classifies every unit in O(total NALs) -- a per-unit
    # rescan would be quadratic and takes minutes on hour-long streams.
    nal_walk_index = 0
    for unit_start, unit_end in zip(unit_start_offsets, unit_end_offsets, strict=True):
        nal_types_in_unit: set[int] = set()
        while (
            nal_walk_index < len(nal_offsets_and_types)
            and nal_offsets_and_types[nal_walk_index][0] < unit_end
        ):
            nal_types_in_unit.add(nal_offsets_and_types[nal_walk_index][1])
            nal_walk_index += 1
        access_units.append(
            AccessUnit(
                data=stream[unit_start:unit_end],
                is_keyframe=_NAL_TYPE_IDR_SLICE in nal_types_in_unit,
                has_parameter_sets=(
                    _NAL_TYPE_SPS in nal_types_in_unit and _NAL_TYPE_PPS in nal_types_in_unit
                ),
            )
        )
    return access_units


def write_access_units_to_mp4(
    units: Iterable[bytes],
    *,
    fps: float,
    output: Path,
) -> Path:
    """Losslessly remux H.264 access units into an MP4 file (no re-encode).

    Concatenates the access units to a raw Annex B stream and remuxes with
    ``ffmpeg -r {fps} -f h264 -i - -c:v copy -movflags +faststart``. The
    resulting file plays in anything; frame timing is constant-rate ``fps``
    (callers needing exact per-frame log times use the message timestamps).
    """
    annex_b_stream = b"".join(units)
    # Write to a sibling temp path and replace atomically: callers cache on
    # bare file existence, so the final path must never hold a partial MP4.
    temporary_output = output.with_name(output.name + ".tmp")
    command: list[str] = [
        str(ffmpeg_path()),
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
        completed = subprocess.run(command, input=annex_b_stream, capture_output=True)
        if completed.returncode != 0:
            raise VideoEncodeError(
                f"ffmpeg remux failed (exit {completed.returncode}): "
                f"{_stderr_tail(completed.stderr)}"
            )
        if not temporary_output.is_file() or temporary_output.stat().st_size == 0:
            raise VideoEncodeError(f"ffmpeg remux exited 0 but produced no output at {output}")
    except BaseException:
        temporary_output.unlink(missing_ok=True)
        raise
    temporary_output.replace(output)
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


def _stderr_tail(stderr: bytes) -> str:
    decoded_stderr = stderr.decode("utf-8", errors="replace").strip()
    if not decoded_stderr:
        return "(no stderr)"
    return decoded_stderr[-_STDERR_TAIL_CHARACTER_LIMIT:]
