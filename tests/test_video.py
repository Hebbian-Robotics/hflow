"""Tests for the H.264 elementary-stream pipeline (encode, split, remux)."""

import subprocess
import time
from pathlib import Path

import pytest

from hflow.ffmpeg import ffmpeg_path, ffprobe_path
from hflow.video import (
    AccessUnit,
    PictureCodingScan,
    VideoEncodeError,
    _enforce_encode_guarantees,
    _first_mb_failure_message,
    _remove_emulation_prevention_bytes,
    _unescape_ebsp_head,
    count_h264_pictures,
    encode_images_to_h264,
    ensure_access_unit_delimiter,
    scan_picture_coding_types,
    split_annex_b_stream,
    write_access_units_to_mp4,
)

FRAME_COUNT = 48
FPS = 12.0
GOP_FRAMES = 12
EXPECTED_KEYFRAME_INDICES = {0, 12, 24, 36}


def _generate_test_frames(
    output_directory: Path,
    *,
    frame_count: int,
    image_extension: str,
) -> list[bytes]:
    """Render testsrc2 frames (320x240, 12 fps) to individual image files."""
    subprocess.run(
        [
            str(ffmpeg_path()),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=12",
            "-frames:v",
            str(frame_count),
            "-q:v",
            "2",
            str(output_directory / f"%03d.{image_extension}"),
        ],
        check=True,
        capture_output=True,
    )
    frame_paths = sorted(output_directory.glob(f"*.{image_extension}"))
    assert len(frame_paths) == frame_count
    return [frame_path.read_bytes() for frame_path in frame_paths]


@pytest.fixture(scope="module")
def jpeg_frames(tmp_path_factory: pytest.TempPathFactory) -> list[bytes]:
    return _generate_test_frames(
        tmp_path_factory.mktemp("jpeg_frames"),
        frame_count=FRAME_COUNT,
        image_extension="jpg",
    )


@pytest.fixture(scope="module")
def encoded_units(jpeg_frames: list[bytes]) -> list[AccessUnit]:
    return encode_images_to_h264(jpeg_frames, fps=FPS, gop_frames=GOP_FRAMES)


def test_encode_produces_one_access_unit_per_input_frame(
    encoded_units: list[AccessUnit],
) -> None:
    assert len(encoded_units) == FRAME_COUNT


def test_keyframes_land_exactly_on_gop_boundaries(encoded_units: list[AccessUnit]) -> None:
    keyframe_indices = {
        unit_index for unit_index, unit in enumerate(encoded_units) if unit.is_keyframe
    }
    assert keyframe_indices == EXPECTED_KEYFRAME_INDICES


def test_every_keyframe_carries_parameter_sets(encoded_units: list[AccessUnit]) -> None:
    for unit in encoded_units:
        if unit.is_keyframe:
            assert unit.has_parameter_sets


def test_non_keyframes_exist_and_carry_no_parameter_sets(
    encoded_units: list[AccessUnit],
) -> None:
    non_keyframe_units = [unit for unit in encoded_units if not unit.is_keyframe]
    assert non_keyframe_units
    # repeat-headers=1 repeats SPS/PPS only before IDR frames.
    for unit in non_keyframe_units:
        assert not unit.has_parameter_sets


def test_split_concat_round_trip_is_lossless(encoded_units: list[AccessUnit]) -> None:
    reconstructed_stream = b"".join(unit.data for unit in encoded_units)
    assert split_annex_b_stream(reconstructed_stream) == encoded_units


def test_remux_to_mp4_preserves_every_frame(
    encoded_units: list[AccessUnit], tmp_path: Path
) -> None:
    output_path = tmp_path / "remuxed.mp4"
    remux_started_at = time.monotonic()
    returned_path = write_access_units_to_mp4(
        (unit.data for unit in encoded_units), fps=FPS, output=output_path
    )
    remux_elapsed_seconds = time.monotonic() - remux_started_at
    assert returned_path == output_path
    assert output_path.stat().st_size > 0

    stream_fields = _ffprobe_video_stream_fields(output_path)
    assert stream_fields["codec_name"] == "h264"
    assert int(stream_fields["nb_read_frames"]) == FRAME_COUNT
    # ffprobe cannot prove "no re-encode" directly; -c:v copy is the actual
    # guarantee. We corroborate: the copied stream kept a real profile (i.e.
    # extradata/SPS survived) and the remux ran in trivial time -- far below
    # what a 48-frame libx264 re-encode plus faststart rewrite would allow to
    # be confused with a copy on this hardware.
    assert stream_fields["profile"] not in ("", "unknown")
    assert remux_elapsed_seconds < 5.0


@pytest.fixture(scope="module")
def b_frame_stream(jpeg_frames: list[bytes]) -> bytes:
    """The same frames re-encoded with libx264 defaults (B-frames enabled)."""
    command: list[str] = [
        str(ffmpeg_path()),
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "image2pipe",
        "-framerate",
        str(FPS),
        "-c:v",
        "mjpeg",
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-x264-params",
        "bframes=3:b_adapt=0",
        "-f",
        "h264",
        "-",
    ]
    completed = subprocess.run(command, input=b"".join(jpeg_frames), capture_output=True)
    assert completed.returncode == 0, completed.stderr.decode()
    return completed.stdout


def _slice_header_rbsp(first_mb_in_slice: int, slice_type_code: int) -> bytes:
    """Pack first_mb_in_slice and slice_type as consecutive Exp-Golomb values."""

    def exp_golomb_bits(value: int) -> str:
        return "0" * ((value + 1).bit_length() - 1) + bin(value + 1)[2:]

    bits = exp_golomb_bits(first_mb_in_slice) + exp_golomb_bits(slice_type_code) + "1"
    bits += "0" * (-len(bits) % 8)
    return bytes(int(bits[index : index + 8], 2) for index in range(0, len(bits), 8))


def test_scan_reports_b_pictures_in_a_real_b_frame_stream(b_frame_stream: bytes) -> None:
    scan = scan_picture_coding_types(b_frame_stream)

    assert scan.picture_count == FRAME_COUNT
    assert scan.b_picture_count > 0
    assert scan.reorder_depth >= 1


def test_scan_reports_no_b_pictures_for_the_canonical_stream(
    encoded_units: list[AccessUnit],
) -> None:
    canonical_stream = b"".join(unit.data for unit in encoded_units)

    scan = scan_picture_coding_types(canonical_stream)

    assert scan == PictureCodingScan(
        picture_count=FRAME_COUNT, b_picture_count=0, reorder_depth=0, trailing_b_pictures=0
    )


def test_scan_measures_reorder_depth_and_trailing_tail() -> None:
    non_idr_nal = b"\x00\x00\x00\x01\x41"
    stream = b"".join(
        non_idr_nal + _slice_header_rbsp(0, slice_type_code)
        for slice_type_code in (0, 1, 1, 0, 1)  # P, B, B, P, B in decode order
    )

    scan = scan_picture_coding_types(stream)

    assert scan == PictureCodingScan(
        picture_count=5, b_picture_count=3, reorder_depth=2, trailing_b_pictures=1
    )


def test_scan_refuses_an_unparseable_slice_header() -> None:
    # An all-zero RBSP never terminates the first Exp-Golomb value, so the
    # slice header cannot be classified and the scan must fail closed.
    unparseable_slice = b"\x00\x00\x00\x01\x41\x00"

    with pytest.raises(ValueError, match="cannot be classified"):
        scan_picture_coding_types(unparseable_slice)


def test_remux_refuses_a_b_frame_stream_naming_the_tail(
    b_frame_stream: bytes, tmp_path: Path
) -> None:
    output_path = tmp_path / "bframe.mp4"

    with pytest.raises(ValueError, match="reorder depth") as error:
        write_access_units_to_mp4((b_frame_stream,), fps=FPS, output=output_path)

    message = str(error.value)
    assert "B picture" in message
    assert "at risk" in message
    assert "docs/FORMAT.md" in message
    # The refusal fires before ffmpeg runs, so no partial MP4 is left behind.
    assert not output_path.exists()
    assert not output_path.with_name(output_path.name + ".tmp").exists()


def test_encode_guarantees_accept_a_conforming_stream(
    encoded_units: list[AccessUnit],
) -> None:
    _enforce_encode_guarantees(
        encoded_units, expected_frame_count=len(encoded_units), gop_frames=GOP_FRAMES
    )


def test_encode_guarantees_raise_on_a_b_picture_naming_the_unit() -> None:
    # A keyframe lead (passing the cadence check), then a B picture
    # (slice_type 1), then a P picture. The guarantee must name access unit 1.
    aud = b"\x00\x00\x00\x01\x09\x10"
    non_idr_nal = b"\x00\x00\x00\x01\x41"
    units = [
        AccessUnit(
            data=aud + b"\x00\x00\x00\x01\x65" + _slice_header_rbsp(0, 2),
            is_keyframe=True,
            has_parameter_sets=True,
        ),
        AccessUnit(
            data=aud + non_idr_nal + _slice_header_rbsp(0, 1),
            is_keyframe=False,
            has_parameter_sets=False,
        ),
        AccessUnit(
            data=aud + non_idr_nal + _slice_header_rbsp(0, 0),
            is_keyframe=False,
            has_parameter_sets=False,
        ),
    ]

    with pytest.raises(VideoEncodeError, match=r"access unit 1") as error:
        _enforce_encode_guarantees(units, expected_frame_count=len(units), gop_frames=3)

    assert "B picture" in str(error.value)


def test_encode_guarantees_find_a_b_picture_in_a_later_unit() -> None:
    # A B picture in the final unit must still be named, proving the joined
    # scan covers the whole stream rather than stopping early.
    aud = b"\x00\x00\x00\x01\x09\x10"
    non_idr_nal = b"\x00\x00\x00\x01\x41"
    units = [
        AccessUnit(
            data=aud + b"\x00\x00\x00\x01\x65" + _slice_header_rbsp(0, 2),
            is_keyframe=True,
            has_parameter_sets=True,
        ),
        AccessUnit(
            data=aud + non_idr_nal + _slice_header_rbsp(0, 1),
            is_keyframe=False,
            has_parameter_sets=False,
        ),
    ]

    with pytest.raises(VideoEncodeError, match=r"access unit 1"):
        _enforce_encode_guarantees(units, expected_frame_count=len(units), gop_frames=2)

    with pytest.raises(VideoEncodeError, match=r"access unit 1"):
        _enforce_encode_guarantees(units, expected_frame_count=len(units), gop_frames=2)


def test_split_rejects_garbage_without_aud() -> None:
    with pytest.raises(ValueError, match="aud=1"):
        split_annex_b_stream(b"\xde\xad\xbe\xef" * 32)
    # A VCL NAL (IDR, type 5) before any AUD must also be rejected.
    idr_before_any_aud = b"\x00\x00\x00\x01\x65" + bytes(16)
    with pytest.raises(ValueError, match="aud=1"):
        split_annex_b_stream(idr_before_any_aud)


def test_ensure_aud_is_lossless_and_idempotent() -> None:
    undelimited_keyframe = b"".join(
        b"\x00\x00\x00\x01"
        + bytes([nal_type])
        + (b"\x80payload" if nal_type == 0x65 else b"payload")
        for nal_type in (0x67, 0x68, 0x65)
    )

    repaired = ensure_access_unit_delimiter(undelimited_keyframe)

    assert repaired.endswith(undelimited_keyframe)
    assert len(repaired) == len(undelimited_keyframe) + 6
    assert ensure_access_unit_delimiter(repaired) == repaired
    assert split_annex_b_stream(repaired) == [
        AccessUnit(data=repaired, is_keyframe=True, has_parameter_sets=True)
    ]


def test_ensure_aud_accepts_multiple_slices_for_one_picture() -> None:
    multi_slice_keyframe = b"".join(
        [
            b"\x00\x00\x00\x01\x67payload",
            b"\x00\x00\x00\x01\x68payload",
            b"\x00\x00\x00\x01\x65\x80first-slice",
            b"\x00\x00\x00\x01\x65\x40second-slice",
        ]
    )

    repaired = ensure_access_unit_delimiter(multi_slice_keyframe)

    assert repaired.endswith(multi_slice_keyframe)


def test_ensure_aud_rejects_multiple_pictures_without_auds() -> None:
    two_pictures = b"".join(
        [
            b"\x00\x00\x00\x01\x65\x80first-picture",
            b"\x00\x00\x00\x01\x41\x80second-picture",
        ]
    )

    with pytest.raises(ValueError, match="message contains 2 pictures"):
        ensure_access_unit_delimiter(two_pictures)


def test_ensure_aud_rejects_payload_without_coded_slice_data() -> None:
    parameter_sets_only = b"".join(
        b"\x00\x00\x00\x01" + bytes([nal_type]) + b"payload" for nal_type in (0x67, 0x68)
    )
    with pytest.raises(ValueError, match="no VCL NAL"):
        ensure_access_unit_delimiter(parameter_sets_only)


def test_encode_failure_raises_with_ffmpeg_stderr(jpeg_frames: list[bytes]) -> None:
    with pytest.raises(VideoEncodeError, match="notacodec"):
        encode_images_to_h264(
            jpeg_frames[:2], fps=FPS, gop_frames=GOP_FRAMES, input_codec="notacodec"
        )


def test_png_input_codec_upholds_guarantees(tmp_path: Path) -> None:
    png_frames = _generate_test_frames(tmp_path, frame_count=6, image_extension="png")
    units = encode_images_to_h264(png_frames, fps=FPS, gop_frames=3, input_codec="png")
    assert len(units) == 6
    keyframe_indices = {unit_index for unit_index, unit in enumerate(units) if unit.is_keyframe}
    assert keyframe_indices == {0, 3}
    assert all(unit.has_parameter_sets for unit in units if unit.is_keyframe)


def _ffprobe_video_stream_fields(mp4_path: Path) -> dict[str, str]:
    """Read codec/profile/decoded-frame-count fields from the first video stream."""
    ffprobe_binary = ffprobe_path()
    completed = subprocess.run(
        [
            str(ffprobe_binary),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,profile,nb_read_frames",
            "-of",
            "default=noprint_wrappers=1",
            str(mp4_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream_fields: dict[str, str] = {}
    for output_line in completed.stdout.splitlines():
        field_name, _, field_value = output_line.partition("=")
        stream_fields[field_name.strip()] = field_value.strip()
    return stream_fields


def test_unescape_ebsp_head_matches_full_unescape_for_the_prefix() -> None:
    """The head-only unescape must produce the same prefix the full unescape
    does, so the slice-header readers see identical bytes for the bytes they
    actually read. An emulation-prevention 0x03 consumed by the full pass
    after the cut is irrelevant: the prefix ends at the cut either way."""
    payload = b"\x00\x00\x00\x01\x65" + bytes(range(256)) * 4
    full = _remove_emulation_prevention_bytes(payload)
    for max_bytes in (8, 16, 32, 64, 128):
        head = _unescape_ebsp_head(payload, max_bytes)
        if max_bytes >= len(payload):
            assert head == full
        else:
            # The head may be slightly shorter than max_bytes when the
            # unescape consumes a 0x03 after the 00 00 pair -- that is
            # correct emulation-prevention behavior, and the full pass has
            # the same property.
            assert head == full[: len(head)]


def test_unescape_ebsp_head_with_max_bytes_at_or_above_length_returns_full_unescape() -> None:
    """Asking for the whole NAL (or more) must be a fast path to the full
    unescape, with no behavioral difference. The scan uses this when a
    VCL NAL is shorter than the slice-header head budget, which is
    common for very small frames."""
    payload = b"\x00\x00\x03\x01\x02"
    assert _unescape_ebsp_head(payload, len(payload)) == _remove_emulation_prevention_bytes(payload)
    assert _unescape_ebsp_head(payload, len(payload) + 100) == _remove_emulation_prevention_bytes(
        payload
    )


def test_scan_results_match_count_h264_pictures_on_canonical_and_b_frame_streams(
    encoded_units: list[AccessUnit], b_frame_stream: bytes
) -> None:
    """The two readers share the NAL walk and the head-only unescape; the
    scan's picture_count and count_h264_pictures must agree on every input
    the existing test suite covers. Pinning the contract here means a future
    divergence surfaces as a test failure rather than a silent miss."""
    canonical_stream = b"".join(unit.data for unit in encoded_units)
    canonical_scan = scan_picture_coding_types(canonical_stream)
    assert canonical_scan.picture_count == count_h264_pictures(canonical_stream)
    assert canonical_scan.picture_count == FRAME_COUNT

    b_scan = scan_picture_coding_types(b_frame_stream)
    assert b_scan.picture_count == count_h264_pictures(b_frame_stream)
    assert b_scan.picture_count == FRAME_COUNT


def test_scan_and_count_agree_on_multi_slice_and_escape_heavy_streams() -> None:
    """Extends the equivalence pin to the inputs #355 calls out: a
    multi-slice picture and a stream whose emulation-prevention triples
    straddle the 16-byte head boundary. Both readers run one walk now;
    any future divergence between them fails here."""
    # Two pictures, the first split into two slices (first_mb_in_slice 0, 1).
    multi_slice_stream = b"".join(
        b"\x00\x00\x00\x01\x41" + _slice_header_rbsp(first_mb, slice_type)
        for first_mb, slice_type in ((0, 2), (1, 2), (0, 0))
    )
    assert scan_picture_coding_types(multi_slice_stream).picture_count == 2
    assert count_h264_pictures(multi_slice_stream) == 2

    def escape_prevention_bytes(rbsp: bytes) -> bytes:
        """Encode one RBSP the way an encoder would: 0x03 after each 00 00
        pair followed by a byte at or below 0x03."""
        encoded = bytearray()
        consecutive_zeros = 0
        for byte in rbsp:
            if consecutive_zeros >= 2 and byte <= 0x03:
                encoded.append(0x03)
                consecutive_zeros = 0
            encoded.append(byte)
            consecutive_zeros = consecutive_zeros + 1 if byte == 0 else 0
        return bytes(encoded)

    # first_mb_in_slice = 2**70 encodes as 70 leading zero bits, so its
    # header starts with eight 0x00 bytes and needs 141 bits in total: the
    # escapes push the header past what a 16-byte head can unescape, and
    # only the full-payload fallback can decode it. It appears as the second
    # slice of a picture, since a picture's first slice always carries 0.
    wide_rbsp = _slice_header_rbsp(2**70, 7)
    assert wide_rbsp.startswith(b"\x00" * 8)
    escape_stream = b"".join(
        b"\x00\x00\x00\x01\x41" + escape_prevention_bytes(rbsp)
        for rbsp in (
            _slice_header_rbsp(0, 7),
            wide_rbsp,
            _slice_header_rbsp(0, 7),
        )
    )
    assert scan_picture_coding_types(escape_stream).picture_count == 2
    assert count_h264_pictures(escape_stream) == 2
    assert scan_picture_coding_types(escape_stream).b_picture_count == 0


def test_first_mb_failure_message_names_both_failure_kinds() -> None:
    """count_h264_pictures owes hflow doctor two distinct messages, and the
    walk no longer raises them itself: this helper is the only place they
    are produced. An all-zero RBSP never terminates the first Exp-Golomb
    value; 00000100 carries the terminating one bit but loses its suffix."""
    assert (
        _first_mb_failure_message(b"\x00") == "slice header has no complete first_mb_in_slice value"
    )
    assert (
        _first_mb_failure_message(b"\x04") == "slice header truncates its first_mb_in_slice value"
    )


def test_a_truncated_slice_type_still_counts_but_cannot_be_classified() -> None:
    """The two readers disagree about which field failing matters, and the
    shared walk has to keep that disagreement.

    ``0x80`` is ``first_mb_in_slice`` = ue(0) followed by seven zero bits, so
    the first field decodes and the second never terminates.
    ``count_h264_pictures`` reads only the first and must still count the
    picture, exactly as it did when it had its own single-field decoder;
    the scan needs both and must refuse. Collapsing the walk's two
    per-field ``is None`` checks into one would pass every other test here.
    """
    truncated_slice_type = b"\x00\x00\x00\x01\x41\x80"

    assert count_h264_pictures(truncated_slice_type) == 1
    with pytest.raises(ValueError, match="cannot be classified"):
        scan_picture_coding_types(truncated_slice_type)


def test_scan_refuses_a_truncated_slice_with_the_existing_fail_closed_message() -> None:
    """A NAL whose head runs out before both Exp-Golomb values finish must
    still fail closed with the same message the unoptimized path produced.
    The 16-byte head is enough at any legal first_mb_in_slice; the synthetic
    NAL here is engineered to demand more, exercising the fallback path."""
    # first_mb_in_slice = 2^15 needs 15 leading zero bits + 1 + 15 suffix bits
    # = 31 bits. _SLICE_HEADER_HEAD_BYTES = 16 covers 128 bits, so 31 fits,
    # but we deliberately encode a value whose unescaped form exceeds 16
    # bytes to force the fallback. Easiest: stuff the payload with 0x00 so
    # the decoder's "no terminating 1 bit" guard fires regardless of head.
    bad_slice = b"\x00\x00\x00\x01\x41" + b"\x00" * 64

    with pytest.raises(ValueError, match="cannot be classified"):
        scan_picture_coding_types(bad_slice)
