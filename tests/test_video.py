"""Tests for the H.264 elementary-stream pipeline (encode, split, remux)."""

import subprocess
import time
from pathlib import Path

import pytest

from hflow.ffmpeg import ffmpeg_path, ffprobe_path
from hflow.video import (
    AccessUnit,
    VideoEncodeError,
    encode_images_to_h264,
    ensure_access_unit_delimiter,
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


def test_split_rejects_garbage_without_aud() -> None:
    with pytest.raises(ValueError, match="aud=1"):
        split_annex_b_stream(b"\xde\xad\xbe\xef" * 32)
    # A VCL NAL (IDR, type 5) before any AUD must also be rejected.
    idr_before_any_aud = b"\x00\x00\x00\x01\x65" + bytes(16)
    with pytest.raises(ValueError, match="aud=1"):
        split_annex_b_stream(idr_before_any_aud)


def test_ensure_aud_is_lossless_and_idempotent() -> None:
    undelimited_keyframe = b"".join(
        b"\x00\x00\x00\x01" + bytes([nal_type]) + b"payload" for nal_type in (0x67, 0x68, 0x65)
    )

    repaired = ensure_access_unit_delimiter(undelimited_keyframe)

    assert repaired.endswith(undelimited_keyframe)
    assert len(repaired) == len(undelimited_keyframe) + 6
    assert ensure_access_unit_delimiter(repaired) == repaired
    assert split_annex_b_stream(repaired) == [
        AccessUnit(data=repaired, is_keyframe=True, has_parameter_sets=True)
    ]


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
