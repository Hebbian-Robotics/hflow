"""Tests for the H.264 elementary-stream pipeline (encode, split, remux)."""

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from foxglove_schemas_protobuf.CompressedVideo_pb2 import CompressedVideo
from mcap_protobuf.writer import Writer
from media_test_helpers import probe_video_stream, render_lavfi, run_ffmpeg

from hflow.episode import Episode
from hflow.video import (
    AccessUnit,
    PictureCodingScan,
    VideoEncodeError,
    _enforce_encode_guarantees,
    count_h264_pictures,
    encode_images_to_h264,
    scan_picture_coding_types,
    split_annex_b_stream,
    write_access_units_to_mp4,
)

pytestmark = pytest.mark.requires_system_ffmpeg

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
    render_lavfi(
        output_directory / f"%03d.{image_extension}",
        "testsrc2=size=320x240:rate=12",
        output_arguments=("-frames:v", str(frame_count), "-q:v", "2"),
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


@pytest.mark.parametrize("invalid_format", [False, True])
def test_episode_video_remuxes_batches_without_materializing_a_channel(
    encoded_units: list[AccessUnit],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_format: bool,
) -> None:
    source = tmp_path / "camera.mcap"
    with Writer(str(source)) as writer:
        for index, unit in enumerate(encoded_units):
            writer.write_message(
                "/camera",
                CompressedVideo(
                    format="h265" if invalid_format and index == FRAME_COUNT - 1 else "h264",
                    data=unit.data,
                ),
                log_time=round(index * 1e9 / FPS),
            )

    def refuse_materialization(*args: Any, **kwargs: Any) -> None:
        pytest.fail("video must not materialize the camera channel")

    monkeypatch.setattr(Episode, "channel", refuse_materialization)
    workdir = tmp_path / "video-cache"
    with Episode(source, workdir=workdir) as episode:
        original_batches = episode._reader.iter_batches

        def small_batches(*args: Any, **kwargs: Any) -> Iterator[Any]:
            kwargs["batch_max_messages"] = 3
            yield from original_batches(*args, **kwargs)

        monkeypatch.setattr(episode._reader, "iter_batches", small_batches)
        if invalid_format:
            with pytest.raises(ValueError, match="carries 'h265', expected 'h264'"):
                episode.video()
            assert not list(episode.workdir.glob("*.mp4"))
            assert not list(episode.workdir.glob(".*.tmp"))
            assert not episode._channel_data_by_id
            return
        output = episode.video()
        assert int(_ffprobe_video_stream_fields(output)["nb_read_frames"]) == FRAME_COUNT
        assert episode._video_fps["/camera"] == pytest.approx(FPS)
        assert not episode._channel_data_by_id

        def refuse_cached_read(*args: Any, **kwargs: Any) -> None:
            pytest.fail("a completed video with cached FPS must not reread the channel")

        with monkeypatch.context() as cached_patch:
            cached_patch.setattr(episode._reader, "iter_batches", refuse_cached_read)
            assert episode.video() == output

        # FPS alone is insufficient: a removed MP4 must be recreated.
        output.unlink()
        assert episode.video() == output
        assert int(_ffprobe_video_stream_fields(output)["nb_read_frames"]) == FRAME_COUNT

    with Episode(source, workdir=workdir) as reopened:
        # A fresh handle must recover FPS, but can reuse the completed MP4.
        monkeypatch.setattr(reopened, "iter_decoded_batches", refuse_cached_read)
        assert not reopened._video_fps
        assert reopened.video() == output
        assert reopened._video_fps["/camera"] == pytest.approx(FPS)


@pytest.fixture(scope="module")
def b_frame_stream(jpeg_frames: list[bytes]) -> bytes:
    """The same frames re-encoded with libx264 defaults (B-frames enabled)."""
    return run_ffmpeg(
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
        "bframes=3:b_adapt=0:aud=1",
        "-f",
        "h264",
        "-",
        stdin=b"".join(jpeg_frames),
    )


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


@pytest.mark.parametrize("split_units", [False, True])
def test_remux_refuses_a_b_frame_stream_naming_the_tail(
    b_frame_stream: bytes, tmp_path: Path, split_units: bool
) -> None:
    output_path = tmp_path / "bframe.mp4"
    units = (
        (unit.data for unit in split_annex_b_stream(b_frame_stream))
        if split_units
        else iter((b_frame_stream,))
    )

    with pytest.raises(ValueError, match="reorder depth") as error:
        write_access_units_to_mp4(units, fps=FPS, output=output_path)

    message = str(error.value)
    scan = scan_picture_coding_types(b_frame_stream)
    assert f"{scan.b_picture_count} B picture(s) across {scan.picture_count}" in message
    assert f"reorder depth {scan.reorder_depth}" in message
    assert f"last {scan.trailing_b_pictures} frame(s) at risk" in message
    assert "docs/FORMAT.md" in message
    # Even a refusal after streaming must remove the partial MP4.
    assert not output_path.exists()
    assert not list(tmp_path.glob(f".{output_path.name}.*.tmp"))


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
    return probe_video_stream(
        mp4_path, "codec_name", "profile", "nb_read_frames", count_frames=True
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


def _slice_header_rbsp(first_mb_in_slice: int, slice_type_code: int) -> bytes:
    """Pack first_mb_in_slice and slice_type as consecutive Exp-Golomb values."""

    def exp_golomb_bits(value: int) -> str:
        return "0" * ((value + 1).bit_length() - 1) + bin(value + 1)[2:]

    bits = exp_golomb_bits(first_mb_in_slice) + exp_golomb_bits(slice_type_code) + "1"
    bits += "0" * (-len(bits) % 8)
    return bytes(int(bits[index : index + 8], 2) for index in range(0, len(bits), 8))
