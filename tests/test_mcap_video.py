from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
from mcap.writer import Writer
from mcap_test_helpers import write_compressed_video_mcap

from hflow.mcap_video import McapMediaError, export_mcap_camera
from hflow.media import VideoLimits

CAMERA_START_NS = 1_700_000_000_000_000_000
# Real MCAP capture can deliver adjacent frames in bursts without exceeding
# the supported average frame rate.
FRAME_TIMES_NS = (0, 147_079, 5_000_147_079, 5_500_147_079)
SOURCE_DURATION_MILLIS = 6001
LIMITS = VideoLimits(maximum_frame_pixels=256, maximum_frames_per_second=60, timeout_seconds=20)


def _h264_packets(width: int = 16) -> list[bytes]:
    encoder = av.CodecContext.create("libx264", "w")
    encoder.width = encoder.height = width
    encoder.pix_fmt = "yuv420p"
    encoder.time_base = Fraction(1, 30)
    encoder.thread_count = 1
    encoder.options = {
        "preset": "ultrafast",
        "tune": "zerolatency",
        "x264-params": "keyint=2:min-keyint=2:scenecut=0:repeat-headers=1:aud=1:bframes=0",
    }
    packets = []
    for index in range(len(FRAME_TIMES_NS)):
        frame = av.VideoFrame.from_ndarray(np.full((width, width, 3), index * 40, dtype=np.uint8))
        frame.pts = index
        packets.extend(bytes(packet) for packet in encoder.encode(frame))
    packets.extend(bytes(packet) for packet in encoder.encode(None))
    return packets


def _omit_h264_headers(packet: bytes, nal_types: frozenset[int]) -> bytes:
    return b"".join(
        b"\x00\x00\x00\x01" + nal
        for nal in re.split(b"\x00\x00(?:\x00)?\x01", packet)
        if nal and nal[0] & 31 not in nal_types
    )


def _write_h264_mcap(
    path: Path,
    topics: tuple[str, ...] = ("/camera/front",),
    packets: list[bytes] | None = None,
    frame_times_ns: tuple[int, ...] = FRAME_TIMES_NS,
) -> list[bytes]:
    packets = _h264_packets() if packets is None else packets
    write_compressed_video_mcap(
        path,
        [
            (topic, CAMERA_START_NS + timestamp, packet)
            for topic in topics
            for timestamp, packet in zip(frame_times_ns, packets, strict=True)
        ],
        frame_id_by_topic=dict.fromkeys(topics, ""),
    )
    return packets


@pytest.mark.parametrize("repeat_parameter_sets", [True, False])
def test_mcap_h264_preserves_frames_gaps_timestamps_and_original_bytes(
    tmp_path: Path, repeat_parameter_sets: bool
) -> None:
    source = tmp_path / "source.mcap"
    packets = _h264_packets()
    if not repeat_parameter_sets:
        packets = [
            packets[0],
            *(_omit_h264_headers(packet, frozenset({7, 8})) for packet in packets[1:]),
        ]
    _write_h264_mcap(source, packets=packets)
    original_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    result = export_mcap_camera(source, tmp_path / "camera.mp4", limits=LIMITS)

    assert result.camera_topic == "/camera/front"
    assert result.start_timestamp_ns == CAMERA_START_NS
    assert result.duration_millis == SOURCE_DURATION_MILLIS
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_digest
    with av.open(result.video_path) as video:
        frames = list(video.decode(video=0))
        for frame, timestamp in zip(frames, FRAME_TIMES_NS, strict=True):
            assert frame.pts is not None and frame.time_base is not None
            assert frame.pts * frame.time_base == Fraction((timestamp + 500) // 1000, 1_000_000)
    decoder = av.CodecContext.create("h264", "r")
    expected_frames = [frame for packet in packets for frame in decoder.decode(av.Packet(packet))]
    for actual, expected in zip(frames, expected_frames, strict=True):
        np.testing.assert_array_equal(
            actual.to_ndarray(format="rgb24"), expected.to_ndarray(format="rgb24")
        )
    assert not list(tmp_path.glob(".camera-export-*"))


def test_mcap_requires_initial_h264_codec_configuration(tmp_path: Path) -> None:
    source = tmp_path / "source.mcap"
    packets = _h264_packets()
    packets[0] = _omit_h264_headers(packets[0], frozenset({7, 8}))
    _write_h264_mcap(source, packets=packets)
    output = tmp_path / "camera.mp4"

    with pytest.raises(McapMediaError):
        export_mcap_camera(source, output, limits=LIMITS)

    assert not output.exists()


def test_multicamera_mcap_requires_an_exact_selection(tmp_path: Path) -> None:
    source = tmp_path / "source.mcap"
    _write_h264_mcap(source, ("/camera/front", "/camera/back"))
    output = tmp_path / "camera.mp4"
    for camera_topic in (None, "front", "/unknown"):
        with pytest.raises(McapMediaError):
            export_mcap_camera(source, output, camera_topic=camera_topic, limits=LIMITS)
        assert not output.exists()
    result = export_mcap_camera(source, output, camera_topic="/camera/back", limits=LIMITS)
    assert result.camera_topic == "/camera/back"
    assert result.duration_millis == SOURCE_DURATION_MILLIS


@pytest.mark.parametrize("repeat_picture_parameter_set", [True, False])
def test_mcap_rejects_a_camera_resolution_change(
    tmp_path: Path, repeat_picture_parameter_set: bool
) -> None:
    source = tmp_path / "source.mcap"
    changed_packets = _h264_packets(32)[:2]
    if not repeat_picture_parameter_set:
        changed_packets[0] = _omit_h264_headers(changed_packets[0], frozenset({8}))
    _write_h264_mcap(source, packets=_h264_packets()[:2] + changed_packets)
    destination = tmp_path / "camera.mp4"

    with pytest.raises(McapMediaError):
        export_mcap_camera(
            source, destination, limits=replace(LIMITS, maximum_frame_pixels=32 * 32)
        )

    assert not destination.exists()


@pytest.mark.parametrize("compressed", [False, True])
def test_image_mcap_preserves_pixels_and_ignores_unselected_cameras(
    tmp_path: Path, compressed: bool
) -> None:
    source = tmp_path / "raw.mcap"
    pixels = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    payload: dict[str, object] = {
        "width": 16,
        "height": 16,
        "step": 48,
        "encoding": "rgb8",
        "data": list(pixels.tobytes()),
    }
    schema_name = "foxglove.RawImage"
    if compressed:
        encoder = av.CodecContext.create("png", "w")
        encoder.width = encoder.height = 16
        encoder.pix_fmt = "rgb24"
        encoded = encoder.encode(av.VideoFrame.from_ndarray(pixels))[0]
        payload = {"format": "png", "data": list(bytes(encoded))}
        schema_name = "foxglove.CompressedImage"
    with source.open("wb") as output:
        writer = Writer(output)
        writer.start()
        schema_id = writer.register_schema(schema_name, "jsonschema", b"{}")
        selected_channel = writer.register_channel("/rgb", "json", schema_id)
        other_channel = writer.register_channel("/unused", "json", schema_id)
        for timestamp in FRAME_TIMES_NS:
            writer.add_message(
                selected_channel,
                CAMERA_START_NS + timestamp,
                json.dumps(payload).encode(),
                CAMERA_START_NS + timestamp,
            )
        writer.add_message(
            other_channel, CAMERA_START_NS, b'{"data":"unsupported"}', CAMERA_START_NS
        )
        writer.finish()
    result = export_mcap_camera(source, tmp_path / "raw.mp4", camera_topic="/rgb", limits=LIMITS)
    assert result.duration_millis == SOURCE_DURATION_MILLIS
    assert result.start_timestamp_ns == CAMERA_START_NS
    with av.open(result.video_path) as video:
        frames = list(video.decode(video=0))
    assert len(frames) == len(FRAME_TIMES_NS)
    for frame, timestamp in zip(frames, FRAME_TIMES_NS, strict=True):
        assert frame.pts is not None and frame.time_base is not None
        assert frame.pts * frame.time_base == Fraction((timestamp + 500) // 1000, 1_000_000)
        np.testing.assert_array_equal(frame.to_ndarray(format="rgb24"), pixels)


@pytest.mark.parametrize(
    ("limits", "frame_times_ns"),
    (
        (VideoLimits(maximum_frame_pixels=1, timeout_seconds=20), FRAME_TIMES_NS),
        (VideoLimits(maximum_duration_seconds=2, timeout_seconds=20), FRAME_TIMES_NS),
        (VideoLimits(maximum_frames_per_second=0.5, timeout_seconds=20), FRAME_TIMES_NS),
        (LIMITS, (0, 100, 1_000_000_000, 2_000_000_000)),
        (LIMITS, (0, 2_147_483_648_000, 2_148_483_648_000, 2_149_483_648_000)),
    ),
)
def test_rejected_mcap_leaves_original_without_publishing_output(
    tmp_path: Path, limits: VideoLimits, frame_times_ns: tuple[int, ...]
) -> None:
    source = tmp_path / "source.mcap"
    _write_h264_mcap(source, frame_times_ns=frame_times_ns)
    original_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = tmp_path / "existing.mp4"
    with pytest.raises(McapMediaError):
        export_mcap_camera(source, destination, limits=limits)

    assert not destination.exists()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_digest
    assert not list(tmp_path.glob(".camera-export-*"))


def test_camera_export_preserves_an_existing_destination(tmp_path: Path) -> None:
    source = tmp_path / "source.mcap"
    _write_h264_mcap(source)
    destination = tmp_path / "existing.mp4"
    destination.write_bytes(b"existing result")
    with pytest.raises(FileExistsError):
        export_mcap_camera(source, destination, limits=LIMITS)
    assert destination.read_bytes() == b"existing result"
