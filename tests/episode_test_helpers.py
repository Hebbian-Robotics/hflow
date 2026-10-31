"""Shared helper for tests that need a canonical episode from a synthetic spec."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hflow.testing import SyntheticEpisodeSpec, synthesize_episode
from hflow.transform import TransformConfig, write_canonical_episode


def synthesize_canonical_episode(
    directory: Path,
    spec: SyntheticEpisodeSpec,
    *,
    config: TransformConfig | None = None,
    stem: str = "episode",
) -> Path:
    """Synthesize ``spec`` as ``<stem>.mcap`` in ``directory`` and canonicalize it.

    Returns the canonical path, ``<stem>.canonical.mcap`` beside the source.
    ``config=None`` means the default :class:`TransformConfig`.
    """
    source = synthesize_episode(directory / f"{stem}.mcap", spec)
    canonical = directory / f"{stem}.canonical.mcap"
    write_canonical_episode(source, canonical, config)
    return canonical


NUMBERED_VIDEO_TOPIC = "/camera/cam0/video"
NUMBERED_VIDEO_START_NS = 1_760_000_000_000_000_000


def write_numbered_h264_episodes(directory: Path) -> dict[int | None, Path]:
    """Canonical MP4 inputs with pixels identifying their original source index.

    Keyed by the IDR left out of the raw recording: ``None`` is complete, and
    ``100`` lacks the keyframe that source frames 101..109 depend on.
    """
    from fractions import Fraction

    import av
    import numpy as np
    from foxglove_schemas_protobuf.CompressedVideo_pb2 import CompressedVideo
    from mcap_protobuf.writer import Writer as ProtobufWriter

    from hflow.transform import TransformConfig, write_canonical_episode

    width, height, source_fps, frame_count, gop, bit_count = 320, 180, 30, 150, 10, 8
    encoder = av.CodecContext.create("libx264", "w")
    encoder.width, encoder.height, encoder.pix_fmt = width, height, "yuv420p"
    encoder.framerate, encoder.time_base = Fraction(source_fps, 1), Fraction(1, source_fps)
    encoder.gop_size, encoder.max_b_frames = gop, 0
    encoder.options = {
        "profile": "baseline",
        "tune": "zerolatency",
        "keyint_min": str(gop),
        "sc_threshold": "0",
    }
    packets: list[bytes] = []
    for index in range(frame_count):
        pixels = np.full((height, width, 3), 128, dtype=np.uint8)
        pixels[:48] = 16
        for bit in range(bit_count):
            if index & (1 << (bit_count - bit - 1)):
                pixels[:48, bit * width // bit_count : (bit + 1) * width // bit_count] = 235
        frame = av.VideoFrame.from_ndarray(pixels, format="rgb24").reformat(format="yuv420p")
        frame.pts = index
        packets.extend(bytes(packet) for packet in encoder.encode(frame))
    packets.extend(bytes(packet) for packet in encoder.encode(None))
    assert len(packets) == frame_count

    episodes: dict[int | None, Path] = {}
    for lost_idr in (None, 100):
        raw_path = directory / f"raw_{lost_idr}.mcap"
        canonical_path = directory / f"canonical_{lost_idr}.mcap"
        with ProtobufWriter(str(raw_path)) as writer:
            for index, packet in enumerate(packets):
                if index == lost_idr:
                    continue
                log_time_ns = NUMBERED_VIDEO_START_NS + index * 1_000_000_000 // source_fps
                message = CompressedVideo(frame_id="cam0", format="h264", data=packet)
                message.timestamp.FromNanoseconds(log_time_ns)
                writer.write_message(
                    NUMBERED_VIDEO_TOPIC,
                    message,
                    log_time=log_time_ns,
                    publish_time=log_time_ns,
                )
        write_canonical_episode(raw_path, canonical_path, TransformConfig())
        episodes[lost_idr] = canonical_path
    return episodes


def number_in_luma(pixels: np.ndarray) -> int:
    """The source frame number drawn into a numbered picture's top band."""
    return int(
        "".join(
            str(int(pixels[:48, bit * 40 + 6 : (bit + 1) * 40 - 6].mean() > 128))
            for bit in range(8)
        ),
        2,
    )


def source_frame_number(log_time_ns: int) -> int:
    return round((log_time_ns - NUMBERED_VIDEO_START_NS) * 30 / 1_000_000_000)
