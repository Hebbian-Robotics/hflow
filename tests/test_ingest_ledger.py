"""``classify_ingest_failure`` pins each refusal to the right ``IngestFailureKind``."""

from pathlib import Path

import numpy as np
import pytest
from mcap.exceptions import InvalidMagic
from mcap.writer import Writer as StockWriter

from hflow import transform
from hflow.app import SourceNotFound
from hflow.format import METADATA_RECORD_EPISODE
from hflow.ingest_ledger import IngestFailureKind, classify_ingest_failure
from hflow.resample import DerivedSeries
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode
from hflow.transform import SourceNotConforming, write_canonical_episode


def test_classify_source_not_found_as_source_missing() -> None:
    error = SourceNotFound("episode 'missing.mcap' not found")
    assert classify_ingest_failure(error) == IngestFailureKind.SOURCE_MISSING


def test_classify_mcap_error_as_source_unreadable() -> None:
    error = InvalidMagic(b"not-an-mcap-file")
    assert classify_ingest_failure(error) == IngestFailureKind.SOURCE_UNREADABLE


def test_classify_source_not_conforming_as_source_unsupported() -> None:
    error = SourceNotConforming("x")
    assert classify_ingest_failure(error) == IngestFailureKind.SOURCE_UNSUPPORTED


def test_classify_unrecognized_error_as_infrastructure() -> None:
    error = RuntimeError("unknown")
    assert classify_ingest_failure(error) == IngestFailureKind.INFRASTRUCTURE


def test_unsupported_compressed_image_format_classifies_as_source_unsupported() -> None:
    with pytest.raises(SourceNotConforming) as raised:
        transform._input_codec_for_image_format("bogus", "/cam")
    assert classify_ingest_failure(raised.value) == IngestFailureKind.SOURCE_UNSUPPORTED


def test_mixed_compressed_image_formats_classify_as_source_unsupported(tmp_path: Path) -> None:
    from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
    from mcap_protobuf.schema import build_file_descriptor_set

    source = tmp_path / "mixed-image-formats.mcap"
    with source.open("wb") as stream:
        writer = StockWriter(stream)
        writer.start(profile="", library="test")
        schema_id = writer.register_schema(
            name="foxglove.CompressedImage",
            encoding="protobuf",
            data=build_file_descriptor_set(CompressedImage).SerializeToString(),
        )
        channel_id = writer.register_channel(
            topic="/cam", message_encoding="protobuf", schema_id=schema_id
        )
        for message_index, image_format in enumerate(("jpeg", "png"), start=1):
            log_time = message_index * 10**9
            message = CompressedImage()
            message.timestamp.FromNanoseconds(log_time)
            message.frame_id = "cam"
            message.data = b"image bytes are not decoded before the format consistency check"
            message.format = image_format
            writer.add_message(
                channel_id,
                log_time=log_time,
                data=message.SerializeToString(),
                publish_time=log_time,
            )
        writer.finish()

    with pytest.raises(SourceNotConforming, match="mixes image formats") as raised:
        write_canonical_episode(source, tmp_path / "out.mcap")
    assert classify_ingest_failure(raised.value) == IngestFailureKind.SOURCE_UNSUPPORTED


def test_nonconforming_passthrough_video_classifies_as_source_unsupported(tmp_path: Path) -> None:
    from foxglove_schemas_protobuf.CompressedVideo_pb2 import CompressedVideo
    from mcap_protobuf.schema import build_file_descriptor_set

    source = tmp_path / "h265.mcap"
    with source.open("wb") as stream:
        writer = StockWriter(stream)
        writer.start(profile="", library="test")
        schema_id = writer.register_schema(
            name="foxglove.CompressedVideo",
            encoding="protobuf",
            data=build_file_descriptor_set(CompressedVideo).SerializeToString(),
        )
        channel_id = writer.register_channel(
            topic="/cam", message_encoding="protobuf", schema_id=schema_id
        )
        message = CompressedVideo()
        message.timestamp.FromNanoseconds(10**9)
        message.frame_id = "cam"
        message.data = b"\x00\x00\x00\x01\x40junk"
        message.format = "h265"
        writer.add_message(
            channel_id, log_time=10**9, data=message.SerializeToString(), publish_time=10**9
        )
        writer.finish()

    with pytest.raises(SourceNotConforming) as raised:
        write_canonical_episode(source, tmp_path / "out.mcap")
    assert classify_ingest_failure(raised.value) == IngestFailureKind.SOURCE_UNSUPPORTED


def test_raw_image_schema_classifies_as_source_unsupported(tmp_path: Path) -> None:
    source = tmp_path / "raw_image.mcap"
    with source.open("wb") as stream:
        writer = StockWriter(stream)
        writer.start(profile="", library="test")
        schema_id = writer.register_schema(
            name="sensor_msgs/msg/Image", encoding="ros2msg", data=b"std_msgs/Header header"
        )
        channel_id = writer.register_channel(
            topic="/raw_cam", message_encoding="cdr", schema_id=schema_id
        )
        writer.add_message(channel_id, log_time=1, data=b"", publish_time=1)
        writer.add_metadata(METADATA_RECORD_EPISODE, {})
        writer.finish()

    with pytest.raises(SourceNotConforming) as raised:
        write_canonical_episode(source, tmp_path / "out.mcap")
    assert classify_ingest_failure(raised.value) == IngestFailureKind.SOURCE_UNSUPPORTED


def test_derived_topic_collision_classifies_as_source_unsupported(tmp_path: Path) -> None:
    cameraless_spec = SyntheticEpisodeSpec(
        duration_s=2.0,
        cameras=(),
        joint_hz=50.0,
        black_segment=None,
        joint_jump_at_s=None,
        timestamp_offset_segment=None,
    )
    source = synthesize_episode(tmp_path / "episode.mcap", cameraless_spec)
    series = DerivedSeries(
        timestamps_ns=np.asarray([1_755_000_000_500_000_000], dtype=np.int64),
        values={"value": np.asarray([1.0])},
    )

    with pytest.raises(SourceNotConforming) as raised:
        write_canonical_episode(
            source, tmp_path / "out.mcap", derived=[("/joint_states", series, "v1")]
        )
    assert classify_ingest_failure(raised.value) == IngestFailureKind.SOURCE_UNSUPPORTED
