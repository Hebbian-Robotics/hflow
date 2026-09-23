"""``classify_ingest_failure`` pins each refusal to the right ``IngestFailureKind``."""

from pathlib import Path

import pytest
from mcap.exceptions import InvalidMagic
from mcap.records import Chunk
from mcap.stream_reader import CRCValidationError
from mcap.writer import Writer as StockWriter

from hflow import transform
from hflow.app import SourceNotFound
from hflow.format import METADATA_RECORD_EPISODE
from hflow.ingest_ledger import IngestFailureKind, classify_ingest_failure
from hflow.transform import SourceNotConforming, write_canonical_episode


@pytest.mark.parametrize(
    ("error", "expected_kind"),
    [
        pytest.param(
            SourceNotFound("episode 'missing.mcap' not found"),
            IngestFailureKind.SOURCE_MISSING,
            id="source-not-found",
        ),
        pytest.param(
            InvalidMagic(b"not-an-mcap-file"),
            IngestFailureKind.SOURCE_UNREADABLE,
            id="mcap-error",
        ),
        # CRCValidationError subclasses ValueError, not McapError (#431):
        # without its own branch it would fall through to INFRASTRUCTURE and
        # blame the platform for a damaged recording.
        pytest.param(
            CRCValidationError(
                expected=1,
                actual=2,
                record=Chunk(
                    compression="",
                    data=b"",
                    message_end_time=0,
                    message_start_time=0,
                    uncompressed_crc=1,
                    uncompressed_size=0,
                ),
            ),
            IngestFailureKind.SOURCE_UNREADABLE,
            id="crc-validation-error",
        ),
        pytest.param(
            SourceNotConforming("x"),
            IngestFailureKind.SOURCE_UNSUPPORTED,
            id="source-not-conforming",
        ),
        pytest.param(
            RuntimeError("unknown"), IngestFailureKind.INFRASTRUCTURE, id="unrecognized-error"
        ),
    ],
)
def test_classify_ingest_failure_maps_each_error_to_its_kind(
    error: Exception, expected_kind: IngestFailureKind
) -> None:
    assert classify_ingest_failure(error) == expected_kind


def test_ingest_refuses_a_source_with_a_damaged_chunk_payload(tmp_path: Path) -> None:
    """A structurally valid MCAP whose chunk payload does not match its
    recorded CRC must not transcode quietly into a canonical episode with a
    fresh receipt over corrupt bytes (#431). ``open_reader`` only checks CRCs
    when told to; ingest's read now asks for it."""
    from reuse_test_helpers import write_payload_damaged_mcap

    source = tmp_path / "payload-damaged.mcap"
    write_payload_damaged_mcap(source)

    output = tmp_path / "out.mcap"
    with pytest.raises(CRCValidationError):
        write_canonical_episode(source, output)

    assert not output.exists()


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
