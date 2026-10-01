"""Transform: synthetic input MCAP -> canonical episode."""

from collections.abc import Callable
from pathlib import Path

import pytest
from mcap.reader import make_reader

from hflow.format import (
    CANONICAL_VIDEO_SCHEMA_NAME,
    DEFAULT_CAMERA_GROUP,
    METADATA_RECORD_EPISODE,
    METADATA_RECORD_PROVENANCE,
    GopPreset,
)
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode
from hflow.transform import (
    TransformConfig,
    compute_pipeline_version,
    write_canonical_episode,
)

SMALL_SPEC = SyntheticEpisodeSpec(
    duration_s=4.0,
    black_segment=(1.0, 2.0),
    joint_jump_at_s=2.5,
    timestamp_offset_segment=(3.0, 3.5),
)


@pytest.fixture(scope="module")
def source_episode(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return synthesize_episode(tmp_path_factory.mktemp("source") / "episode.mcap", SMALL_SPEC)


@pytest.fixture(scope="module")
def canonical_episode(source_episode: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("canonical") / "episode.canonical.mcap"
    write_canonical_episode(source_episode, output, source_uri=str(source_episode))
    return output


def test_pipeline_version_is_a_content_hash() -> None:
    base = TransformConfig()
    assert compute_pipeline_version(base) == compute_pipeline_version(TransformConfig())
    changed = TransformConfig(gop_preset=GopPreset.WORLD_MODEL)
    assert compute_pipeline_version(base) != compute_pipeline_version(changed)
    assert len(compute_pipeline_version(base)) == 12


@pytest.mark.parametrize(
    ("construct", "match"),
    [
        (lambda: TransformConfig(crf=True), r"^crf must be an int, got bool$"),
        (lambda: TransformConfig(crf=-1), r"^crf must be in \[0, 51\], got -1$"),
        (lambda: TransformConfig(crf=52), r"^crf must be in \[0, 51\], got 52$"),
        (
            lambda: TransformConfig(gop_seconds=True),
            r"^gop_seconds must be an int or float, got bool$",
        ),
        (lambda: TransformConfig(gop_seconds=0), r"^gop_seconds must be > 0, got 0$"),
        (
            lambda: TransformConfig(gop_seconds=float("nan")),
            r"^gop_seconds must be finite, got nan$",
        ),
        (
            lambda: TransformConfig(gop_seconds=float("inf")),
            r"^gop_seconds must be finite, got inf$",
        ),
        (
            lambda: TransformConfig(chunk_size_bytes=True),
            r"^chunk_size_bytes must be an int, got bool$",
        ),
        (
            lambda: TransformConfig(chunk_size_bytes=0),
            r"^chunk_size_bytes must be > 0, got 0$",
        ),
        (
            lambda: TransformConfig(chunk_size_bytes=-1),
            r"^chunk_size_bytes must be > 0, got -1$",
        ),
    ],
)
def test_transform_config_rejects_invalid_numeric_settings(
    construct: Callable[[], TransformConfig], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        construct()


def test_transform_config_accepts_valid_numeric_settings() -> None:
    config = TransformConfig(crf=0, gop_seconds=1, chunk_size_bytes=1)

    assert config.crf == 0
    assert config.gop_seconds == 1
    assert config.chunk_size_bytes == 1


def test_stamps_carry_source_robot_software_version(source_episode: Path, tmp_path: Path) -> None:
    stamps = write_canonical_episode(source_episode, tmp_path / "out.mcap")
    assert stamps.schema_version == "1"
    assert len(stamps.pipeline_version) == 12
    assert stamps.robot_software_version == SMALL_SPEC.robot_software_version
    assert "ffmpeg" in stamps.ffmpeg_version


def test_canonical_file_is_conforming_mcap(canonical_episode: Path) -> None:
    with canonical_episode.open("rb") as stream:
        reader = make_reader(stream, validate_crcs=True)
        summary = reader.get_summary()
        assert summary is not None
        message_count = sum(1 for _ in reader.iter_messages())
        assert message_count > 0


def test_camera_channels_became_compressed_video(
    source_episode: Path, canonical_episode: Path
) -> None:
    with canonical_episode.open("rb") as stream:
        summary = make_reader(stream).get_summary()
        assert summary is not None
        by_topic = {
            channel.topic: (channel, summary.schemas[channel.schema_id])
            for channel in summary.channels.values()
        }
    for camera_name in SMALL_SPEC.cameras:
        channel, schema = by_topic[f"/{camera_name}/compressed"]
        assert schema.name == CANONICAL_VIDEO_SCHEMA_NAME
        assert schema.encoding == "protobuf"
        assert channel.message_encoding == "protobuf"
    joint_channel, joint_schema = by_topic["/joint_states"]
    assert joint_schema.name == "sensor_msgs/msg/JointState"
    assert joint_channel.message_encoding == "cdr"


def _write_state_source(path: Path, *, include_empty_camera: bool) -> Path:
    from mcap.writer import Writer as StockWriter

    with path.open("wb") as stream:
        writer = StockWriter(stream)
        writer.start(profile="", library="test")
        if include_empty_camera:
            camera_schema_id = writer.register_schema(
                name="sensor_msgs/msg/CompressedImage",
                encoding="ros2msg",
                data=b"",
            )
            writer.register_channel(
                topic="/camera/empty",
                message_encoding="cdr",
                schema_id=camera_schema_id,
            )
        state_schema_id = writer.register_schema(name="state", encoding="jsonschema", data=b"{}")
        state_channel_id = writer.register_channel(
            topic="/state",
            message_encoding="json",
            schema_id=state_schema_id,
        )
        writer.add_message(
            state_channel_id,
            log_time=1_000_000_000,
            publish_time=1_000_000_000,
            data=b'{"value": 1}',
        )
        writer.finish()
    return path


def test_empty_camera_declaration_survives_transform_and_changes_identity(tmp_path: Path) -> None:
    with_camera = _write_state_source(tmp_path / "with-camera.mcap", include_empty_camera=True)
    without_camera = _write_state_source(tmp_path / "without-camera.mcap", include_empty_camera=False)
    canonical_with = tmp_path / "with-camera.canonical.mcap"
    canonical_without = tmp_path / "without-camera.canonical.mcap"

    write_canonical_episode(with_camera, canonical_with)
    write_canonical_episode(without_camera, canonical_without)

    with canonical_with.open("rb") as stream:
        summary = make_reader(stream).get_summary()
        assert summary is not None
        by_topic = {
            channel.topic: (channel, summary.schemas[channel.schema_id])
            for channel in summary.channels.values()
        }

    camera_channel, camera_schema = by_topic["/camera/empty"]
    assert camera_schema.name == CANONICAL_VIDEO_SCHEMA_NAME
    assert camera_schema.encoding == "protobuf"
    assert camera_channel.message_encoding == "protobuf"
    assert canonical_with.read_bytes() != canonical_without.read_bytes()


def test_state_messages_pass_through_byte_for_byte(
    source_episode: Path, canonical_episode: Path
) -> None:
    def joint_payloads(path: Path) -> list[tuple[int, bytes]]:
        with path.open("rb") as stream:
            return [
                (message.log_time, message.data)
                for _schema, _channel, message in make_reader(stream).iter_messages(
                    topics=["/joint_states"]
                )
            ]

    assert joint_payloads(canonical_episode) == joint_payloads(source_episode)


def test_metadata_and_attachments_survive(canonical_episode: Path) -> None:
    with canonical_episode.open("rb") as stream:
        reader = make_reader(stream)
        records = {record.name: dict(record.metadata) for record in reader.iter_metadata()}
        attachments = list(reader.iter_attachments())
    assert records[METADATA_RECORD_EPISODE]["task"] == SMALL_SPEC.task
    provenance = records[METADATA_RECORD_PROVENANCE]
    assert provenance["schema_version"] == "1"
    assert len(provenance["pipeline_version"]) == 12
    assert "gop_preset" in provenance
    assert any(attachment.name == "calibration.json" for attachment in attachments)


def test_chunks_never_mix_groups(canonical_episode: Path) -> None:
    """Every chunk holds either only camera channels or only state channels."""
    with canonical_episode.open("rb") as stream:
        summary = make_reader(stream).get_summary()
        assert summary is not None
        camera_channel_ids = {
            channel.id
            for channel in summary.channels.values()
            if summary.schemas[channel.schema_id].name == CANONICAL_VIDEO_SCHEMA_NAME
        }
        chunk_group_kinds: set[str] = set()
        assert summary.chunk_indexes, "canonical file must be chunked"
        for chunk_index in summary.chunk_indexes:
            channel_ids = set(chunk_index.message_index_offsets.keys())
            assert channel_ids, "every chunk must carry message indexes"
            in_cameras = channel_ids <= camera_channel_ids
            in_state = channel_ids.isdisjoint(camera_channel_ids)
            assert in_cameras or in_state, (
                f"chunk mixes groups: {channel_ids} vs cameras {camera_channel_ids}"
            )
            chunk_group_kinds.add(DEFAULT_CAMERA_GROUP if in_cameras else "state")
    assert chunk_group_kinds == {DEFAULT_CAMERA_GROUP, "state"}


def test_duplicate_metadata_refuses_with_source_not_conforming(tmp_path: Path) -> None:
    from mcap.writer import Writer as StockWriter

    from hflow.ingest_ledger import IngestFailureKind, classify_ingest_failure
    from hflow.transform import SourceNotConforming

    source = tmp_path / "dup_meta.mcap"
    with source.open("wb") as stream:
        writer = StockWriter(stream)
        writer.start(profile="", library="test")
        schema_id = writer.register_schema(name="dummy", encoding="json", data=b"{}")
        channel_id = writer.register_channel(
            topic="/state", message_encoding="json", schema_id=schema_id
        )
        writer.add_message(channel_id, log_time=10**9, data=b"{}", publish_time=10**9)
        writer.add_metadata("episode/v1", {"task": "first", "success": "false"})
        writer.add_metadata("episode/v1", {"task": "second", "success": "true"})
        writer.finish()

    output = tmp_path / "out.mcap"
    with pytest.raises(
        SourceNotConforming, match=r"duplicate metadata record 'episode/v1' in source"
    ) as exc_info:
        write_canonical_episode(source, output)

    assert classify_ingest_failure(exc_info.value) is IngestFailureKind.SOURCE_UNSUPPORTED


def test_episode_reader_iter_metadata_yields_all_records(tmp_path: Path) -> None:
    from mcap.writer import Writer as StockWriter

    from hflow.reader import open_reader

    path = tmp_path / "multi_meta.mcap"
    with path.open("wb") as stream:
        writer = StockWriter(stream)
        writer.start(profile="", library="test")
        schema_id = writer.register_schema(name="dummy", encoding="json", data=b"{}")
        channel_id = writer.register_channel(
            topic="/state", message_encoding="json", schema_id=schema_id
        )
        writer.add_message(channel_id, log_time=10**9, data=b"{}", publish_time=10**9)
        writer.add_metadata("episode/v1", {"task": "first"})
        writer.add_metadata("episode/v1", {"task": "second"})
        writer.add_metadata("custom/v1", {"foo": "bar"})
        writer.finish()

    reader = open_reader(path)
    records = list(reader.iter_metadata())
    assert len(records) == 3
    assert [r.name for r in records] == ["episode/v1", "episode/v1", "custom/v1"]
    # Keyed metadata() keeps the later one as documented
    assert reader.metadata()["episode/v1"]["task"] == "second"
