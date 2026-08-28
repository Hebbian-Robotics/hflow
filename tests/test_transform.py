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
    ("construct", "field"),
    [
        (lambda: TransformConfig(crf=True), "crf"),
        (lambda: TransformConfig(crf=-1), "crf"),
        (lambda: TransformConfig(crf=52), "crf"),
        (lambda: TransformConfig(gop_seconds=True), "gop_seconds"),
        (lambda: TransformConfig(gop_seconds=0), "gop_seconds"),
        (lambda: TransformConfig(gop_seconds=float("nan")), "gop_seconds"),
        (lambda: TransformConfig(gop_seconds=float("inf")), "gop_seconds"),
        (lambda: TransformConfig(chunk_size_bytes=True), "chunk_size_bytes"),
        (lambda: TransformConfig(chunk_size_bytes=0), "chunk_size_bytes"),
        (lambda: TransformConfig(chunk_size_bytes=-1), "chunk_size_bytes"),
    ],
)
def test_transform_config_rejects_invalid_numeric_settings(
    construct: Callable[[], TransformConfig], field: str
) -> None:
    with pytest.raises(ValueError, match=field):
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
