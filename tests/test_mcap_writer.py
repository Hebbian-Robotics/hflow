"""Conformance tests for the canonical MCAP writer's topic-group chunking.

The written files must be spec-conforming MCAP readable by the stock ``mcap``
package: round-trip through ``SeekingReader`` with CRC validation, correct
summary/statistics/index bookkeeping, and -- the whole point -- chunks that
never mix channels from different topic groups.
"""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import IO, Literal

import pytest
from mcap.data_stream import ReadDataStream
from mcap.opcode import Opcode
from mcap.reader import make_reader
from mcap.records import Chunk, Message, MessageIndex
from mcap.stream_reader import StreamReader, breakup_chunk

from hflow.mcap_writer import CanonicalMcapWriter

CAMERA_GROUP = "cameras"
STATE_GROUP = "state"

# Small enough that both the ~600 KB camera stream and the ~230 KB state
# stream split into several chunks.
TEST_CHUNK_SIZE_BYTES = 64 * 1024

CAMERA_MESSAGES_PER_TOPIC = 100
STATE_MESSAGES_PER_TOPIC = 500


@dataclass(frozen=True)
class ExpectedMessage:
    channel_id: int
    topic: str
    log_time: int
    data: bytes


@dataclass(frozen=True)
class WrittenEpisode:
    """What the writer was fed, for asserting against what the reader sees."""

    # In write order, which is global ascending log_time (all times unique).
    expected_messages: list[ExpectedMessage]
    topic_by_channel_id: dict[int, str]
    group_by_channel_id: dict[int, str]


def _camera_payload(topic: str, message_index: int) -> bytes:
    seed_byte = (len(topic) * 31 + message_index * 7) % 256
    return topic.encode() + bytes((seed_byte + j) % 256 for j in range(3000))


def _state_payload(topic: str, message_index: int) -> bytes:
    seed_byte = (len(topic) * 17 + message_index * 5) % 256
    return topic.encode() + bytes((seed_byte + j) % 256 for j in range(180))


def _write_two_group_episode(
    output: Path | IO[bytes], compression: Literal["zstd", "none"] = "zstd"
) -> WrittenEpisode:
    """Write cameras (2 topics, big messages) and state (2 topics, small messages).

    Messages are written interleaved in global log-time order; every log_time
    is unique by construction (distinct per-topic offsets modulo the strides).
    """
    with CanonicalMcapWriter(
        output, chunk_size=TEST_CHUNK_SIZE_BYTES, compression=compression
    ) as writer:
        camera_schema_id = writer.register_schema("camera_schema", "protobuf", b"\x01\x02\x03")
        state_schema_id = writer.register_schema("state_schema", "jsonschema", b"{}")

        topic_by_channel_id: dict[int, str] = {}
        group_by_channel_id: dict[int, str] = {}
        channel_id_by_topic: dict[str, int] = {}
        for topic, schema_id, group in [
            ("/cam/left", camera_schema_id, CAMERA_GROUP),
            ("/cam/right", camera_schema_id, CAMERA_GROUP),
            ("/state/joint", state_schema_id, STATE_GROUP),
            ("/state/gripper", state_schema_id, STATE_GROUP),
        ]:
            channel_id = writer.register_channel(topic, "protobuf", schema_id, group=group)
            topic_by_channel_id[channel_id] = topic
            group_by_channel_id[channel_id] = group
            channel_id_by_topic[topic] = channel_id

        timeline: list[ExpectedMessage] = []
        for i in range(CAMERA_MESSAGES_PER_TOPIC):
            for topic, time_offset in [("/cam/left", 1), ("/cam/right", 2)]:
                timeline.append(
                    ExpectedMessage(
                        channel_id=channel_id_by_topic[topic],
                        topic=topic,
                        log_time=1_000_000 * i + time_offset,
                        data=_camera_payload(topic, i),
                    )
                )
        for k in range(STATE_MESSAGES_PER_TOPIC):
            for topic, time_offset in [("/state/joint", 3), ("/state/gripper", 4)]:
                timeline.append(
                    ExpectedMessage(
                        channel_id=channel_id_by_topic[topic],
                        topic=topic,
                        log_time=200_000 * k + time_offset,
                        data=_state_payload(topic, k),
                    )
                )
        timeline.sort(key=lambda expected: expected.log_time)

        for expected in timeline:
            writer.write_message(expected.channel_id, expected.log_time, expected.data)

    return WrittenEpisode(
        expected_messages=timeline,
        topic_by_channel_id=topic_by_channel_id,
        group_by_channel_id=group_by_channel_id,
    )


def test_round_trip_two_groups(tmp_path: Path) -> None:
    mcap_path = tmp_path / "episode.mcap"
    episode = _write_two_group_episode(mcap_path)

    with mcap_path.open("rb") as stream:
        reader = make_reader(stream, validate_crcs=True)
        read_back = [
            (channel.topic, message.log_time, message.publish_time, message.data)
            for _, channel, message in reader.iter_messages(log_time_order=True)
        ]
        summary = reader.get_summary()

    assert len(read_back) == len(episode.expected_messages)
    # Global log-time order with unique times: read-back must match the
    # timeline exactly, message data intact.
    for (topic, log_time, publish_time, data), expected in zip(
        read_back, episode.expected_messages, strict=True
    ):
        assert topic == expected.topic
        assert log_time == expected.log_time
        assert publish_time == expected.log_time, "publish_time=None must default to log_time"
        assert data == expected.data

    # Per-topic log-time order preserved.
    for wanted_topic in episode.topic_by_channel_id.values():
        topic_times = [log_time for topic, log_time, _, _ in read_back if topic == wanted_topic]
        assert topic_times == sorted(topic_times)
        assert len(topic_times) in (CAMERA_MESSAGES_PER_TOPIC, STATE_MESSAGES_PER_TOPIC)

    assert summary is not None
    statistics = summary.statistics
    assert statistics is not None
    assert statistics.message_count == len(episode.expected_messages)
    assert statistics.schema_count == 2
    assert statistics.channel_count == 4
    assert statistics.chunk_count == len(summary.chunk_indexes)
    assert statistics.message_start_time == episode.expected_messages[0].log_time
    assert statistics.message_end_time == episode.expected_messages[-1].log_time
    expected_counts: dict[int, int] = {}
    for expected in episode.expected_messages:
        expected_counts[expected.channel_id] = expected_counts.get(expected.channel_id, 0) + 1
    assert dict(statistics.channel_message_counts) == expected_counts
    assert {channel.topic for channel in summary.channels.values()} == set(
        episode.topic_by_channel_id.values()
    )
    assert {schema.name for schema in summary.schemas.values()} == {
        "camera_schema",
        "state_schema",
    }


def _read_raw_chunks(mcap_path: Path) -> list[Chunk]:
    with mcap_path.open("rb") as stream:
        return [
            record
            for record in StreamReader(stream, emit_chunks=True).records
            if isinstance(record, Chunk)
        ]


def test_chunk_purity_and_time_major_order(tmp_path: Path) -> None:
    """No chunk mixes groups; each group has several chunks in time order."""
    mcap_path = tmp_path / "episode.mcap"
    episode = _write_two_group_episode(mcap_path)

    chunk_count_by_group: dict[str, int] = {CAMERA_GROUP: 0, STATE_GROUP: 0}
    previous_end_time_by_group: dict[str, int] = {}
    for chunk in _read_raw_chunks(mcap_path):
        message_channel_ids = [
            record.channel_id
            for record in breakup_chunk(chunk, validate_crc=True)
            if isinstance(record, Message)
        ]
        assert message_channel_ids, "every finalized chunk must contain messages"
        groups_in_chunk = {
            episode.group_by_channel_id[channel_id] for channel_id in message_channel_ids
        }
        assert len(groups_in_chunk) == 1, f"chunk mixes topic groups: {sorted(groups_in_chunk)}"
        group = groups_in_chunk.pop()
        chunk_count_by_group[group] += 1

        previous_end_time = previous_end_time_by_group.get(group)
        if previous_end_time is not None:
            assert chunk.message_start_time >= previous_end_time, (
                f"group {group!r} chunk time ranges must be non-decreasing"
            )
        previous_end_time_by_group[group] = chunk.message_end_time

    assert chunk_count_by_group[CAMERA_GROUP] >= 2
    assert chunk_count_by_group[STATE_GROUP] >= 2


def test_metadata_and_attachment_round_trip(tmp_path: Path) -> None:
    mcap_path = tmp_path / "episode.mcap"
    episode_metadata = {"task": "fold_towel", "operator": "op-7", "success": "true"}
    attachment_data = b'{"camera_matrix": [1, 0, 0]}'

    with CanonicalMcapWriter(mcap_path) as writer:
        schema_id = writer.register_schema("state_schema", "jsonschema", b"{}")
        channel_id = writer.register_channel("/state/joint", "json", schema_id, group=STATE_GROUP)
        writer.write_message(channel_id, 10, b"\x01")
        writer.add_metadata("episode/v1", episode_metadata)
        writer.add_attachment(
            "calibration.json",
            "application/json",
            attachment_data,
            log_time=5,
            create_time=3,
        )

    with mcap_path.open("rb") as stream:
        reader = make_reader(stream, validate_crcs=True)
        metadata_records = list(reader.iter_metadata())
        attachments = list(reader.iter_attachments())
        summary = reader.get_summary()

    assert len(metadata_records) == 1
    assert metadata_records[0].name == "episode/v1"
    assert metadata_records[0].metadata == episode_metadata

    assert len(attachments) == 1
    assert attachments[0].name == "calibration.json"
    assert attachments[0].media_type == "application/json"
    assert attachments[0].data == attachment_data
    assert attachments[0].log_time == 5
    assert attachments[0].create_time == 3

    assert summary is not None
    assert summary.statistics is not None
    assert summary.statistics.metadata_count == 1
    assert summary.statistics.attachment_count == 1
    assert len(summary.metadata_indexes) == 1
    assert len(summary.attachment_indexes) == 1
    assert summary.attachment_indexes[0].data_size == len(attachment_data)


def test_compression_none_round_trips(tmp_path: Path) -> None:
    mcap_path = tmp_path / "episode.mcap"
    episode = _write_two_group_episode(mcap_path, compression="none")

    for chunk in _read_raw_chunks(mcap_path):
        assert chunk.compression == ""
        assert len(chunk.data) == chunk.uncompressed_size

    with mcap_path.open("rb") as stream:
        reader = make_reader(stream, validate_crcs=True)
        read_back = [
            (channel.topic, message.log_time, message.data)
            for _, channel, message in reader.iter_messages(log_time_order=True)
        ]
    assert read_back == [
        (expected.topic, expected.log_time, expected.data) for expected in episode.expected_messages
    ]


def test_zero_message_file_is_readable() -> None:
    stream = BytesIO()
    with CanonicalMcapWriter(stream) as writer:
        schema_id = writer.register_schema("state_schema", "jsonschema", b"{}")
        writer.register_channel("/state/joint", "json", schema_id, group=STATE_GROUP)

    stream.seek(0)
    reader = make_reader(stream, validate_crcs=True)
    summary = reader.get_summary()
    assert summary is not None
    assert summary.chunk_indexes == []
    assert summary.statistics is not None
    assert summary.statistics.message_count == 0
    assert summary.statistics.chunk_count == 0
    assert summary.statistics.schema_count == 1
    assert summary.statistics.channel_count == 1
    assert list(reader.iter_messages()) == []


def test_log_time_zero_and_explicit_publish_time_and_sequence() -> None:
    stream = BytesIO()
    with CanonicalMcapWriter(stream) as writer:
        channel_id = writer.register_channel("/state/joint", "json", 0, group=STATE_GROUP)
        writer.write_message(channel_id, 0, b"first", publish_time=7, sequence=42)
        writer.write_message(channel_id, 10, b"second")

    stream.seek(0)
    reader = make_reader(stream, validate_crcs=True)
    messages = [message for _, _, message in reader.iter_messages(log_time_order=True)]
    assert [(m.log_time, m.publish_time, m.sequence, m.data) for m in messages] == [
        (0, 7, 42, b"first"),
        (10, 10, 0, b"second"),
    ]
    summary = reader.get_summary()
    assert summary is not None
    assert summary.statistics is not None
    assert summary.statistics.message_start_time == 0
    assert summary.statistics.message_end_time == 10


def test_chunk_index_message_index_offsets_are_valid(tmp_path: Path) -> None:
    """Every ChunkIndex offset must point at a parseable MessageIndex record."""
    mcap_path = tmp_path / "episode.mcap"
    episode = _write_two_group_episode(mcap_path)

    with mcap_path.open("rb") as stream:
        summary = make_reader(stream, validate_crcs=True).get_summary()
    assert summary is not None
    assert len(summary.chunk_indexes) >= 4

    with mcap_path.open("rb") as stream:
        for chunk_index in summary.chunk_indexes:
            assert chunk_index.message_index_offsets, "chunks must carry message indexes"
            message_index_region_start = chunk_index.chunk_start_offset + chunk_index.chunk_length
            message_index_region_end = message_index_region_start + chunk_index.message_index_length
            groups_indexed = {
                episode.group_by_channel_id[channel_id]
                for channel_id in chunk_index.message_index_offsets
            }
            assert len(groups_indexed) == 1

            for channel_id, offset in chunk_index.message_index_offsets.items():
                assert message_index_region_start <= offset < message_index_region_end
                stream.seek(offset)
                data_stream = ReadDataStream(stream)
                assert data_stream.read1() == Opcode.MESSAGE_INDEX
                data_stream.read8()  # record length
                message_index = MessageIndex.read(data_stream)
                assert message_index.channel_id == channel_id
                assert message_index.records
                for log_time, _ in message_index.records:
                    assert (
                        chunk_index.message_start_time <= log_time <= chunk_index.message_end_time
                    )


def test_write_to_unknown_channel_id_raises() -> None:
    with (
        CanonicalMcapWriter(BytesIO()) as writer,
        pytest.raises(ValueError, match="unknown channel id 999"),
    ):
        writer.write_message(999, 0, b"")


def test_register_channel_with_unknown_schema_id_raises() -> None:
    with CanonicalMcapWriter(BytesIO()) as writer:
        with pytest.raises(ValueError, match="unknown schema id 42"):
            writer.register_channel("/topic", "json", 42, group=STATE_GROUP)
        # schema_id 0 is the MCAP spec's "no schema" sentinel and must be accepted.
        writer.register_channel("/schemaless", "json", 0, group=STATE_GROUP)


def test_use_after_finish_raises() -> None:
    writer = CanonicalMcapWriter(BytesIO())
    schema_id = writer.register_schema("state_schema", "jsonschema", b"{}")
    channel_id = writer.register_channel("/state/joint", "json", schema_id, group=STATE_GROUP)
    writer.finish()
    writer.finish()  # idempotent, so `with` blocks may also call finish() explicitly

    with pytest.raises(RuntimeError, match="finish"):
        writer.write_message(channel_id, 0, b"")
    with pytest.raises(RuntimeError, match="finish"):
        writer.register_schema("another", "jsonschema", b"{}")
    with pytest.raises(RuntimeError, match="finish"):
        writer.register_channel("/other", "json", schema_id, group=STATE_GROUP)
    with pytest.raises(RuntimeError, match="finish"):
        writer.add_metadata("episode/v1", {})
    with pytest.raises(RuntimeError, match="finish"):
        writer.add_attachment("a.bin", "application/octet-stream", b"")


def test_default_library_identifier_names_project_and_format() -> None:
    stream = BytesIO()
    with CanonicalMcapWriter(stream):
        pass
    stream.seek(0)
    header = make_reader(stream).get_header()
    assert header.library.startswith("hflow ")
    assert "episode-format/1" in header.library
    # The header is inside the bytes content_episode_id hashes, so it must
    # carry no release number: see tests/test_identity_stability.py.
    import hflow

    assert hflow.__version__ not in header.library

    explicit_stream = BytesIO()
    with CanonicalMcapWriter(explicit_stream, library="custom-writer/9"):
        pass
    explicit_stream.seek(0)
    assert make_reader(explicit_stream).get_header().library == "custom-writer/9"


class TestPerGroupChunkTargets:
    """`chunk_size` as a mapping: the layout half of derived chunk targets.

    Groups are written at wildly different byte rates -- six cameras produce
    megabytes a second while a proprio channel produces kilobytes -- so
    hflow.format.derived_chunk_size_bytes computes a target per group and the
    transform passes them here. Nothing asserted that the writer APPLIED them,
    so the whole feature could go inert while provenance kept advertising
    targets the bytes did not honor: the 9.18-against-3.79 chunk fetches per
    training sample docs/BENCHMARKS.md measures.
    """

    @staticmethod
    def _chunks_per_group(written: bytes) -> dict[str, int]:
        """How many chunks each group's stream was split into."""
        stream = BytesIO(written)
        summary = make_reader(stream).get_summary()
        assert summary is not None
        group_by_channel_id = {
            channel_id: channel.metadata["group"]
            for channel_id, channel in summary.channels.items()
        }
        counts: dict[str, int] = {}
        for chunk_index in summary.chunk_indexes:
            groups = {
                group_by_channel_id[channel_id] for channel_id in chunk_index.message_index_offsets
            }
            assert len(groups) == 1, "chunk purity is the invariant every group relies on"
            group = groups.pop()
            counts[group] = counts.get(group, 0) + 1
        return counts

    @staticmethod
    def _write(chunk_size: int | dict[str, int]) -> bytes:
        stream = BytesIO()
        with CanonicalMcapWriter(stream, chunk_size=chunk_size) as writer:
            schema_id = writer.register_schema("s", "jsonschema", b"{}")
            for topic, group in (("/cam/left", CAMERA_GROUP), ("/state/joint", STATE_GROUP)):
                writer.register_channel(
                    topic, "protobuf", schema_id, group=group, metadata={"group": group}
                )
            for index in range(200):
                writer.write_message(1, index * 1000, _camera_payload("/cam/left", index))
                writer.write_message(2, index * 1000 + 1, _state_payload("/state/joint", index))
        return stream.getvalue()

    def test_each_group_is_split_at_its_own_target(self) -> None:
        """The cameras target is 8x the state target over a camera stream ~16x
        the state stream, so the two must NOT come out with the same shape."""
        per_group = self._chunks_per_group(
            self._write({CAMERA_GROUP: 256 * 1024, STATE_GROUP: 32 * 1024})
        )

        assert per_group[CAMERA_GROUP] > 1
        assert per_group[STATE_GROUP] > 1
        # ~600 KB of camera bytes at 256 KB against ~36 KB of state at 32 KB.
        assert per_group[CAMERA_GROUP] != per_group[STATE_GROUP]

    def test_a_group_the_mapping_omits_falls_back_to_the_default(self) -> None:
        """`chunk_size` names the groups it has an opinion about; the rest keep
        DEFAULT_CHUNK_SIZE_BYTES rather than becoming unbounded."""
        from hflow.mcap_writer import DEFAULT_CHUNK_SIZE_BYTES

        omitted = self._chunks_per_group(self._write({CAMERA_GROUP: 32 * 1024}))
        pinned = self._chunks_per_group(self._write(DEFAULT_CHUNK_SIZE_BYTES))

        assert omitted[STATE_GROUP] == pinned[STATE_GROUP]
        assert omitted[CAMERA_GROUP] > pinned[CAMERA_GROUP]

    def test_a_plain_int_still_targets_every_group(self) -> None:
        """The public contract callers already pass."""
        per_group = self._chunks_per_group(self._write(32 * 1024))

        assert per_group[CAMERA_GROUP] > 1
        assert per_group[STATE_GROUP] > 1
