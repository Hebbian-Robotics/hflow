"""Multiple channels per topic are represented by channel id.

MCAP legally allows several channels to share a topic (e.g. a json status
channel and a ros2msg one both on ``/status``). The reader's ``channels()``
accessor, Episode addressing by channel id, and the transform must all
represent both channels; only the topic-keyed convenience views refuse.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from mcap.reader import make_reader
from mcap.writer import CompressionType
from mcap.writer import Writer as StockWriter

from hflow.doctor import diagnose
from hflow.episode import Episode
from hflow.format import METADATA_RECORD_EPISODE
from hflow.reader import EpisodeReader, PythonMcapEpisodeReader, open_reader
from hflow.transform import write_canonical_episode

SHARED_TOPIC = "/status"
BOOL_SCHEMA_NAME = "std_msgs/msg/Bool"
BOOL_SCHEMA_TEXT = b"bool data"
# Little-endian XCDR1 encapsulation header, then the single bool byte.
CDR_ENCAPSULATION_HEADER = b"\x00\x01\x00\x00"

JSON_PAYLOADS = [json.dumps({"ok": True, "index": index}).encode() for index in range(3)]
CDR_PAYLOADS = [CDR_ENCAPSULATION_HEADER + bytes([index % 2]) for index in range(4)]


@pytest.fixture(scope="module")
def dual_channel_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One topic, two channels: json (schema-less) and ros2msg (CDR)."""
    path = tmp_path_factory.mktemp("multichannel") / "dual.mcap"
    with path.open("wb") as stream:
        writer = StockWriter(stream)
        writer.start(profile="", library="test")
        json_channel_id = writer.register_channel(
            topic=SHARED_TOPIC, message_encoding="json", schema_id=0
        )
        bool_schema_id = writer.register_schema(
            name=BOOL_SCHEMA_NAME, encoding="ros2msg", data=BOOL_SCHEMA_TEXT
        )
        cdr_channel_id = writer.register_channel(
            topic=SHARED_TOPIC, message_encoding="cdr", schema_id=bool_schema_id
        )
        for index, payload in enumerate(JSON_PAYLOADS):
            writer.add_message(
                json_channel_id, log_time=index * 10**9, data=payload, publish_time=index * 10**9
            )
        for index, payload in enumerate(CDR_PAYLOADS):
            log_time = index * 10**9 + 500_000_000  # interleave with the json stream
            writer.add_message(
                cdr_channel_id, log_time=log_time, data=payload, publish_time=log_time
            )
        writer.add_metadata(METADATA_RECORD_EPISODE, {"task": "multichannel-demo"})
        writer.finish()
    return path


def test_reader_channels_exposes_both_and_topics_refuses(dual_channel_source: Path) -> None:
    reader = open_reader(dual_channel_source)
    try:
        infos = reader.channels()
        assert len(infos) == 2
        assert all(channel_id == info.channel_id for channel_id, info in infos.items())
        assert {info.topic for info in infos.values()} == {SHARED_TOPIC}
        by_encoding = {info.message_encoding: info for info in infos.values()}
        assert set(by_encoding) == {"json", "cdr"}
        assert by_encoding["json"].has_schema is False
        assert by_encoding["json"].message_count == len(JSON_PAYLOADS)
        assert by_encoding["cdr"].schema_name == BOOL_SCHEMA_NAME
        assert by_encoding["cdr"].schema_data == BOOL_SCHEMA_TEXT
        assert by_encoding["cdr"].message_count == len(CDR_PAYLOADS)
        with pytest.raises(ValueError, match=f"multiple channels for topic '{SHARED_TOPIC}'"):
            reader.topics()
    finally:
        reader.close()


def test_reader_batches_are_per_channel_and_filterable(dual_channel_source: Path) -> None:
    reader = open_reader(dual_channel_source)
    try:
        infos = reader.channels()
        by_encoding = {info.message_encoding: info for info in infos.values()}

        batches = list(reader.iter_batches())
        assert {batch.channel_id for batch in batches} == set(infos)
        for batch in batches:
            assert batch.topic == SHARED_TOPIC
        data_by_channel_id: dict[int, list[bytes]] = {}
        for batch in batches:
            data_by_channel_id.setdefault(batch.channel_id, []).extend(batch.data)
        assert data_by_channel_id[by_encoding["json"].channel_id] == JSON_PAYLOADS
        assert data_by_channel_id[by_encoding["cdr"].channel_id] == CDR_PAYLOADS

        cdr_only = list(reader.iter_batches(channel_ids=[by_encoding["cdr"].channel_id]))
        assert {batch.channel_id for batch in cdr_only} == {by_encoding["cdr"].channel_id}
        assert [payload for batch in cdr_only for payload in batch.data] == CDR_PAYLOADS
    finally:
        reader.close()


def test_episode_addresses_channels_by_id(dual_channel_source: Path) -> None:
    with Episode(dual_channel_source) as episode:
        assert len(episode.channels) == 2
        by_encoding = {info.message_encoding: info for info in episode.channels.values()}

        with pytest.raises(ValueError, match=r"has 2 channels \(ids"):
            episode.channel(SHARED_TOPIC)

        json_channel = episode.channel(by_encoding["json"].channel_id)
        assert json_channel.channel_id == by_encoding["json"].channel_id
        assert json_channel.topic == SHARED_TOPIC
        assert json_channel.messages == [json.loads(payload) for payload in JSON_PAYLOADS]

        cdr_channel = episode.channel(by_encoding["cdr"].channel_id)
        assert [message.data for message in cdr_channel.messages] == [
            bool(payload[-1]) for payload in CDR_PAYLOADS
        ]


def _write_two_topic_mcap(
    path: Path,
    target_payloads: list[bytes],
    camera_payloads: list[bytes],
    **writer_kwargs: Any,
) -> tuple[int, int]:
    """Write a small ``/target`` channel and a large ``/camera`` channel.

    The default small chunk size keeps the two streams in separate chunks,
    the layout where an early topic filter can skip unrelated data. Returns
    the two channel ids in registration order.
    """
    writer_kwargs.setdefault("chunk_size", 1024)
    with path.open("wb") as stream:
        writer = StockWriter(stream, compression=CompressionType.NONE, **writer_kwargs)
        writer.start(profile="", library="test")
        target_channel_id = writer.register_channel(
            topic="/target", message_encoding="json", schema_id=0
        )
        camera_channel_id = writer.register_channel(
            topic="/camera", message_encoding="json", schema_id=0
        )
        for index, payload in enumerate(target_payloads):
            writer.add_message(
                target_channel_id, log_time=1 + index, data=payload, publish_time=1 + index
            )
        for index, payload in enumerate(camera_payloads):
            writer.add_message(
                camera_channel_id, log_time=10 + index, data=payload, publish_time=10 + index
            )
        writer.finish()
    return target_channel_id, camera_channel_id


def _trace_mcap_iter_messages(
    hflow_reader: EpisodeReader, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[Any], list[str]]:
    """Record the ``topics`` argument HFlow passes to the stock mcap reader
    and the topic of every message that crosses that library boundary."""
    assert isinstance(hflow_reader, PythonMcapEpisodeReader)
    mcap_reader = hflow_reader._reader
    original_iter_messages = mcap_reader.iter_messages
    topics_passed: list[Any] = []
    topics_yielded: list[str] = []

    def traced_iter_messages(*args: Any, **kwargs: Any) -> Iterator[Any]:
        topics_passed.append(kwargs.get("topics", args[0] if args else None))
        for schema, channel, message in original_iter_messages(*args, **kwargs):
            topics_yielded.append(channel.topic)
            yield schema, channel, message

    monkeypatch.setattr(mcap_reader, "iter_messages", traced_iter_messages)
    return topics_passed, topics_yielded


def test_channel_read_is_constrained_to_the_requested_topic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reading one small channel must not pull a large unrelated stream
    through the underlying MCAP reader: the channel's topic is passed down,
    so only that topic's messages cross the library boundary."""
    path = tmp_path / "two_topics.mcap"
    target_payload = b"t" * 4096
    camera_payloads = [bytes([65 + index]) * 4096 for index in range(8)]
    _write_two_topic_mcap(path, [target_payload], camera_payloads)

    with Episode(path) as episode:
        topics_passed, topics_yielded = _trace_mcap_iter_messages(episode._reader, monkeypatch)
        channel = episode.channel("/target")
        assert channel.raw == [target_payload]
        assert channel.timestamps.tolist() == [1]
        assert topics_passed == [["/target"]]
        assert topics_yielded == ["/target"]


def test_shared_topic_channel_read_keeps_channel_id_selection(
    dual_channel_source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The topic constraint narrows the read but never replaces channel-id
    selection: with two channels on one topic, the MCAP reader receives the
    shared topic, yields both channels, and exactly the requested channel's
    messages come back."""
    with Episode(dual_channel_source) as episode:
        by_encoding = {info.message_encoding: info for info in episode.channels.values()}
        topics_passed, topics_yielded = _trace_mcap_iter_messages(episode._reader, monkeypatch)
        cdr_channel = episode.channel(by_encoding["cdr"].channel_id)
        assert topics_passed == [[SHARED_TOPIC]]
        assert set(topics_yielded) == {SHARED_TOPIC}
        assert cdr_channel.channel_id == by_encoding["cdr"].channel_id
        assert cdr_channel.raw == CDR_PAYLOADS


def test_decoded_batches_by_channel_id_derive_the_topic_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``Episode.channel()`` passes its topic explicitly, but
    ``iter_decoded_batches(channel_ids=...)`` still reaches the reader with
    ``topics=None`` -- the reader itself must derive the topic filter from
    the channel ids so the underlying read narrows to the requested stream."""
    path = tmp_path / "two_topics_decoded.mcap"
    target_payloads = [json.dumps({"joint": index}).encode() for index in range(2)]
    camera_payloads = [json.dumps({"pad": "y" * 4096}).encode() for _ in range(8)]
    target_channel_id, _ = _write_two_topic_mcap(path, target_payloads, camera_payloads)

    with Episode(path) as episode:
        topics_passed, topics_yielded = _trace_mcap_iter_messages(episode._reader, monkeypatch)
        batches = list(episode.iter_decoded_batches(channel_ids=[target_channel_id]))
        assert [message for batch in batches for message in batch.messages] == [
            {"joint": 0},
            {"joint": 1},
        ]
        assert topics_passed == [["/target"]]
        assert topics_yielded == ["/target", "/target"]


def test_an_explicit_topic_filter_is_never_replaced_by_the_derived_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The derivation fires only when the caller left ``topics`` unset.

    ``Episode.channel()`` passes both a topic and a channel id (#475), so if
    the ``topics is None`` condition were dropped, the caller's explicit
    filter would be silently replaced by one derived from the channel ids.
    For that caller the two agree, which is why deleting the condition left
    the whole suite green. Asking for one topic while naming a channel on a
    different one is what separates them: the reader must receive what the
    caller asked for, and yield nothing, rather than quietly reading the
    other stream instead.
    """
    path = tmp_path / "two_topics_explicit.mcap"
    _, camera_channel_id = _write_two_topic_mcap(path, [b"t" * 16], [b"c" * 16])

    with Episode(path) as episode:
        topics_passed, topics_yielded = _trace_mcap_iter_messages(episode._reader, monkeypatch)
        batches = list(
            episode._reader.iter_batches(topics=["/target"], channel_ids=[camera_channel_id])
        )

    assert topics_passed == [["/target"]]
    assert topics_yielded == ["/target"]
    assert batches == []


def test_channel_id_read_without_a_summary_falls_back_to_a_full_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file with no summary section cannot map channel ids to topics, so a
    channel-id read falls back to the unconstrained scan instead of refusing,
    and the channel-id filter still selects the right messages. A truncated
    file is deliberately not this case: ``channels()`` raises mcap's
    ``RecordLengthLimitExceeded`` rather than ``ValueError``, and any read of
    such a file surfaces the same error, so damage stays loud."""
    path = tmp_path / "unindexed.mcap"
    target_payload = b'{"v": 1}'
    target_channel_id, _ = _write_two_topic_mcap(
        path,
        [target_payload],
        [b"y" * 4096 for _ in range(8)],
        use_chunking=False,
        use_statistics=False,
        use_summary_offsets=False,
        repeat_channels=False,
        repeat_schemas=False,
    )

    reader = open_reader(path)
    try:
        with pytest.raises(ValueError, match="no MCAP summary section"):
            reader.channels()
        topics_passed, topics_yielded = _trace_mcap_iter_messages(reader, monkeypatch)
        batches = list(reader.iter_batches(channel_ids=[target_channel_id]))
        assert [payload for batch in batches for payload in batch.data] == [target_payload]
        assert [batch.topic for batch in batches] == ["/target"]
        assert topics_passed == [None]
        assert set(topics_yielded) == {"/target", "/camera"}
    finally:
        reader.close()


def test_episode_streams_several_decoded_channels_in_bounded_batches(
    dual_channel_source: Path,
) -> None:
    with Episode(dual_channel_source) as episode:
        decoded_batches = list(episode.iter_decoded_batches(batch_max_messages=2))
        channel_info_by_encoding = {
            channel_info.message_encoding: channel_info
            for channel_info in episode.channels.values()
        }

    decoded_messages_by_channel_id: dict[int, list[Any]] = {}
    log_times_by_channel_id: dict[int, list[int]] = {}
    for decoded_batch in decoded_batches:
        assert len(decoded_batch) <= 2
        decoded_messages_by_channel_id.setdefault(decoded_batch.channel_id, []).extend(
            decoded_batch.messages
        )
        log_times_by_channel_id.setdefault(decoded_batch.channel_id, []).extend(
            decoded_batch.log_times.tolist()
        )

    json_channel_id = channel_info_by_encoding["json"].channel_id
    cdr_channel_id = channel_info_by_encoding["cdr"].channel_id
    assert decoded_messages_by_channel_id[json_channel_id] == [
        json.loads(payload) for payload in JSON_PAYLOADS
    ]
    assert [message.data for message in decoded_messages_by_channel_id[cdr_channel_id]] == [
        bool(payload[-1]) for payload in CDR_PAYLOADS
    ]
    assert log_times_by_channel_id[json_channel_id] == [0, 10**9, 2 * 10**9]
    assert log_times_by_channel_id[cdr_channel_id] == [
        500_000_000,
        1_500_000_000,
        2_500_000_000,
        3_500_000_000,
    ]


def test_episode_unknown_keys_raise_helpfully(dual_channel_source: Path) -> None:
    with Episode(dual_channel_source) as episode:
        with pytest.raises(KeyError, match="not in episode"):
            episode.channel("/nonexistent")
        with pytest.raises(KeyError, match="channel id 9999 not in episode"):
            episode.channel(9999)


def test_transform_round_trips_both_channels(dual_channel_source: Path, tmp_path: Path) -> None:
    output = tmp_path / "dual.canonical.mcap"
    write_canonical_episode(dual_channel_source, output)

    with output.open("rb") as stream:
        reader = make_reader(stream)
        summary = reader.get_summary()
        assert summary is not None
        assert [channel.topic for channel in summary.channels.values()] == [
            SHARED_TOPIC,
            SHARED_TOPIC,
        ]
        channels_by_encoding = {
            channel.message_encoding: channel for channel in summary.channels.values()
        }
        assert set(channels_by_encoding) == {"json", "cdr"}
        # The schema-less json channel keeps the "no schema" sentinel; the CDR
        # channel keeps its original schema byte-for-byte.
        assert channels_by_encoding["json"].schema_id == 0
        cdr_schema = summary.schemas[channels_by_encoding["cdr"].schema_id]
        assert cdr_schema.name == BOOL_SCHEMA_NAME
        assert cdr_schema.encoding == "ros2msg"
        assert cdr_schema.data == BOOL_SCHEMA_TEXT

        payloads_by_channel_id: dict[int, list[bytes]] = {}
        for _schema, channel, message in reader.iter_messages(log_time_order=True):
            payloads_by_channel_id.setdefault(channel.id, []).append(message.data)
        assert payloads_by_channel_id[channels_by_encoding["json"].id] == JSON_PAYLOADS
        assert payloads_by_channel_id[channels_by_encoding["cdr"].id] == CDR_PAYLOADS

    report = diagnose(output)
    assert report.conforming, report.summary()
