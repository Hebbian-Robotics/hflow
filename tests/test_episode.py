from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from mcap.writer import CompressionType, Writer

from hflow.episode import Episode


def test_episode_channel_propagates_topic_filter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mcap_path = tmp_path / "test_channel_filter.mcap"

    with mcap_path.open("wb") as f:
        writer = Writer(
            f,
            chunk_size=64 * 1024,
            compression=CompressionType.NONE,
        )
        writer.start()

        target_channel_id = writer.register_channel(
            topic="/target",
            message_encoding="json",
            schema_id=0,
        )
        camera_channel_id = writer.register_channel(
            topic="/camera",
            message_encoding="json",
            schema_id=0,
        )

        writer.add_message(
            channel_id=target_channel_id,
            log_time=1,
            publish_time=1,
            data=b'{"value": "target"}',
        )
        writer.add_message(
            channel_id=camera_channel_id,
            log_time=2,
            publish_time=2,
            data=b'{"value": "camera"}',
        )

        writer.finish()

    ep = Episode(mcap_path)

    original_iter_batches = ep._reader.iter_batches
    topics_passed: list[list[str] | None] = []
    channel_ids_passed: list[list[int] | None] = []

    def traced_iter_batches(
        topics: list[str] | None = None,
        channel_ids: list[int] | None = None,
    ) -> Iterator[Any]:
        topics_passed.append(topics)
        channel_ids_passed.append(channel_ids)
        yield from original_iter_batches(
            topics=topics,
            channel_ids=channel_ids,
        )

    monkeypatch.setattr(ep._reader, "iter_batches", traced_iter_batches)

    channel = ep.channel("/target")

    assert topics_passed == [["/target"]]
    assert channel_ids_passed == [[target_channel_id]]
    assert channel.topic == "/target"
    assert channel.channel_id == target_channel_id
    assert len(channel) == 1
    assert channel.raw == [b'{"value": "target"}']

    ep.close()


def test_episode_channel_keeps_channel_id_filter_for_duplicate_topics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mcap_path = tmp_path / "test_duplicate_topic.mcap"

    with mcap_path.open("wb") as f:
        writer = Writer(
            f,
            chunk_size=64 * 1024,
            compression=CompressionType.NONE,
        )
        writer.start()

        first_channel_id = writer.register_channel(
            topic="/target",
            message_encoding="json",
            schema_id=0,
        )
        second_channel_id = writer.register_channel(
            topic="/target",
            message_encoding="json",
            schema_id=0,
        )

        writer.add_message(
            channel_id=first_channel_id,
            log_time=1,
            publish_time=1,
            data=b'{"channel": 1}',
        )
        writer.add_message(
            channel_id=second_channel_id,
            log_time=2,
            publish_time=2,
            data=b'{"channel": 2}',
        )

        writer.finish()

    ep = Episode(mcap_path)

    original_iter_batches = ep._reader.iter_batches
    topics_passed: list[list[str] | None] = []
    channel_ids_passed: list[list[int] | None] = []

    def traced_iter_batches(
        topics: list[str] | None = None,
        channel_ids: list[int] | None = None,
    ) -> Iterator[Any]:
        topics_passed.append(topics)
        channel_ids_passed.append(channel_ids)
        yield from original_iter_batches(
            topics=topics,
            channel_ids=channel_ids,
        )

    monkeypatch.setattr(ep._reader, "iter_batches", traced_iter_batches)

    channel = ep.channel(first_channel_id)

    assert topics_passed == [["/target"]]
    assert channel_ids_passed == [[first_channel_id]]
    assert channel.topic == "/target"
    assert channel.channel_id == first_channel_id
    assert len(channel) == 1
    assert channel.raw == [b'{"channel": 1}']

    ep.close()
