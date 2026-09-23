from pathlib import Path

import pytest
from mcap.writer import CompressionType, Writer

from hflow.episode import Episode


def _write_two_json_channels(
    mcap_path: Path, topics: tuple[str, str], payloads: tuple[bytes, bytes]
) -> tuple[int, int]:
    """One message per channel, written in channel order; returns the channel ids."""
    with mcap_path.open("wb") as stream:
        writer = Writer(stream, chunk_size=64 * 1024, compression=CompressionType.NONE)
        writer.start()
        channel_ids = tuple(
            writer.register_channel(topic=topic, message_encoding="json", schema_id=0)
            for topic in topics
        )
        for log_time, (channel_id, payload) in enumerate(
            zip(channel_ids, payloads, strict=True), start=1
        ):
            writer.add_message(
                channel_id=channel_id, log_time=log_time, publish_time=log_time, data=payload
            )
        writer.finish()
    first_channel_id, second_channel_id = channel_ids
    return first_channel_id, second_channel_id


@pytest.mark.parametrize(
    ("topics", "payloads", "select_by_channel_id"),
    [
        pytest.param(
            ("/target", "/camera"),
            (b'{"value": "target"}', b'{"value": "camera"}'),
            False,
            id="by-topic-excludes-other-topics",
        ),
        # Two channels share a topic, so filtering on the topic alone would
        # return both messages; only the channel id tells them apart.
        pytest.param(
            ("/target", "/target"),
            (b'{"channel": 1}', b'{"channel": 2}'),
            True,
            id="by-channel-id-excludes-a-duplicate-topic",
        ),
    ],
)
def test_episode_channel_reads_only_the_selected_channel(
    tmp_path: Path,
    topics: tuple[str, str],
    payloads: tuple[bytes, bytes],
    select_by_channel_id: bool,
) -> None:
    mcap_path = tmp_path / "episode.mcap"
    selected_channel_id, _ = _write_two_json_channels(mcap_path, topics, payloads)

    with Episode(mcap_path) as episode:
        channel = episode.channel(selected_channel_id if select_by_channel_id else topics[0])

        assert channel.topic == "/target"
        assert channel.channel_id == selected_channel_id
        assert len(channel) == 1
        assert channel.raw == [payloads[0]]
