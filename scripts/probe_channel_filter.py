from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from mcap.writer import CompressionType, Writer

from hflow.episode import Episode

path = Path("/tmp/hflow-channel-filter-probe.mcap")

with path.open("wb") as f:
    writer = Writer(
        f,
        chunk_size=64 * 1024,
        compression=CompressionType.NONE,
    )
    writer.start()

    target = writer.register_channel(
        topic="/target",
        message_encoding="json",
        schema_id=0,
    )

    camera = writer.register_channel(
        topic="/camera",
        message_encoding="json",
        schema_id=0,
    )

    writer.add_message(
        channel_id=target,
        log_time=1,
        publish_time=1,
        data=b"x" * 100_000,
    )

    for i in range(8):
        writer.add_message(
            channel_id=camera,
            log_time=10 + i,
            publish_time=10 + i,
            data=b"y" * 100_000,
        )

    writer.finish()


ep = Episode(path)

hflow_reader = ep._reader
mcap_reader = hflow_reader._reader

original_iter_messages = mcap_reader.iter_messages

calls: list[list[str] | None] = []
seen_topics: list[str] = []


def traced_iter_messages(
    topics: list[str] | None = None,
    start_time: int | None = None,
    end_time: int | None = None,
    log_time_order: bool = False,
) -> Iterator[Any]:
    calls.append(topics)

    for schema, channel, message in original_iter_messages(
        topics=topics,
        start_time=start_time,
        end_time=end_time,
        log_time_order=log_time_order,
    ):
        seen_topics.append(channel.topic)
        yield schema, channel, message


mcap_reader.iter_messages = traced_iter_messages

result = ep.channel("/target")

print("requested channel: /target")
print("messages returned by HFlow:", len(result))
print("topics passed to MCAP:", calls)
print("messages yielded by MCAP before HFlow filter:")
print(Counter(seen_topics))

ep.close()
path.unlink()
