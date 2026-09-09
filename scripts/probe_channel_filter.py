from collections import Counter
from pathlib import Path

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

calls = []
seen_topics = []


def traced_iter_messages(*args, **kwargs):
    topics = kwargs.get(
        "topics",
        args[0] if args else None,
    )

    calls.append(topics)

    for schema, channel, message in original_iter_messages(*args, **kwargs):
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