"""Shared writers for hand-built compressed-video MCAP test inputs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from types import SimpleNamespace

from foxglove_schemas_protobuf.CompressedVideo_pb2 import CompressedVideo
from mcap.writer import Writer as StockWriter
from mcap_protobuf.schema import build_file_descriptor_set
from mcap_ros2.writer import Writer as Ros2Writer

ANNEX_B_START_CODE = b"\x00\x00\x00\x01"
KEYFRAME_ACCESS_UNIT = b"".join(
    ANNEX_B_START_CODE + bytes([nal_type]) + (b"\x80payload" if nal_type == 0x65 else b"payload")
    for nal_type in (0x09, 0x67, 0x68, 0x65)
)
NON_KEYFRAME_ACCESS_UNIT = b"".join(
    ANNEX_B_START_CODE + bytes([nal_type]) + (b"\x80payload" if nal_type == 0x41 else b"payload")
    for nal_type in (0x09, 0x41)
)
ROS2_COMPRESSED_VIDEO_SCHEMA = "\n".join(
    [
        "builtin_interfaces/Time timestamp",
        "string frame_id",
        "uint8[] data",
        "string format",
        "=" * 80,
        "MSG: builtin_interfaces/Time",
        "int32 sec",
        "uint32 nanosec",
    ]
)

NANOSECONDS_PER_SECOND = 1_000_000_000


def write_compressed_video_mcap(
    path: Path,
    messages: Iterable[tuple[str, int, bytes]],
    *,
    video_format: str = "h264",
    frame_id_by_topic: Mapping[str, str] | None = None,
    empty_topics: Iterable[str] = (),
    metadata: Mapping[str, Mapping[str, str]] | None = None,
    number_messages_per_topic: bool = False,
    library: str = "test",
) -> dict[str, list[bytes]]:
    """Write ``foxglove.CompressedVideo`` messages and return each topic's payloads.

    ``messages`` is ``(topic, log_time_ns, access_unit_data)`` in write order;
    each message's timestamp equals its log time. Channels are registered in
    first-appearance order, then ``empty_topics`` as channels with no
    messages. A topic's frame id defaults to the topic without its slashes.
    ``number_messages_per_topic`` sets each message's ``sequence`` to its
    index within its topic instead of zero.
    """
    messages = list(messages)
    frame_id_by_topic = frame_id_by_topic or {}
    payloads_by_topic: dict[str, list[bytes]] = {}
    with path.open("wb") as stream:
        writer = StockWriter(stream)
        writer.start(profile="", library=library)
        schema_id = writer.register_schema(
            name="foxglove.CompressedVideo",
            encoding="protobuf",
            data=build_file_descriptor_set(CompressedVideo).SerializeToString(),
        )
        channel_ids = {
            topic: writer.register_channel(
                topic=topic, message_encoding="protobuf", schema_id=schema_id
            )
            for topic in dict.fromkeys(topic for topic, _log_time, _data in messages)
        }
        for topic in empty_topics:
            writer.register_channel(topic=topic, message_encoding="protobuf", schema_id=schema_id)
        for topic, log_time, access_unit_data in messages:
            message = CompressedVideo()
            message.timestamp.FromNanoseconds(log_time)
            message.frame_id = frame_id_by_topic.get(topic, topic.strip("/"))
            message.data = access_unit_data
            message.format = video_format
            payload = message.SerializeToString()
            topic_payloads = payloads_by_topic.setdefault(topic, [])
            writer.add_message(
                channel_ids[topic],
                log_time=log_time,
                data=payload,
                publish_time=log_time,
                sequence=len(topic_payloads) if number_messages_per_topic else 0,
            )
            topic_payloads.append(payload)
        for name, record in (metadata or {}).items():
            writer.add_metadata(name, dict(record))
        writer.finish()
    return payloads_by_topic


def write_ros2_compressed_video_mcap(
    path: Path,
    access_unit_data: bytes,
    *,
    log_time: int,
    topic: str = "/cam",
    schema_text: str = ROS2_COMPRESSED_VIDEO_SCHEMA,
    **extra_fields: str,
) -> None:
    """Write one CDR ``foxglove_msgs/msg/CompressedVideo`` message.

    The message timestamp equals ``log_time``. ``extra_fields`` are set on
    the message alongside the standard ones, for schemas that declare more.
    """
    writer = Ros2Writer(str(path))
    schema = writer.register_msgdef("foxglove_msgs/msg/CompressedVideo", schema_text)
    writer.write_message(
        topic,
        schema,
        SimpleNamespace(
            timestamp=SimpleNamespace(
                sec=log_time // NANOSECONDS_PER_SECOND,
                nanosec=log_time % NANOSECONDS_PER_SECOND,
            ),
            frame_id=topic.strip("/"),
            **extra_fields,
            data=access_unit_data,
            format="h264",
        ),
        log_time=log_time,
        publish_time=log_time,
    )
    writer.finish()
