"""The batch-oriented episode read interface: the Rust-swappable seam.

Episode reads go through :class:`EpisodeReader`, a batch interface -- decoded
batches in, arrays/bytes out -- never per-message callbacks in the hot loop.
The interface shape is the part that cannot be retrofitted: a future Rust
backend (PyO3 over the official mcap crate, shipped as a separate wheel) must
drop in behind this protocol unchanged, parity-tested against the pure-Python
backend below. This is the frozen shape that backend implements: the seam is
keyed by CHANNEL id (``channels()`` is the authoritative accessor; parity
tests operate on it), and ``topics()`` is a derived convenience view that
refuses ambiguous files.

Guarantees:

- Within a channel, batches and messages within a batch are ascending in
  ``log_time``. No ordering is guaranteed across channels.
- ``data`` payloads are raw encoded message bytes; decoding happens above the
  seam (see ``hflow.episode``).
"""

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import IO, Protocol

import numpy as np
from mcap.reader import McapReader, make_reader
from mcap.records import Attachment

logger = logging.getLogger(__name__)

DEFAULT_BATCH_MAX_MESSAGES = 1024
DEFAULT_BATCH_MAX_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class TopicInfo:
    """Static description of one CHANNEL in an episode.

    The name is historical: MCAP allows several channels per topic, and one
    ``TopicInfo`` describes exactly one of them (identified by ``channel_id``).
    """

    topic: str
    channel_id: int
    schema_name: str
    schema_encoding: str
    message_encoding: str
    message_count: int
    schema_data: bytes = field(repr=False)
    # False when the channel's schema_id is the MCAP "no schema" sentinel (0);
    # distinguishes that from a schema record that happens to be empty.
    has_schema: bool = True


@dataclass(frozen=True)
class EpisodeTimeBounds:
    """The log-time span one episode's messages cover, in nanoseconds.

    Read from the MCAP summary statistics, so it costs no message decode and
    covers every channel, not only the ones a caller happened to load.
    ``start_ns == end_ns`` is a real single-instant episode.
    """

    start_ns: int
    end_ns: int

    @property
    def duration_s(self) -> float:
        return (self.end_ns - self.start_ns) / 1_000_000_000


@dataclass
class MessageBatch:
    """A contiguous run of raw messages from one channel, ascending in log time.

    Batches are per-CHANNEL: two channels sharing a topic yield separate
    batches, distinguished by ``channel_id``.
    """

    topic: str
    channel_id: int
    log_times: np.ndarray  # int64 nanoseconds, shape (n,)
    publish_times: np.ndarray  # int64 nanoseconds, shape (n,)
    data: list[bytes]

    def __len__(self) -> int:
        return len(self.data)


class EpisodeReader(Protocol):
    """Batch read access to one episode file."""

    def channels(self) -> dict[int, TopicInfo]:
        """All channels in the file, keyed by channel id. The authoritative
        accessor: it represents every channel, including several per topic."""
        ...

    def topics(self) -> dict[str, TopicInfo]:
        """The channels keyed by topic name -- a derived view of
        :meth:`channels` that raises ``ValueError`` when any topic has more
        than one channel. Callers able to handle duplicates use
        :meth:`channels` instead."""
        ...

    def metadata(self) -> dict[str, dict[str, str]]:
        """All MCAP Metadata records, keyed by record name (last record wins)."""
        ...

    def time_bounds(self) -> EpisodeTimeBounds | None:
        """The log-time span of every message in the file, or ``None`` when
        the file records no statistics or holds no messages."""
        ...

    def attachments(self) -> Iterator[Attachment]:
        """All MCAP Attachment records."""
        ...

    def iter_batches(
        self,
        topics: Sequence[str] | None = None,
        start_ns: int | None = None,
        end_ns: int | None = None,
        *,
        channel_ids: Sequence[int] | None = None,
        batch_max_messages: int = DEFAULT_BATCH_MAX_MESSAGES,
        batch_max_bytes: int = DEFAULT_BATCH_MAX_BYTES,
    ) -> Iterator[MessageBatch]:
        """Yield per-channel batches, optionally filtered by topic, channel
        id, and time range (filters intersect when several are given)."""
        ...

    def close(self) -> None:
        """Release underlying file handles and resources."""
        ...


class PythonMcapEpisodeReader:
    """Pure-Python :class:`EpisodeReader` backend over the stock ``mcap`` package."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._stream: IO[bytes] = self.path.open("rb")
        try:
            self._reader: McapReader = make_reader(self._stream)
        except BaseException:
            # make_reader validates the magic bytes; don't leak the handle
            # when it rejects a non-MCAP, empty, or truncated file.
            self._stream.close()
            raise
        self._channels: dict[int, TopicInfo] | None = None

    def channels(self) -> dict[int, TopicInfo]:
        if self._channels is not None:
            return self._channels
        summary = self._reader.get_summary()
        if summary is None:
            raise ValueError(
                f"{self.path} has no MCAP summary section (unindexed or truncated file). "
                "Re-record or rewrite the file with a conforming writer."
            )
        message_counts: dict[int, int] = {}
        if summary.statistics is not None:
            message_counts = dict(summary.statistics.channel_message_counts)
        infos: dict[int, TopicInfo] = {}
        for channel in summary.channels.values():
            schema = summary.schemas.get(channel.schema_id)
            infos[channel.id] = TopicInfo(
                topic=channel.topic,
                channel_id=channel.id,
                schema_name=schema.name if schema else "",
                schema_encoding=schema.encoding if schema else "",
                message_encoding=channel.message_encoding,
                message_count=message_counts.get(channel.id, 0),
                schema_data=schema.data if schema else b"",
                has_schema=schema is not None,
            )
        self._channels = infos
        return infos

    def topics(self) -> dict[str, TopicInfo]:
        channel_ids_by_topic: dict[str, list[int]] = {}
        for info in self.channels().values():
            channel_ids_by_topic.setdefault(info.topic, []).append(info.channel_id)
        duplicated = {
            topic: sorted(ids) for topic, ids in channel_ids_by_topic.items() if len(ids) > 1
        }
        if duplicated:
            # MCAP allows several channels per topic; a topic-keyed view would
            # silently merge them under one schema -- refuse loudly instead.
            described = ", ".join(
                f"{topic!r} (channel ids {ids})" for topic, ids in sorted(duplicated.items())
            )
            raise ValueError(
                f"{self.path} has multiple channels for topic {described}; "
                "the topic-keyed view cannot represent them -- use channels() instead"
            )
        return {info.topic: info for info in self.channels().values()}

    def time_bounds(self) -> EpisodeTimeBounds | None:
        summary = self._reader.get_summary()
        if summary is None or summary.statistics is None:
            return None
        if summary.statistics.message_count == 0:
            return None
        return EpisodeTimeBounds(
            start_ns=int(summary.statistics.message_start_time),
            end_ns=int(summary.statistics.message_end_time),
        )

    def metadata(self) -> dict[str, dict[str, str]]:
        records: dict[str, dict[str, str]] = {}
        for record in self._reader.iter_metadata():
            if record.name in records:
                logger.warning(
                    "duplicate metadata record %r in %s: keeping the later one",
                    record.name,
                    self.path,
                )
            records[record.name] = dict(record.metadata)
        return records

    def attachments(self) -> Iterator[Attachment]:
        return self._reader.iter_attachments()

    def iter_batches(
        self,
        topics: Sequence[str] | None = None,
        start_ns: int | None = None,
        end_ns: int | None = None,
        *,
        channel_ids: Sequence[int] | None = None,
        batch_max_messages: int = DEFAULT_BATCH_MAX_MESSAGES,
        batch_max_bytes: int = DEFAULT_BATCH_MAX_BYTES,
    ) -> Iterator[MessageBatch]:
        wanted_channel_ids = frozenset(channel_ids) if channel_ids is not None else None
        topics_by_channel_id: dict[int, str] = {}
        pending_log_times: dict[int, list[int]] = {}
        pending_publish_times: dict[int, list[int]] = {}
        pending_data: dict[int, list[bytes]] = {}
        pending_bytes: dict[int, int] = {}

        def flush(channel_id: int) -> MessageBatch:
            batch = MessageBatch(
                topic=topics_by_channel_id[channel_id],
                channel_id=channel_id,
                log_times=np.asarray(pending_log_times.pop(channel_id), dtype=np.int64),
                publish_times=np.asarray(pending_publish_times.pop(channel_id), dtype=np.int64),
                data=pending_data.pop(channel_id),
            )
            pending_bytes.pop(channel_id)
            return batch

        for _schema, channel, message in self._reader.iter_messages(
            topics=list(topics) if topics is not None else None,
            start_time=start_ns,
            end_time=end_ns,
            log_time_order=True,
        ):
            # The stock reader filters by topic only; the channel-id filter is
            # applied here in Python (fine for the pure-Python backend).
            if wanted_channel_ids is not None and channel.id not in wanted_channel_ids:
                continue
            channel_id = channel.id
            topics_by_channel_id[channel_id] = channel.topic
            pending_log_times.setdefault(channel_id, []).append(message.log_time)
            pending_publish_times.setdefault(channel_id, []).append(message.publish_time)
            pending_data.setdefault(channel_id, []).append(message.data)
            pending_bytes[channel_id] = pending_bytes.get(channel_id, 0) + len(message.data)
            if (
                len(pending_data[channel_id]) >= batch_max_messages
                or pending_bytes[channel_id] >= batch_max_bytes
            ):
                yield flush(channel_id)

        for channel_id in list(pending_data):
            yield flush(channel_id)

    def close(self) -> None:
        self._stream.close()

    def __enter__(self) -> "PythonMcapEpisodeReader":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def open_reader(path: Path | str) -> EpisodeReader:
    """Open an episode file with the default (pure-Python) reader backend."""
    return PythonMcapEpisodeReader(path)
