"""Canonical MCAP writer: topic-group chunking.

No open-source MCAP writer implements topic-group chunking (the chunk
layout described in Dyna's article); this one does. Each channel is
assigned to a named group at registration; each group maintains its own
chunk buffer and flushes its own chunks, so topics in different groups
NEVER share a chunk and each group's chunk stream is time-major. Chunking
changes write order, not the format: output must be spec-conforming MCAP
readable by the stock ``mcap`` package, Foxglove, and Rerun.

Implementation notes (mirrors ``mcap.writer.Writer``, which hardcodes a
single chunk builder):

- Records serialize themselves via ``mcap.records.*.write(RecordBuilder)``;
  build on that, but do NOT import the private ``mcap._chunk_builder`` --
  keep a local per-group chunk builder (~40 lines).
- Schema and Channel records are written unchunked into the data section at
  registration time (spec-legal) and repeated in the summary section.
- When a group's uncompressed buffer exceeds ``chunk_size`` bytes, that
  group's chunk is finalized: Chunk record, then its MessageIndex records,
  and a ChunkIndex retained for the summary -- exactly like the stock
  writer's ``__finalize_chunk``.
- ``finish()`` finalizes all groups (registration order), then mirrors the
  stock writer: DataEnd, summary (schemas, channels, statistics, chunk
  indexes, attachment indexes, metadata indexes), summary offsets, footer
  with summary CRC, closing magic.
- Chunk compression: zstd (default) or none. CRCs enabled as in the stock
  writer defaults.
"""

import struct
import tempfile
import zlib
from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import IO, Literal

import zstandard
from mcap.data_stream import RecordBuilder
from mcap.opcode import Opcode
from mcap.records import (
    Attachment,
    AttachmentIndex,
    Channel,
    Chunk,
    ChunkIndex,
    DataEnd,
    Footer,
    Header,
    McapRecord,
    Message,
    MessageIndex,
    Metadata,
    MetadataIndex,
    Schema,
    Statistics,
    SummaryOffset,
)

from hflow.behavior import TRANSFORM_BEHAVIOR_VERSION
from hflow.format import DEFAULT_CHUNK_SIZE_BYTES, EPISODE_FORMAT_VERSION

MCAP0_MAGIC = struct.pack("<8B", 137, 77, 67, 65, 80, 48, 13, 10)


def _default_library_identifier() -> str:
    """Project name + episode format version + transform behavior version.

    Informational only (MCAP Header ``library`` field): stored-data
    identifiers stay neutral per ``hflow.format``.
    """
    # Deliberately free of the release number: this string is written into
    # the MCAP header, so it is part of the bytes content_episode_id hashes.
    # Embedding hflow.__version__ here gave a byte-identical input a new
    # episode identity on every release, breaking content-addressed dedupe.
    # The behavior version changes only when canonicalization does.
    return (
        f"hflow episode-format/{EPISODE_FORMAT_VERSION} "
        f"transform-behavior/{TRANSFORM_BEHAVIOR_VERSION}"
    )


class _GroupChunkBuilder:
    """Accumulates one group's Message records destined for its next chunk.

    Local reimplementation of the record-level logic in the private
    ``mcap._chunk_builder.ChunkBuilder``, minus schema/channel handling
    (schemas and channels are written unchunked in the data section).
    """

    def __init__(self) -> None:
        self.record_builder = RecordBuilder()
        self.message_indices: dict[int, MessageIndex] = {}
        self.message_start_time = 0
        self.message_end_time = 0
        self.message_count = 0

    @property
    def uncompressed_size(self) -> int:
        return self.record_builder.count

    def add_message(self, message: Message) -> None:
        if self.message_count == 0:
            self.message_start_time = message.log_time
        else:
            self.message_start_time = min(self.message_start_time, message.log_time)
        self.message_end_time = max(self.message_end_time, message.log_time)

        message_index = self.message_indices.get(message.channel_id)
        if message_index is None:
            message_index = MessageIndex(channel_id=message.channel_id, records=[])
            self.message_indices[message.channel_id] = message_index
        # Offset is relative to the uncompressed chunk data, per the MCAP spec.
        message_index.records.append((message.log_time, self.record_builder.count))

        self.message_count += 1
        message.write(self.record_builder)

    def take_chunk_data(self) -> bytes:
        """Return the accumulated uncompressed chunk data and clear the buffer.

        ``message_indices`` and the time range survive until :meth:`reset` so
        the caller can emit MessageIndex records and the ChunkIndex first.
        """
        return self.record_builder.end()

    def reset(self) -> None:
        self.message_indices.clear()
        self.message_start_time = 0
        self.message_end_time = 0
        self.message_count = 0


class CanonicalMcapWriter:
    """MCAP writer with per-group chunk streams.

    Usage::

        with CanonicalMcapWriter(path) as writer:
            schema_id = writer.register_schema(name, encoding, data)
            channel_id = writer.register_channel(
                topic, message_encoding, schema_id, group="cameras"
            )
            writer.write_message(channel_id, log_time, data, publish_time)
            writer.add_metadata("episode/v1", {...})
    """

    def __init__(
        self,
        output: Path | str | IO[bytes],
        *,
        chunk_size: int | Mapping[str, int] = DEFAULT_CHUNK_SIZE_BYTES,
        compression: Literal["zstd", "none"] = "zstd",
        library: str | None = None,
    ) -> None:
        """``chunk_size`` is one target for every group, or a per-group
        mapping with :data:`DEFAULT_CHUNK_SIZE_BYTES` for groups it omits.

        Per-group targets exist because groups are written at wildly different
        byte rates: six cameras produce megabytes a second while a proprio
        channel produces kilobytes, and one threshold cannot be right for both
        (see :func:`hflow.format.derived_chunk_size_bytes`). A plain int stays
        accepted because it is the public contract callers already pass."""
        if isinstance(output, str | Path):
            # Path outputs publish atomically: all bytes go to a sibling temp
            # file, and finish() replaces the destination only after the
            # closing magic and summary have been written successfully. A
            # failed rewrite therefore preserves the previous valid episode.
            self._output_path: Path | None = Path(output)
            temporary_stream = tempfile.NamedTemporaryFile(  # noqa: SIM115
                mode="w+b",
                dir=self._output_path.parent,
                prefix=f".{self._output_path.name}.",
                suffix=".tmp",
                delete=False,
            )
            self._temporary_output_path: Path | None = Path(temporary_stream.name)
            self._stream: IO[bytes] = temporary_stream
            self._owns_stream = True
        else:
            self._output_path = None
            self._temporary_output_path = None
            self._stream = output
            self._owns_stream = False

        self._chunk_size_by_group: Mapping[str, int] = (
            {} if isinstance(chunk_size, int) else dict(chunk_size)
        )
        self._default_chunk_size = (
            chunk_size if isinstance(chunk_size, int) else (DEFAULT_CHUNK_SIZE_BYTES)
        )
        self._compression: Literal["zstd", "none"] = compression
        self._finished = False

        self._record_builder = RecordBuilder()
        self._schemas_by_id: OrderedDict[int, Schema] = OrderedDict()
        self._channels_by_id: OrderedDict[int, Channel] = OrderedDict()
        self._group_name_by_channel_id: dict[int, str] = {}
        # Insertion order is group-registration order, honored at finish().
        self._chunk_builders_by_group: OrderedDict[str, _GroupChunkBuilder] = OrderedDict()
        self._chunk_indices: list[ChunkIndex] = []
        self._attachment_indexes: list[AttachmentIndex] = []
        self._metadata_indexes: list[MetadataIndex] = []
        self._statistics = Statistics(
            attachment_count=0,
            channel_count=0,
            channel_message_counts=defaultdict(int),
            chunk_count=0,
            message_count=0,
            metadata_count=0,
            message_start_time=0,
            message_end_time=0,
            schema_count=0,
        )

        library_identifier = library if library is not None else _default_library_identifier()
        self._stream.write(MCAP0_MAGIC)
        Header(profile="", library=library_identifier).write(self._record_builder)
        self._flush()

    def register_schema(self, name: str, encoding: str, data: bytes) -> int:
        self._raise_if_finished("register a schema")
        schema_id = len(self._schemas_by_id) + 1
        schema = Schema(id=schema_id, data=data, encoding=encoding, name=name)
        self._schemas_by_id[schema_id] = schema
        self._statistics.schema_count += 1
        # Unchunked into the data section (spec-legal); repeated in the summary.
        schema.write(self._record_builder)
        self._flush()
        return schema_id

    def register_channel(
        self,
        topic: str,
        message_encoding: str,
        schema_id: int,
        *,
        group: str,
        metadata: dict[str, str] | None = None,
    ) -> int:
        self._raise_if_finished("register a channel")
        # schema_id 0 is the MCAP spec's "no schema" sentinel.
        if schema_id != 0 and schema_id not in self._schemas_by_id:
            raise ValueError(
                f"cannot register channel {topic!r}: unknown schema id {schema_id} "
                f"(registered schema ids: {sorted(self._schemas_by_id)})"
            )
        channel_id = len(self._channels_by_id) + 1
        channel = Channel(
            id=channel_id,
            topic=topic,
            message_encoding=message_encoding,
            schema_id=schema_id,
            metadata=metadata if metadata is not None else {},
        )
        self._channels_by_id[channel_id] = channel
        self._group_name_by_channel_id[channel_id] = group
        if group not in self._chunk_builders_by_group:
            self._chunk_builders_by_group[group] = _GroupChunkBuilder()
        self._statistics.channel_count += 1
        # Unchunked into the data section (spec-legal); repeated in the summary.
        channel.write(self._record_builder)
        self._flush()
        return channel_id

    def write_message(
        self,
        channel_id: int,
        log_time: int,
        data: bytes,
        publish_time: int | None = None,
        sequence: int = 0,
    ) -> None:
        self._raise_if_finished("write a message")
        group_name = self._group_name_by_channel_id.get(channel_id)
        if group_name is None:
            raise ValueError(
                f"cannot write message: unknown channel id {channel_id} "
                f"(registered channel ids: {sorted(self._channels_by_id)})"
            )
        message = Message(
            channel_id=channel_id,
            log_time=log_time,
            data=data,
            publish_time=log_time if publish_time is None else publish_time,
            sequence=sequence,
        )

        if self._statistics.message_count == 0:
            self._statistics.message_start_time = log_time
        else:
            self._statistics.message_start_time = min(log_time, self._statistics.message_start_time)
        self._statistics.message_end_time = max(log_time, self._statistics.message_end_time)
        self._statistics.channel_message_counts[channel_id] += 1
        self._statistics.message_count += 1

        chunk_builder = self._chunk_builders_by_group[group_name]
        chunk_builder.add_message(message)
        group_chunk_size = self._chunk_size_by_group.get(group_name, self._default_chunk_size)
        if chunk_builder.uncompressed_size > group_chunk_size:
            self._finalize_group_chunk(group_name)

    def add_metadata(self, name: str, data: dict[str, str]) -> None:
        self._raise_if_finished("add metadata")
        self._flush()
        offset = self._stream.tell()
        self._statistics.metadata_count += 1
        Metadata(name=name, metadata=data).write(self._record_builder)
        self._metadata_indexes.append(
            MetadataIndex(offset=offset, length=self._record_builder.count, name=name)
        )
        self._flush()

    def add_attachment(
        self,
        name: str,
        media_type: str,
        data: bytes,
        *,
        log_time: int = 0,
        create_time: int = 0,
    ) -> None:
        self._raise_if_finished("add an attachment")
        self._flush()
        offset = self._stream.tell()
        self._statistics.attachment_count += 1
        attachment = Attachment(
            create_time=create_time,
            log_time=log_time,
            name=name,
            media_type=media_type,
            data=data,
        )
        attachment.write(self._record_builder)
        self._attachment_indexes.append(
            AttachmentIndex(
                offset=offset,
                length=self._record_builder.count,
                create_time=create_time,
                log_time=log_time,
                data_size=len(data),
                name=name,
                media_type=media_type,
            )
        )
        self._flush()

    def finish(self) -> None:
        if self._finished:
            return
        try:
            for group_name in self._chunk_builders_by_group:
                self._finalize_group_chunk(group_name)

            # No data-section CRC, matching the stock writer's defaults.
            DataEnd(data_section_crc=0).write(self._record_builder)
            self._flush()

            summary_start = self._stream.tell()
            summary_builder = RecordBuilder()
            summary_offsets: list[SummaryOffset] = []

            def write_summary_group(group_opcode: Opcode, records: Sequence[McapRecord]) -> None:
                group_start = summary_builder.count
                for record in records:
                    record.write(summary_builder)
                summary_offsets.append(
                    SummaryOffset(
                        group_opcode=group_opcode,
                        group_start=summary_start + group_start,
                        group_length=summary_builder.count - group_start,
                    )
                )

            write_summary_group(Opcode.SCHEMA, list(self._schemas_by_id.values()))
            write_summary_group(Opcode.CHANNEL, list(self._channels_by_id.values()))

            statistics_group_start = summary_builder.count
            self._statistics.write(summary_builder)
            summary_offsets.append(
                SummaryOffset(
                    group_opcode=Opcode.STATISTICS,
                    group_start=summary_start + statistics_group_start,
                    group_length=summary_builder.count - statistics_group_start,
                )
            )

            write_summary_group(Opcode.CHUNK_INDEX, self._chunk_indices)
            write_summary_group(Opcode.ATTACHMENT_INDEX, self._attachment_indexes)
            write_summary_group(Opcode.METADATA_INDEX, self._metadata_indexes)

            summary_offset_start = summary_start + summary_builder.count
            for summary_offset in summary_offsets:
                summary_offset.write(summary_builder)

            summary_data = summary_builder.end()
            summary_length = len(summary_data)

            # Footer summary_crc covers the summary section plus the footer record
            # up through (but excluding) the crc field itself, exactly like the
            # stock writer.
            summary_crc = zlib.crc32(summary_data)
            summary_crc = zlib.crc32(
                struct.pack(
                    "<BQQQ",
                    Opcode.FOOTER,
                    8 + 8 + 4,
                    0 if summary_length == 0 else summary_start,
                    summary_offset_start,
                ),
                summary_crc,
            )

            self._stream.write(summary_data)
            Footer(
                summary_start=0 if summary_length == 0 else summary_start,
                summary_offset_start=summary_offset_start,
                summary_crc=summary_crc,
            ).write(self._record_builder)
            self._flush()
            self._stream.write(MCAP0_MAGIC)
            if self._owns_stream:
                self._stream.close()
                assert self._temporary_output_path is not None
                assert self._output_path is not None
                self._temporary_output_path.replace(self._output_path)
        except BaseException:
            self._discard_temporary_output()
            self._finished = True
            raise
        self._finished = True

    def _finalize_group_chunk(self, group_name: str) -> None:
        chunk_builder = self._chunk_builders_by_group[group_name]
        if chunk_builder.message_count == 0:
            return

        self._statistics.chunk_count += 1

        chunk_data = chunk_builder.take_chunk_data()
        if self._compression == "zstd":
            compression_identifier = "zstd"
            compressed_data = zstandard.compress(chunk_data)
        else:
            compression_identifier = ""
            compressed_data = chunk_data
        chunk = Chunk(
            compression=compression_identifier,
            data=compressed_data,
            message_start_time=chunk_builder.message_start_time,
            message_end_time=chunk_builder.message_end_time,
            uncompressed_crc=zlib.crc32(chunk_data),
            uncompressed_size=len(chunk_data),
        )

        self._flush()
        chunk_start_offset = self._stream.tell()
        chunk.write(self._record_builder)
        chunk_record_length = self._record_builder.count

        chunk_index = ChunkIndex(
            message_start_time=chunk.message_start_time,
            message_end_time=chunk.message_end_time,
            chunk_start_offset=chunk_start_offset,
            chunk_length=chunk_record_length,
            message_index_offsets={},
            message_index_length=0,
            compression=chunk.compression,
            compressed_size=len(compressed_data),
            uncompressed_size=chunk.uncompressed_size,
        )

        self._flush()
        message_index_start_offset = self._stream.tell()
        for channel_id, message_index in chunk_builder.message_indices.items():
            chunk_index.message_index_offsets[channel_id] = (
                message_index_start_offset + self._record_builder.count
            )
            message_index.write(self._record_builder)
        chunk_index.message_index_length = self._record_builder.count
        self._flush()

        self._chunk_indices.append(chunk_index)
        chunk_builder.reset()

    def abort(self) -> None:
        """Abandon the write and delete only the unpublished temporary file.

        The destination is atomically replaced only by :meth:`finish`, so an
        aborted rewrite never destroys a previously published valid episode.
        Called automatically when an exception escapes the ``with`` block.
        """
        if self._finished:
            return
        self._finished = True
        self._discard_temporary_output()

    def _discard_temporary_output(self) -> None:
        if not self._owns_stream:
            return
        if not self._stream.closed:
            self._stream.close()
        if self._temporary_output_path is not None:
            self._temporary_output_path.unlink(missing_ok=True)

    def _flush(self) -> None:
        self._stream.write(self._record_builder.end())

    def _raise_if_finished(self, operation: str) -> None:
        if self._finished:
            raise RuntimeError(f"cannot {operation}: finish() was already called on this writer")

    def __enter__(self) -> "CanonicalMcapWriter":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.finish()
        else:
            self.abort()
