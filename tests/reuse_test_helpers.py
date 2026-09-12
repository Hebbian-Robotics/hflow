"""Shared helpers for reusing-episode and delivery-corruption tests."""

from __future__ import annotations

import io
import struct
from pathlib import Path

from hflow.catalog import content_episode_id


def flip_chunk_payload_bytes(episode_path: Path, *, count: int = 4) -> None:
    """Corrupt message-data bytes in place, leaving the file valid.

    Locates the first chunk record and flips ``count`` bytes inside its
    message-data records (the tail of the chunk's uncompressed records
    region). The MCAP summary, indexes, and metadata records stay intact;
    the chunk CRC no longer matches the damaged bytes, which is exactly what
    a CRC-validated read catches and what on-disk bit rot or a partial
    external copy looks like when the container survives. The fixture
    writer emits uncompressed chunks (``compression=""``), so the records
    region is plaintext.
    """
    from mcap.reader import make_reader

    data = bytearray(episode_path.read_bytes())
    summary = make_reader(io.BytesIO(bytes(data))).get_summary()
    if summary is None or not summary.chunk_indexes:
        raise ValueError(f"{episode_path} has no chunk records to corrupt")
    chunk_index_record = summary.chunk_indexes[0]
    chunk_start = chunk_index_record.chunk_start_offset
    # Chunk record per the MCAP spec: opcode(1) length(8) then
    # message_start_time(8) message_end_time(8) uncompressed_size(8)
    # uncompressed_crc(4) compression_length(4) compression records_length(8)
    # records.
    compression_length = struct.unpack_from("<I", data, chunk_start + 37)[0]
    record_length = struct.unpack_from("<Q", data, chunk_start + 1)[0]
    records_start = chunk_start + 9 + 8 + 8 + 8 + 4 + 4 + compression_length + 8
    records_end = chunk_start + 9 + record_length
    if records_end - records_start < count:
        raise ValueError("chunk records region too small for the requested corruption")
    for offset in range(records_end - count, records_end):
        data[offset] ^= 0xFF
    episode_path.write_bytes(bytes(data))


def content_id_differs_from_delivery_receipt(episode_path: Path, receipt_content_id: str) -> bool:
    """True when the file on disk no longer matches the recorded content id."""
    return content_episode_id(episode_path) != receipt_content_id


def corrupt_zstd_chunk_payload(episode_path: Path) -> None:
    """Invalidate the first chunk's zstd frame magic, inside compressed records.

    Use the summary chunk index and the same MCAP field layout as
    ``flip_chunk_payload_bytes``. Only the first byte of the zstd frame magic
    changes (0x28 -> 0x29), forcing the actual decompressor to reject it.
    The stored CRC, compression name, record lengths, and all MCAP headers,
    indexes, metadata, and footer remain byte-for-byte intact.
    """
    from mcap.reader import make_reader

    data = bytearray(episode_path.read_bytes())
    summary = make_reader(io.BytesIO(bytes(data))).get_summary()
    assert summary is not None and summary.chunk_indexes
    chunk_start = summary.chunk_indexes[0].chunk_start_offset
    compression_length = struct.unpack_from("<I", data, chunk_start + 37)[0]
    compression_start = chunk_start + 41
    assert data[compression_start : compression_start + compression_length] == b"zstd"
    records_start = compression_start + compression_length + 8
    assert data[records_start : records_start + 4] == b"\x28\xb5\x2f\xfd"
    data[records_start] ^= 0x01
    episode_path.write_bytes(data)
