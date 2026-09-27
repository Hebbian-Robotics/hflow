"""Let FFmpeg read a remote video through a caller-supplied byte-range reader.

FFmpeg seeks inside a container and reads only the ranges it needs, but only
from inputs it can open. :func:`serve_byte_ranges` runs a loopback HTTP server
for the lifetime of a ``with`` block and returns a :class:`LoopbackVideoSource`
that :func:`hflow.media.probe_video` and
:func:`hflow.source_sampling.sample_source_frames` accept in place of a local
path. The caller owns storage, credentials, and object versions; nothing
about them reaches FFmpeg's arguments.

The server fetches fixed-size blocks from the reader only as FFmpeg consumes
them, and caches every block it fetched until the ``with`` block exits, because
each FFmpeg run re-reads the container's header. A reader failure is kept and
re-raised by the consuming call as the original exception, so a storage or
credential failure is never reported as unreadable media.
"""

from __future__ import annotations

import http.server
import os
import re
import secrets
import socket
import threading
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from hflow._field_guards import require_positive_int

_RANGE_PATTERN = re.compile(r"bytes=(\d+)-(\d*)")
_SOCKET_SEND_BUFFER_BYTES = 64 * 1024
_REQUEST_TIMEOUT_SECONDS = 60.0
# FFmpeg's HTTP client honors these even for 127.0.0.1, which would send the
# loopback URL to a proxy that cannot reach it.
_PROXY_ENVIRONMENT_VARIABLES = frozenset(
    {"http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"}
)


class ByteRangeReader(Protocol):
    """Random access to one immutable object of a fixed size.

    ``read_range`` must return exactly ``stop - start`` bytes or raise. Calls
    are serialized, so an implementation need not be thread-safe.
    """

    @property
    def size_bytes(self) -> int: ...

    def read_range(self, start: int, stop: int) -> bytes: ...


class _BlockCache:
    def __init__(
        self, reader: ByteRangeReader, *, block_bytes: int, maximum_cached_bytes: int
    ) -> None:
        self.reader = reader
        self.size_bytes = reader.size_bytes
        self.block_bytes = block_bytes
        self.maximum_cached_bytes = maximum_cached_bytes
        self.blocks: OrderedDict[int, bytes] = OrderedDict()
        self.cached_bytes = 0
        self.bytes_fetched = 0
        self.first_failure: BaseException | None = None
        self.lock = threading.Lock()

    def block(self, block_index: int) -> bytes:
        with self.lock:
            if self.first_failure is not None:
                raise OSError("an earlier byte-range read failed")
            cached = self.blocks.get(block_index)
            if cached is not None:
                self.blocks.move_to_end(block_index)
                return cached
            start = block_index * self.block_bytes
            stop = min(start + self.block_bytes, self.size_bytes)
            try:
                data = self.reader.read_range(start, stop)
                if not isinstance(data, bytes) or len(data) != stop - start:
                    raise OSError("byte-range reader returned the wrong number of bytes")
            except BaseException as error:
                self.first_failure = error
                raise
            self.bytes_fetched += len(data)
            self.blocks[block_index] = data
            self.cached_bytes += len(data)
            while self.cached_bytes > self.maximum_cached_bytes and len(self.blocks) > 1:
                _evicted_index, evicted = self.blocks.popitem(last=False)
                self.cached_bytes -= len(evicted)
            return data


class _RangeRequestHandler(http.server.BaseHTTPRequestHandler):
    server: _LoopbackServer
    timeout = _REQUEST_TIMEOUT_SECONDS

    def setup(self) -> None:
        super().setup()
        # A small send buffer bounds how far upstream reads run ahead of FFmpeg.
        self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, _SOCKET_SEND_BUFFER_BYTES)

    def log_message(self, format: str, *arguments: object) -> None:
        return

    def do_HEAD(self) -> None:
        self._respond(send_body=False)

    def do_GET(self) -> None:
        self._respond(send_body=True)

    def _respond(self, *, send_body: bool) -> None:
        if self.path != self.server.object_path:
            self.send_error(404)
            return
        cache = self.server.cache
        start, stop = 0, cache.size_bytes
        range_header = self.headers.get("Range")
        if range_header is not None:
            match = _RANGE_PATTERN.fullmatch(range_header.strip())
            if match is None or int(match.group(1)) >= cache.size_bytes:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{cache.size_bytes}")
                self.end_headers()
                return
            start = int(match.group(1))
            if match.group(2):
                stop = min(int(match.group(2)) + 1, cache.size_bytes)
            if stop <= start:
                self.send_error(416)
                return
        first_block = None
        if send_body:
            # Fetch before sending headers, so an immediate reader failure is a
            # server error rather than a truncated body.
            try:
                first_block = cache.block(start // cache.block_bytes)
            except BaseException:
                self.send_error(502)
                return
        self.send_response(206 if range_header is not None else 200)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(stop - start))
        if range_header is not None:
            self.send_header("Content-Range", f"bytes {start}-{stop - 1}/{cache.size_bytes}")
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()
        if not send_body or first_block is None:
            return
        position = start
        block_index = start // cache.block_bytes
        block = first_block
        while position < stop:
            block_start = block_index * cache.block_bytes
            chunk = block[
                position - block_start : min(stop, block_start + len(block)) - block_start
            ]
            try:
                self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                return
            position += len(chunk)
            if position >= stop:
                return
            block_index += 1
            try:
                block = cache.block(block_index)
            except BaseException:
                # Headers are already sent: end the body early so FFmpeg fails.
                self.close_connection = True
                return


class _LoopbackServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, cache: _BlockCache) -> None:
        super().__init__(("127.0.0.1", 0), _RangeRequestHandler)
        self.cache = cache
        self.object_path = "/" + secrets.token_urlsafe(24)


class LoopbackVideoSource:
    """A video served on 127.0.0.1 for as long as its ``with`` block runs."""

    def __init__(self, server: _LoopbackServer) -> None:
        self._server = server
        host, port = server.server_address[:2]
        self._url = f"http://{host}:{port}{server.object_path}"

    @property
    def url(self) -> str:
        return self._url

    @property
    def bytes_fetched(self) -> int:
        """Bytes read from the byte-range reader so far, each block counted once."""
        return self._server.cache.bytes_fetched

    def raise_for_reader_failure(self) -> None:
        """Re-raise the reader's first failure, if any, as the original exception."""
        failure = self._server.cache.first_failure
        if failure is not None:
            raise failure

    def __repr__(self) -> str:
        return "LoopbackVideoSource(<loopback>)"


@contextmanager
def serve_byte_ranges(
    reader: ByteRangeReader,
    *,
    block_bytes: int = 256 * 1024,
    maximum_cached_bytes: int = 64 * 1024 * 1024,
) -> Iterator[LoopbackVideoSource]:
    """Serve ``reader`` to FFmpeg on loopback until the ``with`` block exits.

    ``block_bytes`` is the upstream read size; ``maximum_cached_bytes`` bounds
    the block cache, evicting the least recently used block first.
    """
    require_positive_int(block_bytes, "block_bytes")
    require_positive_int(maximum_cached_bytes, "maximum_cached_bytes")
    require_positive_int(reader.size_bytes, "reader.size_bytes")
    server = _LoopbackServer(
        _BlockCache(reader, block_bytes=block_bytes, maximum_cached_bytes=maximum_cached_bytes)
    )
    serving_thread = threading.Thread(
        target=server.serve_forever, name="hflow-byte-range-source", daemon=True
    )
    serving_thread.start()
    try:
        yield LoopbackVideoSource(server)
    finally:
        server.shutdown()
        server.server_close()
        serving_thread.join()


@dataclass(frozen=True)
class MediaInput:
    """How one FFmpeg or ffprobe run opens a local file or a loopback source."""

    location: str
    protocols: str
    environment: Mapping[str, str] | None
    loopback_source: LoopbackVideoSource | None

    def raise_for_reader_failure(self) -> None:
        if self.loopback_source is not None:
            self.loopback_source.raise_for_reader_failure()


def media_input(source: Path | LoopbackVideoSource, *, local_protocols: str) -> MediaInput:
    """Resolve a local path strictly, or open a loopback source over HTTP only."""
    if isinstance(source, LoopbackVideoSource):
        return MediaInput(
            source.url,
            "http,tcp",
            {
                name: value
                for name, value in os.environ.items()
                if name not in _PROXY_ENVIRONMENT_VARIABLES
            },
            source,
        )
    if not isinstance(source, Path):
        raise TypeError("a media source must be a Path or a LoopbackVideoSource")
    return MediaInput(str(source.resolve(strict=True)), local_protocols, None, None)


__all__ = ["ByteRangeReader", "LoopbackVideoSource", "serve_byte_ranges"]
