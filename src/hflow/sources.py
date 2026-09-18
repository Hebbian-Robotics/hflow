"""Read-only, revision-pinned source transfers into caller-owned local storage.

The caller supplies an authenticated obstore-compatible getter. This module
never creates a workspace mirror, discovers credentials, or writes to a bucket.
"""

import hashlib
import hmac
import math
import os
import re
import tempfile
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from obstore import GetOptions
    from obstore.store import ObjectStore

SOURCE_STREAM_CHUNK_BYTES = 8 * 1024 * 1024


class SourceReadError(RuntimeError):
    """A transfer failed; the public message excludes source and provider details."""


class SourceObjectResult(Protocol):
    @property
    def meta(self) -> Mapping[str, object]: ...

    def stream(self, min_chunk_size: int = SOURCE_STREAM_CHUNK_BYTES) -> Iterable[bytes]: ...


class SourceObjectGetter(Protocol):
    def __call__(
        self,
        object_store: object,
        object_name: str,
        /,
        *,
        options: Mapping[str, object] | None = None,
    ) -> SourceObjectResult: ...


@dataclass(frozen=True)
class SourceRevision:
    """An object key and immutable provider revision, scoped by the supplied store."""

    object_name: str = field(repr=False)
    version: str = field(repr=False)

    def __post_init__(self) -> None:
        for value in (self.object_name, self.version):
            if (
                not isinstance(value, str)
                or not value
                or value != value.strip()
                or any(ord(character) < 32 or ord(character) == 127 for character in value)
            ):
                raise ValueError("source revision must contain nonempty opaque coordinates")
            value.encode("utf-8")
        if self.version == "null":
            raise ValueError("source revision must be immutable")


@dataclass(frozen=True)
class DownloadLimits:
    maximum_bytes: int
    timeout_seconds: float = 600.0

    def __post_init__(self) -> None:
        if type(self.maximum_bytes) is not int or self.maximum_bytes <= 0:
            raise ValueError("source byte limit must be a positive integer")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, int | float)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("source timeout must be finite and positive")


@dataclass(frozen=True)
class SourceExpectation:
    """Facts known before transfer; discovery may not yet know the content hash."""

    revision: SourceRevision
    size_bytes: int | None = None
    sha256: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.size_bytes is not None and (
            type(self.size_bytes) is not int or self.size_bytes < 0
        ):
            raise ValueError("source size must be a nonnegative integer")
        if self.sha256 is not None and (
            not isinstance(self.sha256, str) or re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None
        ):
            raise ValueError("source digest must be a lowercase SHA-256")


@dataclass(frozen=True)
class DownloadedSource:
    """Verified transfer metadata; the caller owns the file's subsequent lifetime."""

    path: Path = field(repr=False)
    revision: SourceRevision
    size_bytes: int
    sha256: str = field(repr=False)


def download_source(
    object_store: object,
    expectation: SourceExpectation,
    destination: Path,
    *,
    limits: DownloadLimits,
    object_getter: SourceObjectGetter | None = None,
) -> DownloadedSource:
    """Verify revision, optional identity facts, and actual bytes before replacement.

    Existing destinations survive failed transfers. A successful transfer
    atomically replaces the destination. The elapsed budget is checked between
    chunks; callers must also configure their provider's request/read timeout.
    """
    getter = object_getter or get_source_object
    deadline = time.monotonic() + limits.timeout_seconds
    try:
        result = getter(
            object_store,
            expectation.revision.object_name,
            options={"version": expectation.revision.version},
        )
        metadata = result.meta
        size_bytes = metadata.get("size")
        if (
            type(size_bytes) is not int
            or size_bytes < 0
            or size_bytes > limits.maximum_bytes
            or metadata.get("version") != expectation.revision.version
            or ("path" in metadata and metadata["path"] != expectation.revision.object_name)
            or (expectation.size_bytes is not None and size_bytes != expectation.size_bytes)
        ):
            raise SourceReadError("source metadata does not match its pinned revision")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=destination.parent, prefix=".source-transfer-"
        ) as directory:
            staged = Path(directory) / "source"
            digest = hashlib.sha256()
            downloaded_bytes = 0
            with staged.open("wb") as output:
                for chunk in result.stream(SOURCE_STREAM_CHUNK_BYTES):
                    if time.monotonic() > deadline:
                        raise SourceReadError("source transfer exceeded its deadline")
                    if not isinstance(chunk, bytes):
                        raise SourceReadError("source returned invalid bytes")
                    downloaded_bytes += len(chunk)
                    if downloaded_bytes > size_bytes:
                        raise SourceReadError("source exceeded its declared size")
                    output.write(chunk)
                    digest.update(chunk)
                if downloaded_bytes != size_bytes:
                    raise SourceReadError("source transfer was incomplete")
                source_sha256 = digest.hexdigest()
                if expectation.sha256 is not None and not hmac.compare_digest(
                    source_sha256, expectation.sha256
                ):
                    raise SourceReadError("source digest did not match")
                output.flush()
                os.fsync(output.fileno())
            if time.monotonic() > deadline:
                raise SourceReadError("source transfer exceeded its deadline")
            staged.replace(destination)
        return DownloadedSource(destination, expectation.revision, downloaded_bytes, source_sha256)
    except SourceReadError:
        raise
    except Exception as error:
        raise SourceReadError("source transfer failed") from error


def get_source_object(
    object_store: object,
    object_name: str,
    *,
    options: Mapping[str, object] | None = None,
) -> SourceObjectResult:
    """Optional obstore boundary; uses only the caller's configured store."""
    from obstore import get

    return cast(
        SourceObjectResult,
        get(cast("ObjectStore", object_store), object_name, options=cast("GetOptions", options)),
    )
