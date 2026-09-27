"""Pinned transfers and range reads return verified bytes of exactly one revision."""

import hashlib
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from hflow.byte_range_source import serve_byte_ranges
from hflow.ffmpeg import ffmpeg_path
from hflow.media import probe_video
from hflow.sources import (
    DownloadLimits,
    PinnedSourceRangeReader,
    SourceExpectation,
    SourceObjectGetter,
    SourceReadError,
    SourceRevision,
    download_source,
)


@dataclass
class SourceResponse:
    meta: Mapping[str, object]
    chunks: tuple[bytes, ...]

    def stream(self, min_chunk_size: int = 8 * 1024 * 1024) -> Iterable[bytes]:
        yield from self.chunks


def test_versioned_transfer_publishes_identity_and_preserves_source(tmp_path: Path) -> None:
    source_versions = {"reviewed": b"reviewed source", "latest": b"changed source"}

    def get_source(
        _store: object, object_name: str, *, options: Mapping[str, object] | None = None
    ) -> SourceResponse:
        assert options is not None
        version = str(options["version"])
        payload = source_versions[version]
        return SourceResponse(
            {"path": object_name, "version": version, "size": len(payload)},
            (payload[:3], payload[3:]),
        )

    expectation = SourceExpectation(SourceRevision("media/source", "reviewed"))
    destination = tmp_path / "download"
    destination.write_bytes(b"previous")
    downloaded = download_source(
        object(), expectation, destination, limits=DownloadLimits(100), object_getter=get_source
    )
    assert destination.read_bytes() == source_versions["reviewed"]
    assert downloaded.sha256 == hashlib.sha256(source_versions["reviewed"]).hexdigest()
    assert downloaded.size_bytes == len(source_versions["reviewed"])
    assert sorted(path.name for path in tmp_path.iterdir()) == ["download"]


@pytest.mark.parametrize(
    "mismatch", ["revision", "path", "size", "digest", "short", "oversize", "request"]
)
def test_rejected_transfer_never_replaces_existing_bytes(tmp_path: Path, mismatch: str) -> None:
    payload = b"source"
    metadata: dict[str, object] = {
        "path": "media/source",
        "version": "reviewed",
        "size": len(payload),
    }
    if mismatch == "revision":
        metadata["version"] = "latest"
    if mismatch == "path":
        metadata["path"] = "other"
    if mismatch == "size":
        metadata["size"] = len(payload) + 1
    chunks = (payload[:-1],) if mismatch == "short" else (payload,)
    if mismatch == "oversize":
        chunks = (payload, b"extra")

    def get_source(
        _store: object, _name: str, *, options: Mapping[str, object] | None = None
    ) -> SourceResponse:
        if mismatch == "request":
            raise RuntimeError("credential diagnostic canary")
        return SourceResponse(metadata, chunks)

    expectation = SourceExpectation(
        SourceRevision("media/source", "reviewed"),
        size_bytes=len(payload),
        sha256="0" * 64 if mismatch == "digest" else hashlib.sha256(payload).hexdigest(),
    )
    destination = tmp_path / "download"
    destination.write_bytes(b"previous")
    with pytest.raises(SourceReadError) as captured:
        download_source(
            object(), expectation, destination, limits=DownloadLimits(10), object_getter=get_source
        )
    assert "canary" not in str(captured.value)
    assert destination.read_bytes() == b"previous"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["download"]


def _range_getter(
    payload: bytes, metadata: Mapping[str, object], *, trim: int = 0, extra: bytes = b""
) -> tuple[SourceObjectGetter, list[Mapping[str, object]]]:
    requests: list[Mapping[str, object]] = []

    def get_source(
        _store: object, _name: str, *, options: Mapping[str, object] | None = None
    ) -> SourceResponse:
        assert options is not None
        requests.append(options)
        start, stop = cast("tuple[int, int]", options["range"])
        return SourceResponse(metadata, (payload[start : stop - trim], extra))

    return get_source, requests


def test_range_reads_request_the_pinned_revision_and_exact_bytes() -> None:
    payload = bytes(range(200))
    get_source, requests = _range_getter(
        payload, {"path": "media/source", "version": "reviewed", "size": len(payload)}
    )
    reader = PinnedSourceRangeReader(
        object(),
        SourceExpectation(SourceRevision("media/source", "reviewed"), size_bytes=len(payload)),
        object_getter=get_source,
    )
    assert reader.size_bytes == len(payload)
    assert reader.read_range(10, 74) == payload[10:74]
    assert requests == [{"version": "reviewed", "range": (10, 74)}]


@pytest.mark.parametrize("mismatch", ["revision", "path", "size", "short", "long", "request"])
def test_a_range_from_a_different_object_or_a_partial_body_is_refused(mismatch: str) -> None:
    payload = bytes(range(200))
    metadata: dict[str, object] = {"path": "media/source", "version": "reviewed", "size": 200}
    if mismatch == "revision":
        metadata["version"] = "latest"
    if mismatch == "path":
        metadata["path"] = "other"
    if mismatch == "size":
        metadata["size"] = 201
    get_source, _requests = _range_getter(
        payload,
        metadata,
        trim=1 if mismatch == "short" else 0,
        extra=b"x" if mismatch == "long" else b"",
    )
    if mismatch == "request":

        def get_source(
            _store: object, _name: str, *, options: Mapping[str, object] | None = None
        ) -> SourceResponse:
            raise RuntimeError("credential diagnostic canary")

    reader = PinnedSourceRangeReader(
        object(),
        SourceExpectation(SourceRevision("media/source", "reviewed"), size_bytes=200),
        object_getter=get_source,
    )
    with pytest.raises(SourceReadError) as captured:
        reader.read_range(0, 100)
    assert "canary" not in str(captured.value)


def test_a_range_reader_requires_the_pinned_size_and_refuses_a_digest_it_cannot_check() -> None:
    revision = SourceRevision("media/source", "v")
    with pytest.raises(ValueError, match="pinned, positive source size"):
        PinnedSourceRangeReader(object(), SourceExpectation(revision))
    with pytest.raises(ValueError, match="cannot verify an expected SHA-256"):
        PinnedSourceRangeReader(
            object(), SourceExpectation(revision, size_bytes=10, sha256="0" * 64)
        )


def test_a_replaced_object_fails_probing_as_a_read_error_through_loopback(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.mp4"
    subprocess.run(
        [
            str(ffmpeg_path()), "-v", "error", "-f", "lavfi",
            "-i", "testsrc2=size=96x64:rate=10:duration=2", "-c:v", "libx264", str(source_path),
        ],
        check=True, capture_output=True, timeout=60,
    )  # fmt: skip
    payload = source_path.read_bytes()
    get_source, _requests = _range_getter(
        payload, {"path": "media/source", "version": "latest", "size": len(payload)}
    )
    reader = PinnedSourceRangeReader(
        object(),
        SourceExpectation(SourceRevision("media/source", "reviewed"), size_bytes=len(payload)),
        object_getter=get_source,
    )
    with serve_byte_ranges(reader) as source, pytest.raises(SourceReadError):
        probe_video(source)
