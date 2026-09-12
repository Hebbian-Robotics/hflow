"""Pinned transfers publish verified bytes or preserve the previous destination."""

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest

from hflow.sources import (
    DownloadLimits,
    SourceExpectation,
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
    assert source_versions["latest"] == b"changed source"
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
