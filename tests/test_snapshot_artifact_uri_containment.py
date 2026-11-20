"""Copy-mode snapshot export must not read artifact URIs outside the workspace.

The workspace root is the parent of the catalog directory (the data root in the
standard layout). Artifact files the pipeline publishes live under that parent,
not inside the catalog directory, so copy mode still materializes those files.
A URI whose resolved file is outside that root, or an object-store URL when the
catalog itself is local, is attacker-controlled catalog data: copy mode must
refuse it before those bytes are written anywhere the export creates.
"""

from __future__ import annotations

import os
import shutil
import threading
from collections.abc import Generator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import duckdb
import pytest
from catalog_test_helpers import SNAPSHOT_EPISODE_STAMPS

import hflow
import hflow.snapshot as snapshot_module
import hflow.storage as storage_module
from hflow.catalog import Catalog, CheckRunRow
from hflow.cli import main as cli_main

OUTSIDE_MARKER = b"outside-workspace-artifact-bytes"
INSIDE_MARKER = b"inside-workspace-artifact-bytes"


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "data"
    workspace.mkdir()
    return workspace


def _append_artifact(catalog: Catalog, workspace: Path, artifact_uri: str) -> None:
    canonical_episode = workspace / "episode.canonical.mcap"
    canonical_episode.write_bytes(b"canonical bytes for artifact uri containment")
    catalog.append_episode(
        canonical_path=canonical_episode,
        stamps=SNAPSHOT_EPISODE_STAMPS,
        episode_metadata={"task": "containment"},
        check_rows=[
            CheckRunRow(
                check_name="media/contact_sheet",
                check_version="media-v1",
                critical=False,
                status=hflow.CheckStatus.MEASURED,
                duration_s=0.1,
                measurements={"artifact/preview.jpg": artifact_uri},
            )
        ],
    )


def _tree_contains(root: Path, marker: bytes) -> bool:
    if not root.exists():
        return False
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink() and marker in path.read_bytes():
            return True
    return False


def _record_marker_writes(monkeypatch: pytest.MonkeyPatch, marker: bytes) -> list[str]:
    """Remember every copy or bucket download that persisted ``marker``."""
    hits: list[str] = []
    real_copyfile = shutil.copyfile

    def copyfile(
        source: Path | str, destination: Path | str, *, follow_symlinks: bool = True
    ) -> Path | str:
        result = real_copyfile(source, destination, follow_symlinks=follow_symlinks)
        written = Path(destination)
        if written.is_file() and marker in written.read_bytes():
            hits.append(str(written))
        return result

    # snapshot and storage both bind the stdlib shutil module.
    monkeypatch.setattr(snapshot_module.shutil, "copyfile", copyfile)
    monkeypatch.setattr(storage_module.shutil, "copyfile", copyfile)
    real_download = storage_module._download_to_file_atomically

    def download(get_result: object, destination: Path) -> None:
        real_download(get_result, destination)
        if destination.is_file() and marker in destination.read_bytes():
            hits.append(str(destination))

    monkeypatch.setattr(storage_module, "_download_to_file_atomically", download)
    return hits


def _assert_marker_absent(*roots: Path, hits: list[str]) -> None:
    assert hits == []
    for root in roots:
        assert not _tree_contains(root, OUTSIDE_MARKER)


def test_copy_mode_still_materializes_artifact_inside_workspace(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    preview = workspace / "episodes" / "preview.jpg"
    preview.parent.mkdir()
    preview.write_bytes(INSIDE_MARKER)
    catalog = Catalog(workspace / "catalog")
    _append_artifact(catalog, workspace, str(preview.resolve()))
    output = tmp_path / "snapshot"

    report = hflow.export_dataset_snapshot(
        catalog.root, output, media_mode=hflow.SnapshotMediaMode.COPY
    )

    assert report.copied_media_count == 1
    assert _tree_contains(output, INSIDE_MARKER)
    assert not _tree_contains(output, OUTSIDE_MARKER)


def test_copy_mode_refuses_artifact_uri_outside_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    secret = tmp_path / "outside.bin"
    secret.write_bytes(OUTSIDE_MARKER)
    catalog = Catalog(workspace / "catalog")
    _append_artifact(catalog, workspace, str(secret.resolve()))
    output = tmp_path / "snapshot"
    hits = _record_marker_writes(monkeypatch, OUTSIDE_MARKER)

    with pytest.raises(ValueError, match="outside the workspace"):
        hflow.export_dataset_snapshot(catalog.root, output, media_mode=hflow.SnapshotMediaMode.COPY)

    _assert_marker_absent(output, hits=hits)


def test_cli_copy_mode_refuses_artifact_uri_outside_workspace(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = _workspace(tmp_path)
    secret = tmp_path / "outside.bin"
    secret.write_bytes(OUTSIDE_MARKER)
    catalog = Catalog(workspace / "catalog")
    _append_artifact(catalog, workspace, str(secret.resolve()))
    output = tmp_path / "snapshot"

    exit_code = cli_main(
        [
            "export",
            "snapshot",
            "--catalog",
            str(catalog.root),
            "--output",
            str(output),
            "--media",
            "copy",
        ]
    )

    assert exit_code == 2
    assert "outside the workspace" in capsys.readouterr().err
    assert not _tree_contains(output, OUTSIDE_MARKER)


def test_copy_mode_refuses_symlink_that_resolves_outside_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    secret = tmp_path / "outside.bin"
    secret.write_bytes(OUTSIDE_MARKER)
    link = workspace / "episodes" / "preview.jpg"
    link.parent.mkdir()
    link.symlink_to(secret)
    catalog = Catalog(workspace / "catalog")
    _append_artifact(catalog, workspace, str(link))
    output = tmp_path / "snapshot"
    hits = _record_marker_writes(monkeypatch, OUTSIDE_MARKER)

    with pytest.raises(ValueError, match="outside the workspace"):
        hflow.export_dataset_snapshot(catalog.root, output, media_mode=hflow.SnapshotMediaMode.COPY)

    _assert_marker_absent(output, hits=hits)


def test_copy_mode_refuses_parent_segments_that_escape_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    secret = tmp_path / "outside.bin"
    secret.write_bytes(OUTSIDE_MARKER)
    (workspace / "episodes").mkdir()
    catalog = Catalog(workspace / "catalog")
    escaped = workspace / "episodes" / ".." / ".." / "outside.bin"
    _append_artifact(catalog, workspace, str(escaped))
    output = tmp_path / "snapshot"
    hits = _record_marker_writes(monkeypatch, OUTSIDE_MARKER)

    with pytest.raises(ValueError, match="outside the workspace"):
        hflow.export_dataset_snapshot(catalog.root, output, media_mode=hflow.SnapshotMediaMode.COPY)

    _assert_marker_absent(output, hits=hits)


def test_copy_mode_refuses_relative_uri_that_escapes_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    secret = tmp_path / "outside.bin"
    secret.write_bytes(OUTSIDE_MARKER)
    catalog = Catalog(workspace / "catalog")
    _append_artifact(catalog, workspace, "../outside.bin")
    monkeypatch.chdir(workspace)
    output = tmp_path / "snapshot"
    hits = _record_marker_writes(monkeypatch, OUTSIDE_MARKER)

    with pytest.raises(ValueError, match="outside the workspace"):
        hflow.export_dataset_snapshot(catalog.root, output, media_mode=hflow.SnapshotMediaMode.COPY)

    _assert_marker_absent(output, hits=hits)


def test_copy_mode_does_not_read_file_replaced_after_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A containment check that opens the path again must not observe a new file.

    The catalog names a regular file inside the workspace. After the first
    status lookup of that path, the file is replaced with a symlink to a file
    outside the workspace. Copy mode must keep the bytes it already opened, or
    refuse, and must not write the outside file.
    """
    workspace = _workspace(tmp_path)
    secret = tmp_path / "outside.bin"
    secret.write_bytes(OUTSIDE_MARKER)
    preview = workspace / "episodes" / "preview.jpg"
    preview.parent.mkdir()
    preview.write_bytes(INSIDE_MARKER)
    catalog = Catalog(workspace / "catalog")
    _append_artifact(catalog, workspace, str(preview))
    output = tmp_path / "snapshot"
    hits = _record_marker_writes(monkeypatch, OUTSIDE_MARKER)
    swapped = False
    real_stat = os.stat
    preview_path = os.fspath(preview)

    def stat_and_swap(
        path: int | os.PathLike[str] | str | bytes,
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        nonlocal swapped
        result = real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)
        if not swapped and not isinstance(path, int) and os.fspath(path) == preview_path:
            swapped = True
            preview.unlink()
            preview.symlink_to(secret)
        return result

    monkeypatch.setattr(os, "stat", stat_and_swap)

    try:
        hflow.export_dataset_snapshot(catalog.root, output, media_mode=hflow.SnapshotMediaMode.COPY)
    except ValueError as error:
        assert "outside the workspace" in str(error)
    else:
        assert not _tree_contains(output, OUTSIDE_MARKER)
        assert _tree_contains(output, INSIDE_MARKER)

    assert swapped
    _assert_marker_absent(output, hits=hits)


class _ObjectStoreHandler(BaseHTTPRequestHandler):
    payload = OUTSIDE_MARKER

    def do_HEAD(self) -> None:
        self._respond(include_body=False)

    def do_GET(self) -> None:
        self._respond(include_body=True)

    def _respond(self, *, include_body: bool) -> None:
        body = self.payload
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("ETag", '"outside"')
        self.end_headers()
        if include_body:
            self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


@contextmanager
def _serve_object() -> Generator[str, None, None]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ObjectStoreHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_copy_mode_does_not_fetch_object_store_url_outside_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An s3 URL in a local catalog is fetched with ambient client configuration.

    ``fetch_uri`` sends every ``s3``/``gs``/``az`` URL through the same bucket
    branch. This uses ``s3`` because that client reads ``AWS_ENDPOINT`` from
    the environment; the endpoint is a local stand-in for the object store.
    """
    workspace = _workspace(tmp_path)
    catalog = Catalog(workspace / "catalog")
    _append_artifact(catalog, workspace, "s3://other-bucket/outside/secret.bin")
    output = tmp_path / "snapshot"
    hits = _record_marker_writes(monkeypatch, OUTSIDE_MARKER)
    mirror = tmp_path / "mirrors"

    with _serve_object() as endpoint:
        monkeypatch.setenv("HFLOW_MIRROR_DIR", str(mirror))
        monkeypatch.setenv("AWS_ENDPOINT", endpoint)
        monkeypatch.setenv("AWS_ALLOW_HTTP", "true")
        monkeypatch.setenv("AWS_SKIP_SIGNATURE", "true")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
        with pytest.raises(ValueError, match="outside the workspace"):
            hflow.export_dataset_snapshot(
                catalog.root, output, media_mode=hflow.SnapshotMediaMode.COPY
            )

    _assert_marker_absent(output, mirror, hits=hits)


def test_references_mode_does_not_read_artifact_uris(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    secret = tmp_path / "outside.bin"
    secret.write_bytes(OUTSIDE_MARKER)
    catalog = Catalog(workspace / "catalog")
    outside_uri = str(secret.resolve())
    _append_artifact(catalog, workspace, outside_uri)
    output = tmp_path / "snapshot"

    report = hflow.export_dataset_snapshot(
        catalog.root, output, media_mode=hflow.SnapshotMediaMode.REFERENCES
    )

    assert report.copied_media_count == 0
    assert not _tree_contains(output, OUTSIDE_MARKER)
    media_uri = output / "media.parquet"
    assert duckdb.execute("SELECT uri FROM read_parquet(?)", [str(media_uri)]).fetchone() == (
        outside_uri,
    )
