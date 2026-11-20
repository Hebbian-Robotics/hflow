"""Ingest must not follow a data-root symlink out of the workspace.

These tests are the spec for a containment bug. They fail while a data-root
relative URI can name a recording outside the real data root, and they pass
once opening and republishing that recording is refused. Refusal has to happen
before the outside file is opened and before any workspace file is created
from its bytes.

Two layouts share one missing check. A directory symlink plus parent segments
(``episodes-in/../neighbor.mcap``) stays inside the data root under lexical
normalization, including the batch planner's size lookup, and then the
original spelling is joined to the data root and resolved. A file symlink
with no parent segments (``episodes-in/alias.mcap``) is lexically inside the
root and resolves outside when the symlink is followed. Repeating the lexical
check, or sizing the normalized key and then opening the original spelling,
still fails these tests.

Object-store data roots are not covered: episode references there are URL
joins, with no local symlink walk. ``App.test`` and ``App.process_many`` are
not called separately; both read through ``App.process`` / ``source_identity``.
Stages other than sync are not given their own cases: with sync omitted they
do not fetch the raw source.
"""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import ctypes.util
import os
import select
from collections.abc import Callable
from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient
from hflow_server import ServerSettings, create_app

import hflow
from hflow.cli import main as cli_main
from hflow.curation import open_catalog_connection
from hflow.runtime import AirflowClient, AirflowDagRun, RuntimeConfig, render_bundle
from hflow.stage_execution import plan_stage_batches, process_stage_batch, resolve_episode_reference
from hflow.stage_planning import outstanding_stage_uris
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode
from hflow.uri import parse_data_root_relative_uri

PARENT_SEGMENT_URI = "episodes-in/../neighbor.mcap"
FILE_SYMLINK_URI = "episodes-in/alias.mcap"
OUTSIDE_MARKER = "outside-recording-marker"
DECOY_MARKER = "inside-decoy-marker"

# Linux inotify IN_OPEN. A subscription of only this bit means every queued
# event is an open of the watched inode; stat does not produce one.
_IN_OPEN = 0x00000020

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_libc.inotify_init1.argtypes = [ctypes.c_int]
_libc.inotify_init1.restype = ctypes.c_int
_libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
_libc.inotify_add_watch.restype = ctypes.c_int


class _OpenWatcher:
    """Records whether one file inode is opened while the watcher is active."""

    def __init__(self, path: Path) -> None:
        self._fd = _libc.inotify_init1(os.O_NONBLOCK)
        if self._fd < 0:
            raise OSError(ctypes.get_errno(), "inotify_init1")
        watch = _libc.inotify_add_watch(self._fd, os.fsencode(path), _IN_OPEN)
        if watch < 0:
            os.close(self._fd)
            raise OSError(ctypes.get_errno(), "inotify_add_watch")

    def __enter__(self) -> _OpenWatcher:
        return self

    def __exit__(self, *exc_info: object) -> None:
        os.close(self._fd)

    def opened(self) -> bool:
        saw_open = False
        while True:
            ready, _, _ = select.select([self._fd], [], [], 0)
            if not ready:
                return saw_open
            os.read(self._fd, 4096)
            saw_open = True


def _synthesize(path: Path, operator: str, *, seed: int) -> None:
    synthesize_episode(
        path,
        SyntheticEpisodeSpec(duration_s=1.0, cameras=(), operator=operator, seed=seed),
    )
    path.chmod(0o600)


def _parent_segment_escape(data_root: Path, outside: Path) -> Path:
    """``episodes-in`` is a directory symlink; the URI climbs out of its target.

    ``neighbor.mcap`` also exists at the lexically normalized key inside the
    root, so a size check or a second lexical normalization has a different
    file to look at than the one filesystem resolution of the original URI
    opens.
    """
    data_root.mkdir(parents=True, exist_ok=True)
    disk = outside / "disk"
    disk.mkdir(parents=True)
    secret = outside / "neighbor.mcap"
    _synthesize(secret, OUTSIDE_MARKER, seed=0)
    (data_root / "episodes-in").symlink_to(disk, target_is_directory=True)
    _synthesize(data_root / "neighbor.mcap", DECOY_MARKER, seed=1)
    resolved = (data_root / "episodes-in" / ".." / "neighbor.mcap").resolve()
    assert resolved == secret.resolve()
    assert not resolved.is_relative_to(data_root.resolve())
    assert (data_root / "neighbor.mcap").resolve().is_relative_to(data_root.resolve())
    return secret


def _file_symlink_escape(data_root: Path, outside: Path) -> Path:
    """A lexically in-root URI whose last component is a symlink to the outside."""
    data_root.mkdir(parents=True, exist_ok=True)
    episodes_in = data_root / "episodes-in"
    episodes_in.mkdir()
    secret = outside / "alias-target.mcap"
    _synthesize(secret, OUTSIDE_MARKER, seed=2)
    (episodes_in / "alias.mcap").symlink_to(secret)
    resolved = (data_root / FILE_SYMLINK_URI).resolve()
    assert resolved == secret.resolve()
    assert not resolved.is_relative_to(data_root.resolve())
    return secret


def _attempt(operation: Callable[[], object]) -> object:
    try:
        return operation()
    except (OSError, RuntimeError, ValueError):
        return None


def _republished(outcome: object) -> bool:
    if outcome is None or isinstance(outcome, Exception):
        return False
    if isinstance(outcome, dict):
        return bool(outcome.get("processed") or outcome.get("quarantined"))
    return True


def _paths_containing(root: Path, needle: bytes) -> list[Path]:
    matches: list[Path] = []
    if not root.exists():
        return matches
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(directory)
        dirnames[:] = [name for name in dirnames if not (current / name).is_symlink()]
        for name in filenames:
            path = current / name
            if path.is_symlink():
                continue
            if needle in path.read_bytes():
                matches.append(path)
    return matches


def _identity_is_contained(data_root: Path, uri: str) -> bool:
    try:
        parsed = parse_data_root_relative_uri(uri)
    except ValueError:
        return True
    app = hflow.App("identity", data_root=data_root, default_checks=())
    try:
        identity = app.source_identity(resolve_episode_reference(str(data_root), parsed))
    except (OSError, ValueError):
        return True
    candidate = Path(identity)
    real = candidate.resolve() if candidate.is_absolute() else (data_root / candidate).resolve()
    return real.is_relative_to(data_root.resolve())


def _served_canonical_payloads(data_root: Path) -> list[bytes]:
    catalog = data_root / "catalog"
    if not catalog.exists():
        return []
    try:
        connection = open_catalog_connection(catalog)
    except (OSError, duckdb.Error):
        return []
    try:
        try:
            rows = connection.execute("SELECT episode_id FROM episodes").fetchall()
        except duckdb.Error:
            return []
    finally:
        connection.close()
    payloads: list[bytes] = []
    with TestClient(
        create_app(ServerSettings(data_root=str(data_root), open_browser=False))
    ) as client:
        for (episode_id,) in rows:
            response = client.get(f"/api/v1/episodes/{episode_id}/canonical")
            if response.status_code == 200:
                payloads.append(response.content)
    return payloads


def _assert_contained(data_root: Path, secret: Path, watcher: _OpenWatcher) -> None:
    marker = OUTSIDE_MARKER.encode()
    leaked = _paths_containing(data_root, marker)
    served = [payload for payload in _served_canonical_payloads(data_root) if marker in payload]
    assert watcher.opened() is False
    assert leaked == []
    assert served == []
    assert secret.is_file()


def _run_worker(data_root: Path, uri: str, *, modes: tuple[str, ...]) -> list[object]:
    app = hflow.App("worker", data_root=data_root, default_checks=())
    outcomes: list[object] = []
    for mode in modes:
        try:
            batches = plan_stage_batches(
                [uri], mode=mode, batch_count=None, data_root=str(data_root)
            )
        except ValueError:
            continue
        for batch in batches:
            items = list(batch["items"])
            outcomes.append(
                _attempt(lambda items=items: asyncio.run(process_stage_batch(app, items, "sync")))
            )
    outcomes.append(
        _attempt(lambda: asyncio.run(app.process(uri, record=False, stages={hflow.Stage.SYNC})))
    )
    with contextlib.suppress(OSError, RuntimeError, ValueError):
        outstanding_stage_uris(app, [uri], hflow.Stage.META, data_root=str(data_root))
    return outcomes


def test_parent_segment_symlink_is_not_opened_or_republished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    secret = _parent_segment_escape(data_root, tmp_path / "outside")
    (tmp_path / "cwd").mkdir()
    monkeypatch.chdir(tmp_path / "cwd")
    with _OpenWatcher(secret) as watcher:
        contained_identity = _identity_is_contained(data_root, PARENT_SEGMENT_URI)
        outcomes = _run_worker(data_root, PARENT_SEGMENT_URI, modes=("batch", "online"))
        _assert_contained(data_root, secret, watcher)
    assert contained_identity
    assert not any(_republished(outcome) for outcome in outcomes)


def test_file_symlink_uri_is_not_opened_or_republished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    secret = _file_symlink_escape(data_root, tmp_path / "outside")
    (tmp_path / "cwd").mkdir()
    monkeypatch.chdir(tmp_path / "cwd")
    with _OpenWatcher(secret) as watcher:
        contained_identity = _identity_is_contained(data_root, FILE_SYMLINK_URI)
        outcomes = _run_worker(data_root, FILE_SYMLINK_URI, modes=("batch", "online"))
        _assert_contained(data_root, secret, watcher)
    assert contained_identity
    assert not any(_republished(outcome) for outcome in outcomes)


def _cli_project(
    tmp_path: Path, secret_builder: Callable[[Path, Path], Path]
) -> tuple[Path, Path, Path]:
    project = tmp_path / "project"
    data_root = project / "data"
    secret = secret_builder(data_root, tmp_path / "outside")
    (project / "pipeline.py").write_text(
        "import hflow\napp = hflow.App('in-process', default_checks=())\n"
    )
    (project / "hflow.toml").write_text('data_root = "./data"\n')
    return project, data_root, secret


@pytest.mark.parametrize(
    ("uri", "builder"),
    [
        (PARENT_SEGMENT_URI, _parent_segment_escape),
        (FILE_SYMLINK_URI, _file_symlink_escape),
    ],
    ids=["parent-segments", "file-symlink"],
)
def test_in_process_ingest_cli_does_not_republish_outside_recordings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    uri: str,
    builder: Callable[[Path, Path], Path],
) -> None:
    project, data_root, secret = _cli_project(tmp_path, builder)
    monkeypatch.chdir(project)
    for variable in ("HFLOW_DATA_ROOT", "HFLOW_AIRFLOW_URL", "HFLOW_AIRFLOW_DAG_ID"):
        monkeypatch.delenv(variable, raising=False)
    with _OpenWatcher(secret) as watcher:
        exit_code = cli_main(["ingest", uri])
        _assert_contained(data_root, secret, watcher)
    assert exit_code != 0


def _render_project_bundle(data_root: Path, project: Path) -> None:
    pipeline = project / "pipeline.py"
    pipeline.write_text("import hflow\napp = hflow.App('in-process', default_checks=())\n")
    (project / "hflow.toml").write_text('data_root = "./data"\n')
    render_bundle(RuntimeConfig(pipeline_file=pipeline, data_root=data_root), data_root / "runtime")


@pytest.mark.parametrize(
    ("uri", "builder", "mode"),
    [
        (PARENT_SEGMENT_URI, _parent_segment_escape, "online"),
        (FILE_SYMLINK_URI, _file_symlink_escape, "batch"),
    ],
    ids=["parent-segments-online", "file-symlink-batch"],
)
def test_server_ingest_does_not_republish_a_symlink_resolved_uri(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    uri: str,
    builder: Callable[[Path, Path], Path],
    mode: str,
) -> None:
    """POST /api/v1/runtime/ingest delivers the URI to the sync task's library calls.

    Airflow is not started. ``AirflowClient.trigger_dag_run`` is replaced with
    the two calls the rendered sync task makes after it has an App:
    ``plan_stage_batches`` then ``process_stage_batch``. Everything before that
    — server parse, client parse, trigger conf — is the real code.
    """
    project = tmp_path / "project"
    data_root = project / "data"
    secret = builder(data_root, tmp_path / "outside")
    _render_project_bundle(data_root, project)
    for variable in ("HFLOW_AIRFLOW_URL", "HFLOW_AIRFLOW_DAG_ID", "HFLOW_AIRFLOW_TOKEN"):
        monkeypatch.delenv(variable, raising=False)
    app = hflow.App("worker", data_root=data_root, default_checks=())
    published = 0

    def run_sync_task(
        self: AirflowClient,
        dag_id: str,
        conf: dict[str, object] | None = None,
        *,
        dag_run_id: str | None = None,
    ) -> AirflowDagRun:
        nonlocal published
        del self, dag_id, dag_run_id
        payload = conf or {}
        uris = payload.get("uris")
        assert isinstance(uris, list)
        task_mode = payload.get("mode")
        assert isinstance(task_mode, str)
        batch_count = payload.get("batch_count")
        batches = plan_stage_batches(
            [str(item) for item in uris],
            mode=task_mode,
            batch_count=batch_count if isinstance(batch_count, int) else None,
            data_root=str(data_root),
        )
        for batch in batches:
            counts = asyncio.run(process_stage_batch(app, list(batch["items"]), "sync"))
            published += counts["processed"] + counts["quarantined"]
        return AirflowDagRun(
            dag_run_id="manual",
            state="queued",
            logical_date=None,
            start_date=None,
            end_date=None,
            conf={},
        )

    monkeypatch.setattr(AirflowClient, "trigger_dag_run", run_sync_task)
    with _OpenWatcher(secret) as watcher:
        with (
            TestClient(
                create_app(ServerSettings(data_root=str(data_root), open_browser=False))
            ) as client,
            contextlib.suppress(OSError, RuntimeError, ValueError),
        ):
            client.post(
                "/api/v1/runtime/ingest",
                json={"uris": [uri], "profile": "full", "mode": mode},
            )
        _assert_contained(data_root, secret, watcher)
    assert published == 0
