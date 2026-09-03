"""The local DuckDB catalog browser lifecycle."""

from dataclasses import replace
from enum import IntEnum
from pathlib import Path
from socket import AF_INET, SOCK_STREAM, socket
from threading import Event, Thread
from typing import TYPE_CHECKING

import duckdb
import pytest

import hflow
import hflow.catalog_ui as catalog_ui
from hflow.catalog import Catalog
from hflow.format import CATALOG_FORMAT_VERSION
from hflow.storage import StorageRoot
from hflow.transform import EpisodeStamps

if TYPE_CHECKING:
    from hflow.storage import BucketStorageRoot

_BUCKET_STAMPS = EpisodeStamps(
    schema_version="1",
    pipeline_version="bucket-ui-test",
    ffmpeg_version="ffmpeg version test",
    robot_software_version="sim-0.1.0",
)


def _write_remote_catalog_marker(catalog_root: "BucketStorageRoot") -> None:
    catalog_root.write_bytes_if_absent(
        "format_version",
        (CATALOG_FORMAT_VERSION + "\n").encode(),
    )


def _append_remote_episode(
    tmp_path: Path,
    catalog_root: "BucketStorageRoot",
    *,
    task: str,
    canonical_bytes: bytes = b"canonical episode",
) -> None:
    canonical_episode = tmp_path / f"{task}.canonical.mcap"
    canonical_episode.write_bytes(canonical_bytes)
    Catalog(catalog_root).append_episode(
        canonical_path=canonical_episode,
        stamps=_BUCKET_STAMPS,
        episode_metadata={"task": task},
        check_rows=[],
    )


def _writer_catalog_root(tmp_path: Path, remote_dir: Path) -> "BucketStorageRoot":
    """A catalog writer with its own mirror over the same bucket as the UI.

    Production ingest is a separate process with a separate mirror. Sharing the
    UI's mirror would let appends prime DuckDB without exercising sync.
    """
    from hflow.storage import BucketStorageRoot

    return BucketStorageRoot(
        f"file://{remote_dir}",
        mirror=tmp_path / "writer-mirror",
    ).child("catalog")


def test_catalog_ui_starts_empty_then_exposes_the_first_completed_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_root = tmp_path / "catalog"
    shutdown_event = Event()
    server_started_event = Event()
    empty_catalog_ready_event = Event()
    catalog_refreshed_event = Event()
    catalog_connections: list[duckdb.DuckDBPyConnection] = []
    background_failures: list[BaseException] = []

    def record_server_start(catalog_connection: duckdb.DuckDBPyConnection, port: int) -> None:
        assert port == catalog_ui.DEFAULT_CATALOG_UI_PORT
        catalog_connections.append(catalog_connection)
        server_started_event.set()

    refresh_local_catalog_connection = catalog_ui._refresh_local_catalog_connection
    catalog_connection_contains_episodes = catalog_ui._catalog_connection_contains_episodes

    def record_initial_catalog_read(
        catalog_connection: duckdb.DuckDBPyConnection,
    ) -> bool:
        contains_episodes = catalog_connection_contains_episodes(catalog_connection)
        empty_catalog_ready_event.set()
        return contains_episodes

    def record_catalog_refresh(
        catalog_connection: duckdb.DuckDBPyConnection,
        refreshed_catalog_root: Path,
    ) -> None:
        refresh_local_catalog_connection(catalog_connection, refreshed_catalog_root)
        catalog_refreshed_event.set()

    monkeypatch.setattr(catalog_ui, "_start_duckdb_ui_server", record_server_start)
    monkeypatch.setattr(
        catalog_ui,
        "_catalog_connection_contains_episodes",
        record_initial_catalog_read,
    )
    monkeypatch.setattr(
        catalog_ui,
        "_refresh_local_catalog_connection",
        record_catalog_refresh,
    )

    def run_catalog_ui() -> None:
        try:
            catalog_ui.serve_catalog_ui(
                catalog_ui.CatalogUiSettings(
                    catalog_root=catalog_root,
                    open_browser=False,
                    catalog_poll_interval_seconds=0.01,
                ),
                shutdown_event=shutdown_event,
            )
        except BaseException as error:
            background_failures.append(error)

    catalog_ui_thread = Thread(target=run_catalog_ui)
    catalog_ui_thread.start()
    try:
        assert server_started_event.wait(timeout=2)
        assert empty_catalog_ready_event.wait(timeout=2)
        (catalog_connection,) = catalog_connections
        assert catalog_connection.execute("SELECT count(*) FROM episodes").fetchone() == (0,)

        canonical_episode = tmp_path / "episode.canonical.mcap"
        canonical_episode.write_bytes(b"canonical episode")
        hflow.Catalog(catalog_root).append_episode(
            canonical_path=canonical_episode,
            stamps=hflow.EpisodeStamps(
                schema_version="1",
                pipeline_version="test-pipeline",
                ffmpeg_version="test-ffmpeg",
                robot_software_version="test-robot",
            ),
            episode_metadata={"task": "demo"},
            check_rows=[],
        )

        assert catalog_refreshed_event.wait(timeout=2)
        assert catalog_connection.execute(
            "SELECT count(*), min(task) FROM episodes"
        ).fetchone() == (1, "demo")
    finally:
        shutdown_event.set()
        catalog_ui_thread.join(timeout=2)

    assert not catalog_ui_thread.is_alive()
    assert background_failures == []


@pytest.mark.parametrize("port", [0, 65536])
def test_catalog_ui_refuses_an_invalid_port(tmp_path: Path, port: int) -> None:
    with pytest.raises(ValueError, match="port must be between 1 and 65535"):
        catalog_ui.CatalogUiSettings(catalog_root=tmp_path / "catalog", port=port)


@pytest.mark.parametrize(
    ("port", "type_name"),
    [(True, "bool"), (4213.0, "float"), ("4213", "str")],
)
def test_catalog_ui_refuses_a_port_of_the_wrong_type(
    tmp_path: Path, port: object, type_name: str
) -> None:
    settings = catalog_ui.CatalogUiSettings(catalog_root=tmp_path / "catalog")
    with pytest.raises(ValueError, match=f"port must be an int, got {type_name}"):
        replace(settings, port=port)


class _RecordingCatalogConnection:
    """DuckDB connection proxy that records every statement it executes.

    ``CALL stop_ui_server()`` is recorded but not delegated to the inner
    connection: on a connection without the loaded ui extension, executing it
    would make DuckDB auto-install the extension from the network -- the exact
    behavior #336 removes. The shutdown tests assert which statements the
    shutdown path issues, not what DuckDB does with them.
    """

    def __init__(self, inner: duckdb.DuckDBPyConnection) -> None:
        self._inner = inner
        self.executed_statements: list[str] = []
        self.closed = False

    def execute(self, sql: str, *parameters: object) -> object:
        self.executed_statements.append(sql)
        if sql == "CALL stop_ui_server()":
            return None
        return self._inner.execute(sql, *parameters)

    def close(self) -> None:
        self.closed = True
        self._inner.close()

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)


def _serve_until_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ui_extension_loaded: bool,
) -> _RecordingCatalogConnection:
    shutdown_event = Event()
    server_started_event = Event()
    connections: list[_RecordingCatalogConnection] = []
    background_failures: list[BaseException] = []

    real_open_catalog_connection = catalog_ui.open_catalog_connection

    def record_server_start(catalog_connection: duckdb.DuckDBPyConnection, port: int) -> None:
        assert port == catalog_ui.DEFAULT_CATALOG_UI_PORT
        server_started_event.set()

    def open_recorded_connection(catalog_root: Path) -> _RecordingCatalogConnection:
        connection = _RecordingCatalogConnection(real_open_catalog_connection(catalog_root))
        connections.append(connection)
        return connection

    monkeypatch.setattr(catalog_ui, "open_catalog_connection", open_recorded_connection)
    monkeypatch.setattr(catalog_ui, "_start_duckdb_ui_server", record_server_start)
    monkeypatch.setattr(
        catalog_ui,
        "_ui_extension_is_loaded",
        lambda _catalog_connection: ui_extension_loaded,
    )

    def run_catalog_ui() -> None:
        try:
            catalog_ui.serve_catalog_ui(
                catalog_ui.CatalogUiSettings(
                    catalog_root=tmp_path / "catalog",
                    open_browser=False,
                    catalog_poll_interval_seconds=0.01,
                ),
                shutdown_event=shutdown_event,
            )
        except BaseException as error:
            background_failures.append(error)

    catalog_ui_thread = Thread(target=run_catalog_ui)
    catalog_ui_thread.start()
    try:
        assert server_started_event.wait(timeout=2)
    finally:
        shutdown_event.set()
        catalog_ui_thread.join(timeout=2)

    assert not catalog_ui_thread.is_alive()
    assert background_failures == []
    (connection,) = connections
    return connection


def test_catalog_ui_shutdown_does_not_stop_a_server_the_ui_extension_never_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    connection = _serve_until_shutdown(tmp_path, monkeypatch, ui_extension_loaded=False)

    assert "CALL stop_ui_server()" not in connection.executed_statements
    assert connection.closed


def test_catalog_ui_shutdown_stops_the_server_when_the_ui_extension_is_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    connection = _serve_until_shutdown(tmp_path, monkeypatch, ui_extension_loaded=True)

    assert "CALL stop_ui_server()" in connection.executed_statements
    assert connection.closed


@pytest.mark.parametrize("port", [1, 65535])
def test_catalog_ui_accepts_port_boundaries(tmp_path: Path, port: int) -> None:
    settings = catalog_ui.CatalogUiSettings(catalog_root=tmp_path / "catalog", port=port)

    assert settings.port == port


def test_catalog_ui_accepts_an_int_enum_port(tmp_path: Path) -> None:
    class Port(IntEnum):
        CATALOG_UI = 4213

    settings = catalog_ui.CatalogUiSettings(catalog_root=tmp_path / "catalog", port=Port.CATALOG_UI)

    assert settings.port == Port.CATALOG_UI


@pytest.mark.parametrize(
    ("poll_interval", "type_name"),
    [(True, "bool"), ("0.5", "str"), (None, "NoneType")],
)
def test_catalog_ui_refuses_a_poll_interval_of_the_wrong_type(
    tmp_path: Path, poll_interval: object, type_name: str
) -> None:
    settings = catalog_ui.CatalogUiSettings(catalog_root=tmp_path / "catalog")
    with pytest.raises(
        ValueError,
        match=f"catalog_poll_interval_seconds must be an int or float, got {type_name}",
    ):
        replace(settings, catalog_poll_interval_seconds=poll_interval)


@pytest.mark.parametrize(
    "poll_interval",
    [0, -0.5, float("nan"), float("inf"), float("-inf")],
)
def test_catalog_ui_refuses_a_nonpositive_or_nonfinite_poll_interval(
    tmp_path: Path, poll_interval: float
) -> None:
    with pytest.raises(
        ValueError,
        match="catalog_poll_interval_seconds must be positive and finite",
    ):
        catalog_ui.CatalogUiSettings(
            catalog_root=tmp_path / "catalog",
            catalog_poll_interval_seconds=poll_interval,
        )


@pytest.mark.parametrize("poll_interval", [1, 0.001])
def test_catalog_ui_accepts_a_positive_finite_poll_interval(
    tmp_path: Path, poll_interval: float
) -> None:
    settings = catalog_ui.CatalogUiSettings(
        catalog_root=tmp_path / "catalog",
        catalog_poll_interval_seconds=poll_interval,
    )

    assert settings.catalog_poll_interval_seconds == poll_interval


def test_catalog_ui_refuses_a_port_owned_by_another_process() -> None:
    with socket(AF_INET, SOCK_STREAM) as occupied_port_socket:
        occupied_port_socket.bind(("127.0.0.1", 0))
        occupied_port = int(occupied_port_socket.getsockname()[1])

        with pytest.raises(
            catalog_ui.CatalogUiStartupError,
            match=rf"port {occupied_port} is already in use",
        ):
            catalog_ui._raise_if_loopback_port_is_unavailable(occupied_port)


def _run_bucket_catalog_ui_in_thread(
    catalog_root: Path | str | StorageRoot,
    *,
    shutdown_event: Event,
    background_failures: list[BaseException],
    monkeypatch: pytest.MonkeyPatch,
    server_started_event: Event | None = None,
    empty_catalog_ready_event: Event | None = None,
    catalog_refreshed_event: Event | None = None,
    catalog_connections: list[duckdb.DuckDBPyConnection] | None = None,
    display_labels: list[str] | None = None,
) -> Thread:
    def record_server_start(catalog_connection: duckdb.DuckDBPyConnection, port: int) -> None:
        if catalog_connections is not None:
            catalog_connections.append(catalog_connection)
        if server_started_event is not None:
            server_started_event.set()

    refresh_local_catalog_connection = catalog_ui._refresh_local_catalog_connection
    catalog_connection_contains_episodes = catalog_ui._catalog_connection_contains_episodes

    def record_initial_catalog_read(
        catalog_connection: duckdb.DuckDBPyConnection,
    ) -> bool:
        contains_episodes = catalog_connection_contains_episodes(catalog_connection)
        if empty_catalog_ready_event is not None:
            empty_catalog_ready_event.set()
        return contains_episodes

    def record_catalog_refresh(
        catalog_connection: duckdb.DuckDBPyConnection,
        refreshed_catalog_root: Path | str,
    ) -> None:
        refresh_local_catalog_connection(catalog_connection, refreshed_catalog_root)
        if catalog_refreshed_event is not None:
            catalog_refreshed_event.set()

    monkeypatch.setattr(catalog_ui, "_start_duckdb_ui_server", record_server_start)
    monkeypatch.setattr(
        catalog_ui,
        "_catalog_connection_contains_episodes",
        record_initial_catalog_read,
    )
    monkeypatch.setattr(
        catalog_ui,
        "_refresh_local_catalog_connection",
        record_catalog_refresh,
    )

    real_display_label = catalog_ui._catalog_display_label

    def record_display_label(root: Path | str | StorageRoot) -> str:
        label = real_display_label(root)
        if display_labels is not None:
            display_labels.append(label)
        return label

    monkeypatch.setattr(catalog_ui, "_catalog_display_label", record_display_label)

    def run_catalog_ui() -> None:
        try:
            catalog_ui.serve_catalog_ui(
                catalog_ui.CatalogUiSettings(
                    catalog_root=catalog_root,
                    open_browser=False,
                    catalog_poll_interval_seconds=0.01,
                ),
                shutdown_event=shutdown_event,
            )
        except BaseException as error:
            background_failures.append(error)

    catalog_ui_thread = Thread(target=run_catalog_ui)
    catalog_ui_thread.start()
    return catalog_ui_thread


def test_catalog_ui_starts_against_a_populated_bucket_catalog(
    tmp_path: Path,
    bucket_over_tmp: tuple["BucketStorageRoot", Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root, _remote_dir = bucket_over_tmp
    catalog_root = data_root.child("catalog")
    _append_remote_episode(tmp_path, catalog_root, task="first")
    shutdown_event = Event()
    server_started_event = Event()
    background_failures: list[BaseException] = []
    catalog_connections: list[duckdb.DuckDBPyConnection] = []
    display_labels: list[str] = []
    catalog_ready_event = Event()

    catalog_ui_thread = _run_bucket_catalog_ui_in_thread(
        catalog_root,
        shutdown_event=shutdown_event,
        background_failures=background_failures,
        monkeypatch=monkeypatch,
        server_started_event=server_started_event,
        empty_catalog_ready_event=catalog_ready_event,
        catalog_connections=catalog_connections,
        display_labels=display_labels,
    )
    try:
        assert server_started_event.wait(timeout=2)
        assert catalog_ready_event.wait(timeout=2)
        (catalog_connection,) = catalog_connections
        assert catalog_connection.execute(
            "SELECT count(*), min(task) FROM episodes"
        ).fetchone() == (1, "first")
        assert display_labels == [str(catalog_root)]
    finally:
        shutdown_event.set()
        catalog_ui_thread.join(timeout=2)

    assert not catalog_ui_thread.is_alive()
    assert background_failures == []


def test_catalog_ui_waits_for_the_first_remote_append(
    tmp_path: Path,
    bucket_over_tmp: tuple["BucketStorageRoot", Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root, remote_dir = bucket_over_tmp
    ui_catalog_root = data_root.child("catalog")
    writer_catalog_root = _writer_catalog_root(tmp_path, remote_dir)
    _write_remote_catalog_marker(writer_catalog_root)
    shutdown_event = Event()
    server_started_event = Event()
    empty_catalog_ready_event = Event()
    catalog_refreshed_event = Event()
    background_failures: list[BaseException] = []
    catalog_connections: list[duckdb.DuckDBPyConnection] = []

    catalog_ui_thread = _run_bucket_catalog_ui_in_thread(
        ui_catalog_root,
        shutdown_event=shutdown_event,
        background_failures=background_failures,
        monkeypatch=monkeypatch,
        server_started_event=server_started_event,
        empty_catalog_ready_event=empty_catalog_ready_event,
        catalog_refreshed_event=catalog_refreshed_event,
        catalog_connections=catalog_connections,
    )
    try:
        assert server_started_event.wait(timeout=2)
        assert empty_catalog_ready_event.wait(timeout=2)
        (catalog_connection,) = catalog_connections
        assert catalog_connection.execute("SELECT count(*) FROM episodes").fetchone() == (0,)

        _append_remote_episode(tmp_path, writer_catalog_root, task="remote-first")

        assert catalog_refreshed_event.wait(timeout=2)
        assert catalog_connection.execute(
            "SELECT count(*), min(task) FROM episodes"
        ).fetchone() == (1, "remote-first")
    finally:
        shutdown_event.set()
        catalog_ui_thread.join(timeout=2)

    assert not catalog_ui_thread.is_alive()
    assert background_failures == []


def test_catalog_ui_sees_a_later_remote_append_without_restart(
    tmp_path: Path,
    bucket_over_tmp: tuple["BucketStorageRoot", Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root, remote_dir = bucket_over_tmp
    ui_catalog_root = data_root.child("catalog")
    writer_catalog_root = _writer_catalog_root(tmp_path, remote_dir)
    _append_remote_episode(tmp_path, writer_catalog_root, task="first")
    shutdown_event = Event()
    server_started_event = Event()
    background_failures: list[BaseException] = []
    catalog_connections: list[duckdb.DuckDBPyConnection] = []
    catalog_ready_event = Event()

    catalog_ui_thread = _run_bucket_catalog_ui_in_thread(
        ui_catalog_root,
        shutdown_event=shutdown_event,
        background_failures=background_failures,
        monkeypatch=monkeypatch,
        server_started_event=server_started_event,
        empty_catalog_ready_event=catalog_ready_event,
        catalog_connections=catalog_connections,
    )
    try:
        assert server_started_event.wait(timeout=2)
        assert catalog_ready_event.wait(timeout=2)
        (catalog_connection,) = catalog_connections
        assert catalog_connection.execute("SELECT count(*) FROM episodes").fetchone() == (1,)

        _append_remote_episode(
            tmp_path,
            writer_catalog_root,
            task="second",
            canonical_bytes=b"another canonical episode",
        )

        import time

        start = time.monotonic()
        while time.monotonic() - start < 2:
            row_count = catalog_connection.execute("SELECT count(*) FROM episodes").fetchone()
            if row_count == (2,):
                break
            time.sleep(0.02)
        else:
            raise AssertionError("expected the second remote append to become visible")
    finally:
        shutdown_event.set()
        catalog_ui_thread.join(timeout=2)

    assert not catalog_ui_thread.is_alive()
    assert background_failures == []


def test_catalog_ui_does_not_write_bucket_objects_while_browsing(
    tmp_path: Path,
    bucket_over_tmp: tuple["BucketStorageRoot", Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root, _remote_dir = bucket_over_tmp
    catalog_root = data_root.child("catalog")
    _append_remote_episode(tmp_path, catalog_root, task="stable")
    remote_keys_before = catalog_root.list_names()
    catalog_constructor_calls = 0
    real_catalog = Catalog

    def record_catalog_constructor(root: Path | str) -> Catalog:
        nonlocal catalog_constructor_calls
        catalog_constructor_calls += 1
        return real_catalog(root)

    monkeypatch.setattr(catalog_ui, "Catalog", record_catalog_constructor)
    shutdown_event = Event()
    server_started_event = Event()
    background_failures: list[BaseException] = []

    catalog_ui_thread = _run_bucket_catalog_ui_in_thread(
        catalog_root,
        shutdown_event=shutdown_event,
        background_failures=background_failures,
        monkeypatch=monkeypatch,
        server_started_event=server_started_event,
    )
    try:
        assert server_started_event.wait(timeout=2)
    finally:
        shutdown_event.set()
        catalog_ui_thread.join(timeout=2)

    assert catalog_constructor_calls == 0
    assert catalog_root.list_names() == remote_keys_before
    assert background_failures == []
