"""The local DuckDB catalog browser lifecycle."""

from dataclasses import replace
from enum import IntEnum
from pathlib import Path
from socket import AF_INET, SOCK_STREAM, socket
from threading import Event, Thread

import duckdb
import pytest

import hflow
import hflow.catalog_ui as catalog_ui


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
