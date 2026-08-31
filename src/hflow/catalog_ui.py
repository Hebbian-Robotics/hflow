"""DuckDB's browser UI over a local HFlow catalog."""

from __future__ import annotations

import math
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from socket import AF_INET, SOCK_STREAM, socket
from threading import Event

import duckdb

from hflow.catalog import Catalog
from hflow.curation import _refresh_local_catalog_connection, open_catalog_connection

DEFAULT_CATALOG_UI_PORT = 4213
DEFAULT_CATALOG_POLL_INTERVAL_SECONDS = 0.5


class CatalogUiStartupError(RuntimeError):
    """DuckDB UI could not start, so no catalog browser is serving."""


@dataclass(frozen=True)
class CatalogUiSettings:
    """Configuration for one local DuckDB catalog browser."""

    catalog_root: Path
    port: int = DEFAULT_CATALOG_UI_PORT
    open_browser: bool = True
    catalog_poll_interval_seconds: float = DEFAULT_CATALOG_POLL_INTERVAL_SECONDS

    def __post_init__(self) -> None:
        if isinstance(self.port, bool) or not isinstance(self.port, int):
            raise ValueError(f"port must be an int, got {type(self.port).__name__}")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")

        if isinstance(self.catalog_poll_interval_seconds, bool) or not isinstance(
            self.catalog_poll_interval_seconds, int | float
        ):
            raise ValueError(
                "catalog_poll_interval_seconds must be an int or float, got "
                f"{type(self.catalog_poll_interval_seconds).__name__}"
            )
        if (
            not math.isfinite(self.catalog_poll_interval_seconds)
            or self.catalog_poll_interval_seconds <= 0
        ):
            raise ValueError(
                "catalog_poll_interval_seconds must be positive and finite, got "
                f"{self.catalog_poll_interval_seconds}"
            )


def _raise_if_loopback_port_is_unavailable(port: int) -> None:
    with socket(AF_INET, SOCK_STREAM) as port_probe_socket:
        try:
            port_probe_socket.bind(("127.0.0.1", port))
        except OSError as error:
            raise CatalogUiStartupError(
                f"port {port} is already in use on 127.0.0.1; stop the existing "
                "process or pass --port with another value"
            ) from error


def _start_duckdb_ui_server(catalog_connection: duckdb.DuckDBPyConnection, port: int) -> None:
    try:
        catalog_connection.execute("INSTALL ui")
        catalog_connection.execute("LOAD ui")
        # DuckDB 1.5 can report that start_ui_server succeeded even when
        # another DuckDB UI owns the requested port. Probe it ourselves before
        # printing a URL that could belong to a different catalog.
        _raise_if_loopback_port_is_unavailable(port)
        catalog_connection.execute(f"SET ui_local_port = {port}")
        catalog_connection.execute("CALL start_ui_server()")
    except CatalogUiStartupError:
        raise
    except duckdb.Error as error:
        raise CatalogUiStartupError(
            "DuckDB UI could not start. Check that the port is free and that "
            "the first launch can download DuckDB's ui extension. "
            f"DuckDB reported: {error}"
        ) from error


def _ui_extension_is_loaded(catalog_connection: duckdb.DuckDBPyConnection) -> bool:
    row = catalog_connection.execute(
        "SELECT count(*) FROM duckdb_extensions() WHERE extension_name = 'ui' AND loaded"
    ).fetchone()
    return row is not None and int(row[0]) > 0


def _stop_duckdb_ui_server(
    catalog_connection: duckdb.DuckDBPyConnection, server_started: bool
) -> None:
    try:
        # stop_ui_server only exists once the ui extension is loaded. Calling it
        # without the extension makes DuckDB auto-install the extension from the
        # network -- seconds on a slow connection, and pointless: there is no
        # server to stop if the extension was never loaded.
        if server_started and _ui_extension_is_loaded(catalog_connection):
            catalog_connection.execute("CALL stop_ui_server()")
    except duckdb.Error:
        # The process is already leaving. A server that stopped independently
        # should not turn a clean Ctrl+C into a traceback.
        pass
    finally:
        catalog_connection.close()


def _catalog_connection_contains_episodes(
    catalog_connection: duckdb.DuckDBPyConnection,
) -> bool:
    episode_count_row = catalog_connection.execute("SELECT count(*) FROM episodes_raw").fetchone()
    return episode_count_row is not None and int(episode_count_row[0]) > 0


def _first_completed_append_exists(catalog_root: Path) -> bool:
    # Catalog.append_episode writes the episodes file last. Its presence is
    # therefore the commit marker for all table files in that append.
    return any((catalog_root / "episodes").glob("*.parquet"))


def serve_catalog_ui(settings: CatalogUiSettings, *, shutdown_event: Event | None = None) -> None:
    """Serve DuckDB UI now, including when the local catalog is still empty.

    Empty catalog relations begin as in-memory tables because DuckDB refuses a
    Parquet glob with no matches. The first completed append replaces those
    tables with the normal Parquet-backed HFlow views on the same connection,
    so the already-open UI becomes queryable without restarting this command.
    """
    effective_shutdown_event = shutdown_event or Event()

    # Creating a local Catalog is safe before the runtime starts. It writes the
    # format marker and empty table directories that the ingest will use later.
    Catalog(settings.catalog_root)
    catalog_connection = open_catalog_connection(settings.catalog_root)
    server_started = False
    try:
        _start_duckdb_ui_server(catalog_connection, settings.port)
        server_started = True

        browser_url = f"http://127.0.0.1:{settings.port}"
        print(f"DuckDB UI: {browser_url}", flush=True)
        print(f"Catalog: {settings.catalog_root.resolve()}", flush=True)
        print("Press Ctrl+C to stop DuckDB UI.", flush=True)
        if settings.open_browser:
            try:
                webbrowser.open(browser_url)
            except webbrowser.Error as error:
                print(
                    f"catalog ui: could not open a browser automatically: {error}",
                    file=sys.stderr,
                    flush=True,
                )

        if not _catalog_connection_contains_episodes(catalog_connection):
            print(
                "Catalog is empty. The UI is ready and is waiting for the first completed append.",
                flush=True,
            )
            while not effective_shutdown_event.wait(settings.catalog_poll_interval_seconds):
                if _first_completed_append_exists(settings.catalog_root):
                    _refresh_local_catalog_connection(catalog_connection, settings.catalog_root)
                    print(
                        "First completed append detected. Catalog views are now "
                        "available in the open UI.",
                        flush=True,
                    )
                    break

        while not effective_shutdown_event.wait(settings.catalog_poll_interval_seconds):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        _stop_duckdb_ui_server(catalog_connection, server_started)
