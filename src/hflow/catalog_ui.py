"""DuckDB's browser UI over a local or bucket-backed HFlow catalog."""

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
from hflow.curation import (
    _empty_table_relations_have_landed_parquet,
    _refresh_local_catalog_connection,
    _sync_catalog_mirror,
    open_catalog_connection,
)
from hflow.storage import (
    BucketStorageRoot,
    LocalStorageRoot,
    StorageRoot,
    parse_storage_root,
)

DEFAULT_CATALOG_UI_PORT = 4213
DEFAULT_CATALOG_POLL_INTERVAL_SECONDS = 0.5


class CatalogUiStartupError(RuntimeError):
    """DuckDB UI could not start, so no catalog browser is serving."""


@dataclass(frozen=True)
class CatalogUiSettings:
    """Configuration for one DuckDB catalog browser."""

    catalog_root: Path | str | StorageRoot
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


def _catalog_display_label(catalog_root: Path | str | StorageRoot) -> str:
    """The catalog location to show the operator, never the mirror directory."""
    return str(parse_storage_root(catalog_root))


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


def _catalog_connection_awaits_parquet(
    catalog_connection: duckdb.DuckDBPyConnection,
) -> bool:
    """Whether the open surface still has no episodes and no ingest failures."""
    if _catalog_connection_contains_episodes(catalog_connection):
        return False
    failure_count_row = catalog_connection.execute(
        "SELECT count(*) FROM ingest_failures"
    ).fetchone()
    return failure_count_row is None or int(failure_count_row[0]) == 0


def _prepare_catalog_for_ui(location: StorageRoot) -> None:
    """Ensure a local catalog exists; bucket catalogs must already be present."""
    if isinstance(location, LocalStorageRoot):
        Catalog(location.path)


def serve_catalog_ui(settings: CatalogUiSettings, *, shutdown_event: Event | None = None) -> None:
    """Serve DuckDB UI now, including when the catalog is still empty.

    Empty catalog relations begin as in-memory tables because DuckDB refuses a
    Parquet glob with no matches. Whenever those empty shells gain Parquet on
    disk — the first episode append, a later ``ingest_failures`` ledger write,
    or any other long table that lands files mid-session — the poll loop
    rebinds them to Parquet-backed views on the same connection so the open UI
    stays current without restarting this command.

    Bucket catalogs are read-only: the remote ``format_version`` marker must
    already exist and this command never creates or changes object-store keys.
    """
    effective_shutdown_event = shutdown_event or Event()
    catalog_root = settings.catalog_root
    location = parse_storage_root(catalog_root)
    bucket_catalog = isinstance(location, BucketStorageRoot)

    _prepare_catalog_for_ui(location)
    catalog_connection = open_catalog_connection(catalog_root)
    server_started = False
    try:
        _start_duckdb_ui_server(catalog_connection, settings.port)
        server_started = True

        browser_url = f"http://127.0.0.1:{settings.port}"
        print(f"DuckDB UI: {browser_url}", flush=True)
        print(f"Catalog: {_catalog_display_label(catalog_root)}", flush=True)
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

        waiting_for_first_parquet = _catalog_connection_awaits_parquet(catalog_connection)
        if waiting_for_first_parquet:
            print(
                "Catalog is empty. The UI is ready and is waiting for catalog "
                "Parquet (episodes or ingest_failures).",
                flush=True,
            )

        while not effective_shutdown_event.wait(settings.catalog_poll_interval_seconds):
            if bucket_catalog:
                _sync_catalog_mirror(catalog_root)
            if not _empty_table_relations_have_landed_parquet(catalog_connection, catalog_root):
                continue
            _refresh_local_catalog_connection(catalog_connection, catalog_root)
            if waiting_for_first_parquet:
                print(
                    "Catalog Parquet detected. Views are now available in the open UI.",
                    flush=True,
                )
                waiting_for_first_parquet = False
            else:
                print(
                    "New catalog Parquet bound into the open UI.",
                    flush=True,
                )
    except KeyboardInterrupt:
        pass
    finally:
        _stop_duckdb_ui_server(catalog_connection, server_started)
