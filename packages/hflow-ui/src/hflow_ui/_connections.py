"""Opening a catalog connection for one request -- and refusing as one voice.

Every request that reads the catalog opens a FRESH connection (see
``_catalog``'s module note for why) and must close it again, and every one of
them owes the caller the same answer when the workspace cannot serve it. Both
facts live here so no route restates either:

- a data root with no catalog is a MISSING RESOURCE (404);
- a catalog present but written in a format version this build cannot read is
  a STATE CONFLICT (409) -- the workspace is there, this build just cannot
  speak to it.

The context managers are the only supported way to open a connection inside a
request: they own the ``open -> use -> close`` shape too, so no endpoint
hand-writes another ``try/finally``.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import duckdb
from fastapi import HTTPException

from hflow.curation import open_catalog_connection
from hflow.workspace import Workspace
from hflow_ui import _catalog


def catalog_unavailable_refusal(error: FileNotFoundError | ValueError) -> HTTPException:
    """The HTTP refusal one unusable catalog maps to (see the module note)."""
    if isinstance(error, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(error))
    return HTTPException(status_code=409, detail=str(error))


@contextmanager
def opened_workspace_connection_or_refuse(data_root: str) -> Iterator[duckdb.DuckDBPyConnection]:
    """A live (UTC-pinned) connection for the server's OWN queries."""
    try:
        connection = _catalog.open_workspace_connection(data_root)
    except (FileNotFoundError, ValueError) as error:
        raise catalog_unavailable_refusal(error) from error
    try:
        yield connection
    finally:
        connection.close()


@contextmanager
def opened_workspace_connection_or_none(
    data_root: str,
) -> Iterator[duckdb.DuckDBPyConnection | None]:
    """The same connection, but a workspace with NO catalog yields ``None``.

    For the endpoints where "nothing has been recorded yet" is an answer
    rather than a 404. A catalog this build cannot read still refuses: that is
    a conflict either way.
    """
    try:
        connection = _catalog.open_workspace_connection(data_root)
    except FileNotFoundError:
        yield None
        return
    except ValueError as error:
        raise catalog_unavailable_refusal(error) from error
    try:
        yield connection
    finally:
        connection.close()


@contextmanager
def opened_constrained_connection_or_refuse(data_root: str) -> Iterator[duckdb.DuckDBPyConnection]:
    """The connection USER SQL runs on: catalog materialized in memory, file
    access and extension loading locked out.

    Its configuration is locked at open, so the ``SET TimeZone`` pin the live
    connection uses cannot apply here; timestamp columns are instead rendered
    to UTC ISO text in SQL (``_curation._timestamp_replace_clause``).
    """
    try:
        connection = open_catalog_connection(
            Workspace.parse(data_root).catalog_root, constrained=True
        )
    except (FileNotFoundError, ValueError) as error:
        raise catalog_unavailable_refusal(error) from error
    try:
        yield connection
    finally:
        connection.close()
