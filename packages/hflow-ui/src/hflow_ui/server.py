"""The FastAPI app (pure, testable) and the ``hflow ui`` server entry point.

``create_app`` builds the whole read-only API plus SPA serving from one
:class:`UiSettings` -- no sockets, no side effects. ``serve`` adds the launch
behavior: pick a free port, print the tokened URL, open the browser, run
uvicorn. Nothing in this package ever mutates the workspace, and the server
never mints workspace identity.
"""

import importlib.resources
import os
import socket
import threading
import webbrowser
from pathlib import Path
from typing import Annotated, Literal

import duckdb
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

import hflow
from hflow.format import CATALOG_FORMAT_VERSION
from hflow.storage import LocalStorageRoot
from hflow.workspace import Workspace
from hflow_ui import _catalog, _media
from hflow_ui._auth import SessionTokenMiddleware
from hflow_ui._settings import UiSettings

ASSETS_ENVIRONMENT_VARIABLE = "HFLOW_UI_ASSETS"

_PORT_RETRY_ATTEMPTS = 10

_FRONTEND_PLACEHOLDER_PAGE = """<!doctype html>
<html>
  <head><title>HFlow workspace UI</title></head>
  <body>
    <h1>HFlow workspace UI</h1>
    <p>The frontend bundle is not built into this installation, but the JSON
    API is live under <code>/api/v1</code> (interactive docs at
    <code>/api/docs</code>).</p>
    <p>To develop the frontend, run <code>pnpm install &amp;&amp; pnpm dev</code>
    inside the repository's <code>ui/</code> directory; to serve a local build,
    run <code>pnpm build</code> there and point the <code>HFLOW_UI_ASSETS</code>
    environment variable at <code>ui/dist</code>.</p>
  </body>
</html>
"""


def create_app(settings: UiSettings) -> FastAPI:
    """The whole UI server as a plain ASGI app."""
    # Late import: hflow_ui/__init__ imports this module, so the package
    # attribute exists only once init finished -- which any create_app call is.
    from hflow_ui import __version__ as hflow_ui_version

    application = FastAPI(
        title="HFlow workspace UI",
        version=hflow_ui_version,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
    )
    if settings.token is not None:
        application.add_middleware(SessionTokenMiddleware, session_token=settings.token)

    def open_connection_or_refuse() -> duckdb.DuckDBPyConnection:
        # A FRESH connection per request: the wide episodes view binds its
        # measurement columns at open time (hflow.curation), so a held
        # connection would never show keys recorded after startup.
        try:
            return _catalog.open_workspace_connection(settings.data_root)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            # Present but unreadable: a catalog format version this build
            # cannot read is a state conflict, not a missing resource.
            raise HTTPException(status_code=409, detail=str(error)) from error

    @application.get("/api/v1/health")
    def read_health() -> JSONResponse:
        return JSONResponse({"ok": True})

    @application.get("/api/v1/config")
    def read_config() -> JSONResponse:
        workspace = Workspace.parse(settings.data_root)
        try:
            identity = workspace.identity()
        except ValueError:
            # A corrupt identity marker must not stop the UI from booting: the
            # id is informational here, and this surface never mints one.
            identity = None
        return JSONResponse(
            {
                "mode": "local",
                "read_only": True,
                "hflow_version": hflow.__version__,
                "hflow_ui_version": hflow_ui_version,
                "data_root": settings.data_root,
                "workspace_id": identity.workspace_id if identity is not None else None,
                "capabilities": {
                    "catalog": _catalog_marker_readable(workspace),
                    "media": isinstance(workspace.storage_root, LocalStorageRoot),
                    "runtime": False,  # M0: no runs monitor
                },
            }
        )

    @application.get("/api/v1/episodes")
    def list_episodes(
        task: Annotated[list[str] | None, Query()] = None,
        operator: Annotated[list[str] | None, Query()] = None,
        embodiment: Annotated[list[str] | None, Query()] = None,
        status: Annotated[Literal["ok", "quarantined"] | None, Query()] = None,
        success: Annotated[Literal["true", "false"] | None, Query()] = None,
        search: Annotated[str | None, Query()] = None,
        order_by: Annotated[str, Query()] = "recorded_at",
        order: Annotated[Literal["asc", "desc"], Query()] = "desc",
        limit: Annotated[int, Query(ge=1, le=500)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> JSONResponse:
        filters = _catalog.EpisodeListFilters(
            tasks=tuple(task or ()),
            operators=tuple(operator or ()),
            embodiments=tuple(embodiment or ()),
            status=status,
            success=success,
            search=search,
        )
        connection = open_connection_or_refuse()
        try:
            page = _catalog.query_episode_page(
                connection,
                filters,
                order_by=order_by,
                descending=order == "desc",
                limit=limit,
                offset=offset,
            )
        except _catalog.UnknownOrderColumnError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        finally:
            connection.close()
        return JSONResponse(
            {"rows": page.rows, "total": page.total, "columns": page.columns, "sql": page.sql}
        )

    @application.get("/api/v1/episodes/facets")
    def read_episode_facets() -> JSONResponse:
        connection = open_connection_or_refuse()
        try:
            facets = _catalog.query_episode_facets(connection)
        finally:
            connection.close()
        return JSONResponse(facets)

    @application.get("/api/v1/episodes/{episode_id}")
    def read_episode(episode_id: str) -> JSONResponse:
        connection = open_connection_or_refuse()
        try:
            dossier = _catalog.query_episode_dossier(
                connection, episode_id, data_root=settings.data_root
            )
        finally:
            connection.close()
        if dossier is None:
            raise HTTPException(
                status_code=404, detail=f"no episode {episode_id!r} in this catalog"
            )
        return JSONResponse(dossier)

    @application.get("/api/v1/episodes/{episode_id}/media/{artifact_name:path}")
    def read_episode_media(episode_id: str, artifact_name: str) -> FileResponse:
        connection = open_connection_or_refuse()
        try:
            media_uri = _catalog.find_media_uri(connection, episode_id, artifact_name)
        finally:
            connection.close()
        if media_uri is None:
            raise HTTPException(
                status_code=404,
                detail=f"episode {episode_id!r} has no media artifact named {artifact_name!r}",
            )
        return _served_file_response_or_refuse(media_uri, settings.data_root)

    @application.get("/api/v1/episodes/{episode_id}/canonical")
    def read_episode_canonical(episode_id: str) -> FileResponse:
        connection = open_connection_or_refuse()
        try:
            canonical_uri = _catalog.find_canonical_uri(connection, episode_id)
        finally:
            connection.close()
        if canonical_uri is None:
            raise HTTPException(
                status_code=404, detail=f"no episode {episode_id!r} in this catalog"
            )
        return _served_file_response_or_refuse(canonical_uri, settings.data_root)

    @application.get("/{requested_path:path}", include_in_schema=False)
    def serve_spa(requested_path: str) -> Response:
        return _spa_response(settings, requested_path)

    return application


def _catalog_marker_readable(workspace: Workspace) -> bool:
    """Whether the catalog's format marker is present and this build reads it."""
    try:
        found_version = workspace.catalog_root.read_bytes("format_version").decode().strip()
    except (OSError, UnicodeDecodeError):
        return False
    return found_version == CATALOG_FORMAT_VERSION


def _served_file_response_or_refuse(uri: str, data_root: str) -> FileResponse:
    try:
        resolved_file = _media.resolve_served_file(uri, data_root=data_root)
    except _media.MediaResolutionError as error:
        raise HTTPException(status_code=error.status_code, detail=error.detail) from error
    return _media.served_file_response(resolved_file)


def _assets_directory(settings: UiSettings) -> Path | None:
    """Where the built SPA lives: explicit setting, env override, then the wheel."""
    if settings.assets_dir is not None:
        return settings.assets_dir
    environment_override = os.environ.get(ASSETS_ENVIRONMENT_VARIABLE)
    if environment_override:
        return Path(environment_override)
    # This package ships as a plain directory wheel (uv_build, never zipped),
    # so the packaged resource is always a real filesystem path.
    packaged_static = Path(str(importlib.resources.files("hflow_ui").joinpath("static")))
    return packaged_static if packaged_static.is_dir() else None


def _spa_response(settings: UiSettings, requested_path: str) -> Response:
    if requested_path == "api" or requested_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="unknown API path")
    assets_directory = _assets_directory(settings)
    if assets_directory is not None and requested_path:
        asset_response = _contained_asset_response(assets_directory, requested_path)
        if asset_response is not None:
            return asset_response
    final_segment = requested_path.rsplit("/", 1)[-1]
    if "." in final_segment:
        # Looks like a file: a missing asset is a 404, never index.html.
        raise HTTPException(status_code=404, detail="no such asset")
    if assets_directory is not None:
        index_file = assets_directory / "index.html"
        if index_file.is_file():
            return FileResponse(index_file)
    return HTMLResponse(_FRONTEND_PLACEHOLDER_PAGE)


def _contained_asset_response(assets_directory: Path, requested_path: str) -> FileResponse | None:
    resolved_assets_directory = assets_directory.resolve()
    try:
        resolved_candidate = (assets_directory / requested_path).resolve(strict=True)
    except (OSError, ValueError):
        return None
    if not resolved_candidate.is_relative_to(resolved_assets_directory):
        # Traversal outside the assets tree is answered as if absent.
        return None
    if not resolved_candidate.is_file():
        return None
    return FileResponse(resolved_candidate)


def serve(settings: UiSettings) -> None:
    """Run the workspace UI: free port, printed (tokened) URL, browser, uvicorn."""
    application = create_app(settings)
    chosen_port = _first_free_port(settings.host, settings.port)
    if chosen_port != settings.port:
        # flush=True throughout: the login URL must reach a piped stdout
        # (tee, a supervisor's log) before the blocking uvicorn.run call.
        print(
            f"hflow ui: port {settings.port} is taken; serving on port {chosen_port} instead",
            flush=True,
        )
    url_host = "127.0.0.1" if settings.host == "0.0.0.0" else settings.host
    login_url = f"http://{url_host}:{chosen_port}/"
    if settings.token is not None:
        login_url += f"?token={settings.token}"
    print(f"hflow ui: browsing {settings.data_root} at {login_url}", flush=True)
    if settings.open_browser:
        # uvicorn.run blocks this thread; a short timer opens the browser
        # once the server has had time to bind.
        browser_timer = threading.Timer(1.0, webbrowser.open, args=[login_url])
        browser_timer.daemon = True
        browser_timer.start()
    uvicorn.run(application, host=settings.host, port=chosen_port, log_level="info")


def _first_free_port(host: str, preferred_port: int) -> int:
    """The preferred port, or the first free one in the handful above it."""
    for candidate_port in range(preferred_port, preferred_port + _PORT_RETRY_ATTEMPTS):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe_socket:
            try:
                probe_socket.bind((host, candidate_port))
            except OSError:
                continue
        return candidate_port
    raise RuntimeError(
        f"no free port between {preferred_port} and "
        f"{preferred_port + _PORT_RETRY_ATTEMPTS - 1} on {host}"
    )
