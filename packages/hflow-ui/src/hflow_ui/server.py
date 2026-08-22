"""The FastAPI app (pure, testable) and the ``hflow ui`` server entry point.

``create_app`` builds the whole API plus SPA serving from one
:class:`UiSettings` -- no sockets, no side effects. ``serve`` adds the launch
behavior: pick a free port, print the tokened URL, open the browser, run
uvicorn. The only workspace files this package ever writes are the curation
studio's: immutable pinned manifests under ``<data_root>/manifests/`` and
the ``<data_root>/ui/state.json`` sidecar (both refused when
``settings.read_only``); the server never mints workspace identity.
"""

import copy
import importlib.resources
import logging
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
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from uvicorn.config import LOGGING_CONFIG
from uvicorn.logging import AccessFormatter

import hflow
from hflow.format import CATALOG_FORMAT_VERSION
from hflow.steps import RUN_PROFILES, IngestMode
from hflow.storage import LocalStorageRoot
from hflow.workspace import Workspace
from hflow_ui import _catalog, _curation, _graph, _media, _pipeline, _runtime
from hflow_ui._auth import SessionTokenMiddleware
from hflow_ui._settings import UiSettings

ASSETS_ENVIRONMENT_VARIABLE = "HFLOW_UI_ASSETS"

_PORT_RETRY_ATTEMPTS = 10

# A blanket cap on request-body size: comfortably above the curation studio's
# own per-field limits (a 100k SQL body plus JSON overhead), but a hard stop
# on an unbounded POST -- the sidecar is the one file the UI writes outside
# manifests/, so nothing it persists should be able to grow without limit.
_MAX_REQUEST_BODY_BYTES = 4 * 1024 * 1024


class RequestBodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Refuses any request whose declared body exceeds the size cap (413)."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError:
                declared_length = None
            if declared_length is not None and declared_length > _MAX_REQUEST_BODY_BYTES:
                return JSONResponse(
                    {"detail": "request body too large"},
                    status_code=413,
                )
        return await call_next(request)


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
    # Body-size cap is outermost (added last): reject an oversized POST before
    # auth or routing touches it.
    if settings.token is not None:
        application.add_middleware(SessionTokenMiddleware, session_token=settings.token)
    application.add_middleware(RequestBodySizeLimitMiddleware)

    # --pipeline is imported -- EXECUTED -- exactly once, here at app
    # construction; the outcome (the live App, or the remembered failure) is
    # what /api/v1/pipeline and the config capability report for this launch.
    pipeline_state = _pipeline.load_pipeline_state(settings.pipeline)
    # One runtime resolver per launch, shared by the runs monitor and the
    # graph routes so both read the same briefly-cached addressing.
    runtime_resolver = _runtime.RuntimeResolver(settings.data_root)

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
                "read_only": settings.read_only,
                "hflow_version": hflow.__version__,
                "hflow_ui_version": hflow_ui_version,
                "data_root": settings.data_root,
                "workspace_id": identity.workspace_id if identity is not None else None,
                "capabilities": {
                    "catalog": _catalog_marker_readable(workspace),
                    "media": isinstance(workspace.storage_root, LocalStorageRoot),
                    # Addressed (bundle dir or HFLOW_AIRFLOW_URL), not
                    # necessarily reachable -- /runtime/status owns liveness.
                    "runtime": _runtime.runtime_configured(settings.data_root),
                    "pipeline": pipeline_state.available,
                },
                # Deep-link base for the Runs page, honestly derived from the
                # addressed bundle's own records; null when none is addressed.
                "airflow_web_url": _runtime.local_bundle_web_url(settings.data_root),
                # The trigger form's vocabularies, served so the frontend
                # never hardcodes them (hflow.steps stays the one owner).
                "run_profiles": list(RUN_PROFILES),
                "ingest_modes": [mode.value for mode in IngestMode],
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

    # Registered before the {episode_id} route below so the literal path
    # segment "stats" can never be read as an episode id.
    @application.get("/api/v1/episodes/stats")
    def read_episode_stats(
        task: Annotated[list[str] | None, Query()] = None,
        operator: Annotated[list[str] | None, Query()] = None,
        embodiment: Annotated[list[str] | None, Query()] = None,
        status: Annotated[Literal["ok", "quarantined"] | None, Query()] = None,
        success: Annotated[Literal["true", "false"] | None, Query()] = None,
        search: Annotated[str | None, Query()] = None,
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
            stats = _catalog.query_episode_stats(connection, filters)
        finally:
            connection.close()
        return JSONResponse(stats)

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

    @application.get("/api/v1/episodes/{episode_id}/timeline")
    def read_episode_timeline(episode_id: str) -> JSONResponse:
        connection = open_connection_or_refuse()
        try:
            timeline = _catalog.query_episode_timeline(connection, episode_id)
        finally:
            connection.close()
        if timeline is None:
            raise HTTPException(
                status_code=404, detail=f"no episode {episode_id!r} in this catalog"
            )
        return JSONResponse(timeline)

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

    # M1 curation studio, M2 runs monitor and pipeline routes, and the
    # visualization routes -- included BEFORE the SPA catch-all below so they
    # win route matching.
    application.include_router(_curation.create_curation_router(settings))
    application.include_router(_runtime.create_runtime_router(settings, runtime_resolver))
    application.include_router(_pipeline.create_pipeline_router(settings, pipeline_state))
    application.include_router(_graph.create_graph_router(pipeline_state, runtime_resolver))

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


class _QueryStringStrippingAccessFormatter(AccessFormatter):
    """uvicorn access formatter that logs the path WITHOUT its query string.

    The login ``?token=`` (and any other query param) would otherwise land in
    the access line for every request -- a 256-bit credential written to
    stdout, which ``serve`` itself anticipates being piped to a supervisor's
    log. Dropping the query string keeps the useful method/path/status line.
    """

    def formatMessage(self, record: logging.LogRecord) -> str:
        # uvicorn's access record carries (client_addr, method, full_path,
        # http_version, status_code); full_path includes the query string.
        if isinstance(record.args, tuple) and len(record.args) == 5:
            client_addr, method, full_path, http_version, status_code = record.args
            path_without_query = str(full_path).split("?", 1)[0]
            record.args = (client_addr, method, path_without_query, http_version, status_code)
        return super().formatMessage(record)


def _access_log_config() -> dict[str, object]:
    """uvicorn's default logging config with the query-string-dropping access
    formatter swapped in."""
    log_config = copy.deepcopy(LOGGING_CONFIG)
    log_config["formatters"]["access"]["()"] = (
        f"{__name__}.{_QueryStringStrippingAccessFormatter.__name__}"
    )
    return log_config


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
    uvicorn.run(
        application,
        host=settings.host,
        port=chosen_port,
        log_level="info",
        log_config=_access_log_config(),
    )


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
