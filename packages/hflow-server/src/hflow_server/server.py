"""The FastAPI app (pure, testable) and the ``hflow serve`` server entry point.

``create_app`` builds the whole API plus SPA serving from one
:class:`ServerSettings` -- no sockets, no side effects. ``serve`` adds the launch
behavior: pick a free port, print the URL, open the browser, run uvicorn. The
server authenticates nobody: whoever can reach the bound address gets the
whole API (docs/SERVE.md, "Trust posture"). The only workspace files this package
ever writes are the curation studio's: immutable pinned manifests under
``<data_root>/manifests/`` and the ``<data_root>/curation/state.json`` sidecar (both
refused when ``settings.read_only``); the server never mints workspace
identity.
"""

import importlib.resources
import os
import socket
import threading
import webbrowser
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

import hflow
from hflow.format import CATALOG_FORMAT_VERSION
from hflow.steps import RUN_PROFILES, IngestMode
from hflow.workspace import Workspace
from hflow_server import _catalog, _connections, _curation, _graph, _media, _pipeline, _runtime
from hflow_server._contract import (
    BINARY_FILE_RESPONSES,
    EpisodeDossierResponse,
    EpisodeFacetsResponse,
    EpisodePageResponse,
    EpisodeStatsResponse,
    EpisodeStatus,
    EpisodeTimelineResponse,
    HealthResponse,
    ListingOrder,
    SuccessFilterValue,
    WorkspaceCapabilities,
    WorkspaceConfigResponse,
)
from hflow_server._settings import MAX_PORT, ServerSettings, local_data_root_or_none

ASSETS_ENVIRONMENT_VARIABLE = "HFLOW_UI_ASSETS"

_PORT_RETRY_ATTEMPTS = 10

# A blanket cap on request-body size: comfortably above the curation studio's
# own per-field limits (a 100k SQL body plus JSON overhead), but a hard stop
# on an unbounded POST -- the sidecar is the one file this server writes outside
# manifests/, so nothing it persists should be able to grow without limit.
_MAX_REQUEST_BODY_BYTES = 4 * 1024 * 1024


def _declared_request_body_bytes(scope: Scope) -> int | None:
    """The request's Content-Length, or None when it declares none (or lies)."""
    declared = Headers(scope=scope).get("content-length")
    if declared is None:
        return None
    try:
        return int(declared)
    except ValueError:
        return None


class RequestBodySizeLimitMiddleware:
    """Refuses any request body over the size cap with a 413.

    Pure ASGI, and it counts the bytes rather than trusting a header: a
    declared ``Content-Length`` is only a claim, and a chunked request makes
    none at all, so a header-only check let exactly the thing this middleware
    exists to stop -- an unbounded POST buffered whole before any validation
    runs -- through by simply omitting the header. A declared length over the
    cap is still refused up front, without reading a byte.

    The body is read HERE and replayed downstream rather than counted inside
    a wrapped receive channel: an oversized-body error raised from inside the
    channel gets rewritten by whatever was reading it (FastAPI's body reader
    turns any exception but its own into a generic 400, and an intervening
    ``BaseHTTPMiddleware`` collapses even that into an exception group), which
    would leave the cap enforced but unsayable. What is buffered is bounded by
    the cap itself -- the first chunk that crosses it ends the request -- and
    every route on this API reads its whole body anyway, so nothing that would
    otherwise have streamed is being held here.
    """

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        declared_body_bytes = _declared_request_body_bytes(scope)
        if declared_body_bytes is not None and declared_body_bytes > _MAX_REQUEST_BODY_BYTES:
            await _refuse_oversized_body(scope, receive, send)
            return
        body_messages = await _capped_body_messages(receive)
        if body_messages is None:
            await _refuse_oversized_body(scope, receive, send)
            return
        await self._app(scope, _replaying_receive(body_messages, receive), send)


async def _capped_body_messages(receive: Receive) -> list[Message] | None:
    """One request's body messages, or ``None`` once they exceed the cap."""
    body_messages: list[Message] = []
    received_body_bytes = 0
    while True:
        message = await receive()
        body_messages.append(message)
        if message["type"] != "http.request":
            # http.disconnect: no body is coming, and none ever will.
            return body_messages
        received_body_bytes += len(message.get("body", b""))
        if received_body_bytes > _MAX_REQUEST_BODY_BYTES:
            return None
        if not message.get("more_body", False):
            return body_messages


def _replaying_receive(body_messages: list[Message], receive: Receive) -> Receive:
    """A receive channel handing back the read body, then the real channel."""
    unread_messages = iter(body_messages)

    async def replaying_receive() -> Message:
        unread = next(unread_messages, None)
        # Past the buffered body the real channel takes over, so a downstream
        # reader still sees the eventual http.disconnect.
        return unread if unread is not None else await receive()

    return replaying_receive


async def _refuse_oversized_body(scope: Scope, receive: Receive, send: Send) -> None:
    await JSONResponse({"detail": "request body too large"}, status_code=413)(scope, receive, send)


_FRONTEND_PLACEHOLDER_PAGE = """<!doctype html>
<html>
  <head><title>HFlow workspace API</title></head>
  <body>
    <h1>HFlow workspace API</h1>
    <p>No frontend bundle is installed here. The REST API is live under
    <code>/api/v1</code>, and its OpenAPI schema is at
    <code>/api/openapi.json</code> &mdash; that schema is the product surface:
    everything a UI can show is reachable from it, so any client can be built
    against it without touching this package.</p>
    <p>To serve your own build, point the <code>HFLOW_UI_ASSETS</code>
    environment variable at a directory containing an
    <code>index.html</code>, or pass <code>assets_dir</code> to
    <code>ServerSettings</code>. A bundle packaged inside <code>hflow_server</code>
    is picked up automatically.</p>
  </body>
</html>
"""


def parse_episode_list_filters(
    task: Annotated[list[str] | None, Query()] = None,
    operator: Annotated[list[str] | None, Query()] = None,
    embodiment: Annotated[list[str] | None, Query()] = None,
    orchestrator_run_id: Annotated[list[str] | None, Query()] = None,
    status: Annotated[EpisodeStatus | None, Query()] = None,
    success: Annotated[SuccessFilterValue | None, Query()] = None,
    search: Annotated[str | None, Query()] = None,
) -> _catalog.EpisodeListFilters:
    """The filter params /episodes and /episodes/stats BOTH accept.

    One owner for the pair: the two endpoints must describe the same rows, so
    a filter added here reaches the listing and its distributions together --
    they cannot drift into accepting different query strings.
    """
    return _catalog.EpisodeListFilters(
        tasks=tuple(task or ()),
        operators=tuple(operator or ()),
        embodiments=tuple(embodiment or ()),
        orchestrator_run_ids=tuple(orchestrator_run_id or ()),
        status=status,
        success=success,
        search=search,
    )


EpisodeListFilterParams = Annotated[
    _catalog.EpisodeListFilters, Depends(parse_episode_list_filters)
]


def create_app(settings: ServerSettings) -> FastAPI:
    """The whole workspace server as a plain ASGI app."""
    # Late import: hflow_server/__init__ imports this module, so the package
    # attribute exists only once init finished -- which any create_app call is.
    from hflow_server import __version__ as hflow_server_version

    application = FastAPI(
        title="HFlow workspace API",
        version=hflow_server_version,
        # No Swagger or ReDoc HTML page. Both of FastAPI's built-in pages load
        # their JS and CSS from cdn.jsdelivr.net, which would break the offline
        # promise this UI makes (docs/SERVE.md, "Trust posture": no CDN, no
        # outbound requests) and would run third-party script same-origin with
        # this workspace's API. The generated schema is served as JSON instead
        # -- that IS the contract, and any local OpenAPI viewer or client
        # generator reads it. test_ui_offline_posture.py pins this.
        docs_url=None,
        openapi_url="/api/openapi.json",
        redoc_url=None,
    )
    # Starlette runs middleware outermost-first in REVERSE registration order,
    # so the body-size cap being registered last is what makes it outermost:
    # an oversized POST is refused before routing touches it. There is no
    # request guard here -- this server authenticates nobody (docs/SERVE.md,
    # "Trust posture") -- and if one is ever added, this is where it goes, in
    # front of the routes and behind the cap.
    application.add_middleware(RequestBodySizeLimitMiddleware)

    # --pipeline is imported -- EXECUTED -- exactly once, here at app
    # construction; the outcome (the live App, or the remembered failure) is
    # what /api/v1/pipeline and the config capability report for this launch.
    pipeline_state = _pipeline.load_pipeline_state(settings.pipeline)
    # One runtime resolver per launch, shared by the runs monitor and the
    # graph routes so both read the same briefly-cached addressing.
    runtime_resolver = _runtime.RuntimeResolver(settings.data_root)

    @application.get("/api/v1/health")
    def read_health() -> HealthResponse:
        return HealthResponse(ok=True)

    @application.get("/api/v1/config")
    def read_config() -> WorkspaceConfigResponse:
        workspace = Workspace.parse(settings.data_root)
        try:
            identity = workspace.identity()
        except ValueError:
            # A corrupt identity marker must not stop the server from booting: the
            # id is informational here, and this surface never mints one.
            identity = None
        workspace_is_local = local_data_root_or_none(settings.data_root) is not None
        return WorkspaceConfigResponse(
            mode="local",
            read_only=settings.read_only,
            hflow_version=hflow.__version__,
            hflow_server_version=hflow_server_version,
            data_root=settings.data_root,
            workspace_id=identity.workspace_id if identity is not None else None,
            capabilities=WorkspaceCapabilities(
                catalog=_catalog_marker_readable(workspace),
                # Media bytes and the studio's writes both need the workspace
                # reachable as local paths; they are separate flags because
                # bucket support will arrive for them separately.
                media=workspace_is_local,
                curation=workspace_is_local,
                # Addressed (bundle dir or HFLOW_AIRFLOW_URL), not necessarily
                # reachable -- /runtime/status owns liveness, and it is also
                # the one endpoint that serves the Airflow deep-link base.
                runtime=_runtime.runtime_addressed(runtime_resolver.resolve()),
                pipeline=isinstance(pipeline_state, _pipeline.PipelineLoaded),
            ),
            # The trigger form's vocabularies, served so the frontend never
            # hardcodes them (hflow.steps stays the one owner).
            run_profiles=list(RUN_PROFILES),
            ingest_modes=[mode.value for mode in IngestMode],
        )

    @application.get("/api/v1/episodes")
    def list_episodes(
        filters: EpisodeListFilterParams,
        order_by: Annotated[str, Query()] = "recorded_at",
        order: Annotated[ListingOrder, Query()] = "desc",
        limit: Annotated[int, Query(ge=1, le=500)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> EpisodePageResponse:
        with _connections.opened_workspace_connection_or_refuse(settings.data_root) as connection:
            try:
                return _catalog.query_episode_page(
                    connection,
                    filters,
                    order_by=order_by,
                    descending=order == "desc",
                    limit=limit,
                    offset=offset,
                )
            except _catalog.UnknownOrderColumnError as error:
                raise HTTPException(status_code=400, detail=str(error)) from error

    @application.get("/api/v1/episodes/facets")
    def read_episode_facets() -> EpisodeFacetsResponse:
        with _connections.opened_workspace_connection_or_refuse(settings.data_root) as connection:
            return _catalog.query_episode_facets(connection)

    # Registered before the {episode_id} route below so the literal path
    # segment "stats" can never be read as an episode id.
    @application.get("/api/v1/episodes/stats")
    def read_episode_stats(filters: EpisodeListFilterParams) -> EpisodeStatsResponse:
        with _connections.opened_workspace_connection_or_refuse(settings.data_root) as connection:
            return _catalog.query_episode_stats(connection, filters)

    @application.get("/api/v1/episodes/{episode_id}")
    def read_episode(episode_id: str) -> EpisodeDossierResponse:
        with _connections.opened_workspace_connection_or_refuse(settings.data_root) as connection:
            dossier = _catalog.query_episode_dossier(
                connection, episode_id, data_root=settings.data_root
            )
        if dossier is None:
            raise HTTPException(
                status_code=404, detail=f"no episode {episode_id!r} in this catalog"
            )
        return dossier

    @application.get("/api/v1/episodes/{episode_id}/timeline")
    def read_episode_timeline(episode_id: str) -> EpisodeTimelineResponse:
        with _connections.opened_workspace_connection_or_refuse(settings.data_root) as connection:
            timeline = _catalog.query_episode_timeline(connection, episode_id)
        if timeline is None:
            raise HTTPException(
                status_code=404, detail=f"no episode {episode_id!r} in this catalog"
            )
        return timeline

    @application.get(
        "/api/v1/episodes/{episode_id}/media/{artifact_name:path}",
        response_class=FileResponse,
        responses=BINARY_FILE_RESPONSES,
    )
    def read_episode_media(episode_id: str, artifact_name: str) -> FileResponse:
        with _connections.opened_workspace_connection_or_refuse(settings.data_root) as connection:
            media_uri = _catalog.find_media_uri(connection, episode_id, artifact_name)
        if media_uri is None:
            raise HTTPException(
                status_code=404,
                detail=f"episode {episode_id!r} has no media artifact named {artifact_name!r}",
            )
        return _served_file_response_or_refuse(media_uri, settings.data_root)

    @application.get(
        "/api/v1/episodes/{episode_id}/canonical",
        response_class=FileResponse,
        responses=BINARY_FILE_RESPONSES,
    )
    def read_episode_canonical(episode_id: str) -> FileResponse:
        with _connections.opened_workspace_connection_or_refuse(settings.data_root) as connection:
            canonical_uri = _catalog.find_canonical_uri(connection, episode_id)
        if canonical_uri is None:
            raise HTTPException(
                status_code=404, detail=f"no episode {episode_id!r} in this catalog"
            )
        return _served_file_response_or_refuse(canonical_uri, settings.data_root)

    # The curation studio, runs monitor, pipeline and visualization routes --
    # included BEFORE the SPA catch-all below so they win route matching.
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
    """Bytes for one catalog URI. An unservable URI raises its own refusal
    (a ``MediaResolutionError`` IS an ``HTTPException``), so there is nothing
    to catch and convert here."""
    return _media.served_file_response(_media.resolve_served_file(uri, data_root=data_root))


def _assets_directory(settings: ServerSettings) -> Path | None:
    """Where the built SPA lives: explicit setting, env override, then the wheel."""
    if settings.assets_dir is not None:
        return settings.assets_dir
    environment_override = os.environ.get(ASSETS_ENVIRONMENT_VARIABLE)
    if environment_override:
        return Path(environment_override)
    # This package ships as a plain directory wheel (uv_build, never zipped),
    # so the packaged resource is always a real filesystem path.
    packaged_static = Path(str(importlib.resources.files("hflow_server").joinpath("static")))
    return packaged_static if packaged_static.is_dir() else None


def _spa_response(settings: ServerSettings, requested_path: str) -> Response:
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


class ServerStartupError(RuntimeError):
    """The launch could not be started, and nothing is serving.

    Its own type so ``hflow serve`` can answer a launch that never got off the
    ground with exit 2 (bad input, nothing happened) without also catching a
    RuntimeError raised out of ``uvicorn.run`` once the server is up, which is
    exit 1 (started, then failed). Subclasses RuntimeError so callers that
    already catch that keep working.
    """


def serve(settings: ServerSettings) -> None:
    """Run the workspace server: free port, printed URL, browser, uvicorn."""
    application = create_app(settings)
    chosen_port = _first_free_port(settings.host, settings.port)
    if chosen_port != settings.port:
        # flush=True throughout: the URL must reach a piped stdout (tee, a
        # supervisor's log) before the blocking uvicorn.run call.
        print(
            f"hflow serve: port {settings.port} is taken; serving on port {chosen_port} instead",
            flush=True,
        )
    url_host = "127.0.0.1" if settings.host == "0.0.0.0" else settings.host
    workspace_url = f"http://{url_host}:{chosen_port}/"
    print(f"hflow serve: serving {settings.data_root} at {workspace_url}", flush=True)
    if settings.open_browser:
        # uvicorn.run blocks this thread; a short timer opens the browser
        # once the server has had time to bind.
        browser_timer = threading.Timer(1.0, webbrowser.open, args=[workspace_url])
        browser_timer.daemon = True
        browser_timer.start()
    # uvicorn's stock logging config: the access line's query string carries
    # episode filters and paging, which are useful when debugging a request
    # and are not credentials -- this server has none.
    uvicorn.run(application, host=settings.host, port=chosen_port, log_level="info")


def _first_free_port(host: str, preferred_port: int) -> int:
    """The preferred port, or the first free one in the handful above it.

    ``preferred_port`` is already in ``MIN_PORT..MAX_PORT`` (``ServerSettings``
    parses it there), and the retry window is clipped to MAX_PORT so walking
    off the top of the range refuses with the same sentence as an occupied
    range rather than with bind(2)'s OverflowError.
    """
    last_candidate_port = min(preferred_port + _PORT_RETRY_ATTEMPTS - 1, MAX_PORT)
    for candidate_port in range(preferred_port, last_candidate_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe_socket:
            try:
                probe_socket.bind((host, candidate_port))
            except OSError:
                continue
        return candidate_port
    raise ServerStartupError(
        f"no free port between {preferred_port} and {last_candidate_port} on {host}"
    )
