"""The runs monitor API: address the workspace's ingest runtime, proxy Airflow.

Addressing mirrors the CLI's ``_resolve_bundle_dir``: a local Compose bundle
at ``<data_root>/runtime`` (skipped for bucket data roots), then the
``./runtime`` fallback, else a remote endpoint resolved from the
``HFLOW_AIRFLOW_*`` environment via :func:`hflow.runtime.resolve_remote_endpoint`.
Resolution happens lazily per request -- the stack may come up (or go away)
after the UI started -- and is cached briefly per launch.

Two rules hold at this boundary:

- The browser NEVER receives credentials: the bundle's admin password and any
  remote token stay inside :class:`hflow.runtime.AirflowClient`; this server
  proxies every Airflow call. The only URL it exposes is the deep-link base
  the operator already knows (the bundle's own recorded api-server address).
- A missing or unreachable runtime is an ANSWER, never a traceback:
  ``/runtime/status`` reports ``available: false`` with the reason, and the
  other endpoints refuse with a clear 4xx/502 detail.
"""

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from posixpath import normpath
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from hflow.runtime import (
    AirflowClient,
    AirflowClientError,
    client_for_bundle,
    client_for_endpoint,
    load_bundle,
    resolve_remote_endpoint,
    sub_dag_id_for_stage,
)
from hflow.steps import RUN_PROFILES, IngestMode, Stage
from hflow.storage import is_bucket_url
from hflow.workspace import RUNTIME_BUNDLE_DIRECTORY_NAME
from hflow_ui._settings import UiSettings

# Mirrors hflow.runtime._endpoint's variable name (a documented public
# contract); restated here rather than imported from that private module.
AIRFLOW_URL_ENVIRONMENT_VARIABLE = "HFLOW_AIRFLOW_URL"

# How long one resolution (bundle files read, client built) is reused before
# the next request re-probes -- long enough to spare a busy Runs page the
# filesystem walk, short enough that `hflow up` shows up within seconds.
RESOLUTION_CACHE_TTL_S = 5.0

# The health components /runtime/status reports, per AirflowHealth's contract.
_HEALTH_COMPONENT_NAMES = ("metadatabase", "scheduler", "triggerer", "dag_processor")

_RECENT_STAGE_RUN_LIMIT = 5

_LOGGER = logging.getLogger("hflow_ui.runtime")


def _client_error_reason(error: AirflowClientError) -> str:
    """A stable machine-readable classification of an Airflow call failure."""
    if error.status in (401, 403):
        return "unauthorized"
    if error.status is not None:
        return "http_error"
    return "unreachable"


def client_error_detail(error: AirflowClientError, *, source: str) -> str:
    """A browser-safe detail for an Airflow call failure (shared with _graph).

    A local bundle's api-server address is one the operator already has, so
    its verbatim message (which embeds that URL) is fine. A REMOTE runtime's
    base URL is deliberately withheld on the success path, and the verbatim
    message embeds that URL plus an excerpt of the upstream response body --
    so a remote failure returns only a generic detail with a stable reason
    code, and the full error is logged server-side for the operator.
    """
    if source == "bundle":
        return str(error)
    reason = _client_error_reason(error)
    _LOGGER.warning("remote Airflow call failed (reason=%s): %s", reason, error)
    if error.status is not None:
        return (
            f"the remote ingest runtime returned an error (reason: {reason}, status {error.status})"
        )
    return f"the remote ingest runtime is not reachable (reason: {reason})"


@dataclass(frozen=True)
class ResolvedRuntime:
    """One addressable ingest runtime: the client, its DAG, and its shape."""

    client: AirflowClient
    dag_id: str
    source: Literal["bundle", "remote"]
    airflow_web_url: str | None
    # (stage name, sub-DAG id) in stage-graph order; None for remote runtimes
    # (only the bundle manifest records the stage sub-DAG ids).
    stage_dag_ids: tuple[tuple[str, str], ...] | None


@dataclass(frozen=True)
class RuntimeUnavailable:
    """No addressable runtime, and exactly why."""

    detail: str


RuntimeResolution = ResolvedRuntime | RuntimeUnavailable


class IngestRequest(BaseModel):
    uris: list[str] = Field(min_length=1)
    profile: str = "full"
    mode: str = IngestMode.BATCH.value
    batch_count: int | None = Field(default=None, ge=1)


def find_bundle_directory(data_root: str) -> Path | None:
    """The rendered local bundle this workspace addresses, if one exists.

    Mirrors the CLI's ``_resolve_bundle_dir`` probing: ``<data_root>/runtime``
    first (bucket data roots have no local root, so only the fallback
    applies), then ``./runtime``; a candidate counts only when its
    ``docker-compose.yaml`` exists. Unlike the CLI there is no primary-
    candidate fallback -- "no bundle anywhere" is a real answer here.
    """
    candidates = [Path(RUNTIME_BUNDLE_DIRECTORY_NAME)]
    if not is_bucket_url(data_root):
        candidates.insert(0, Path(data_root) / RUNTIME_BUNDLE_DIRECTORY_NAME)
    for candidate in candidates:
        if (candidate / "docker-compose.yaml").is_file():
            return candidate
    return None


def runtime_configured(data_root: str) -> bool:
    """Whether a runtime is ADDRESSED (bundle dir or remote URL) -- the
    /api/v1/config capability. Configured, not necessarily reachable."""
    if find_bundle_directory(data_root) is not None:
        return True
    return bool(os.environ.get(AIRFLOW_URL_ENVIRONMENT_VARIABLE))


def local_bundle_web_url(data_root: str) -> str | None:
    """The addressed local bundle's Airflow URL for browser deep links.

    Derived from what the bundle itself records (``load_bundle`` reads the
    preserved ``.env``'s API_PORT), so it is the address the operator's own
    ``hflow up`` printed; ``None`` when no loadable bundle is addressed.
    """
    bundle_directory = find_bundle_directory(data_root)
    if bundle_directory is None:
        return None
    try:
        return load_bundle(bundle_directory).api_base_url
    except (FileNotFoundError, ValueError):
        return None


def _stage_dag_ids(master_dag_id: str) -> tuple[tuple[str, str], ...]:
    """(stage name, sub-DAG id) pairs in stage-graph order.

    The sub-DAG ids derive from the master's id the same way the renderer
    minted them (:func:`hflow.runtime.sub_dag_id_for_stage`), so there is one
    owner of that mapping and no second bundle-manifest parser to drift from
    the library's version-guarded :func:`hflow.runtime.load_bundle`.
    """
    return tuple((stage.value, sub_dag_id_for_stage(master_dag_id, stage)) for stage in Stage)


def resolve_runtime(data_root: str) -> RuntimeResolution:
    """One resolution pass: local bundle first, else the remote environment.

    Every failure mode (no bundle anywhere and no URL exported; a half-formed
    bundle; a URL exported without dag id or credentials) becomes a
    :class:`RuntimeUnavailable` whose detail names the fix -- never an
    exception that would surface as a 500.
    """
    bundle_directory = find_bundle_directory(data_root)
    if bundle_directory is not None:
        try:
            bundle_paths = load_bundle(bundle_directory)
        except (FileNotFoundError, ValueError) as error:
            return RuntimeUnavailable(detail=str(error))
        return ResolvedRuntime(
            client=client_for_bundle(bundle_paths),
            dag_id=bundle_paths.dag_id,
            source="bundle",
            airflow_web_url=bundle_paths.api_base_url,
            stage_dag_ids=_stage_dag_ids(bundle_paths.dag_id),
        )
    try:
        endpoint = resolve_remote_endpoint()
    except ValueError as error:
        # A URL is exported but the resolution is incomplete; the message
        # names exactly which HFLOW_AIRFLOW_* variable to set.
        return RuntimeUnavailable(detail=str(error))
    if endpoint is None:
        return RuntimeUnavailable(
            detail=(
                "no ingest runtime addressed: no rendered bundle at "
                f"{Path(data_root) / RUNTIME_BUNDLE_DIRECTORY_NAME} or ./runtime "
                f"(run `hflow up`), and {AIRFLOW_URL_ENVIRONMENT_VARIABLE} is not set"
            )
        )
    return ResolvedRuntime(
        client=client_for_endpoint(endpoint),
        dag_id=endpoint.dag_id,
        source="remote",
        # Only a bundle records its own web address; guessing that a remote
        # API base URL also serves the web UI would not be honest.
        airflow_web_url=None,
        stage_dag_ids=None,
    )


class RuntimeResolver:
    """Per-launch cache around :func:`resolve_runtime` (see the TTL note)."""

    def __init__(self, data_root: str) -> None:
        self._data_root = data_root
        self._cached_resolution: RuntimeResolution | None = None
        self._expires_at_monotonic = 0.0

    def resolve(self) -> RuntimeResolution:
        now_monotonic = time.monotonic()
        if self._cached_resolution is None or now_monotonic >= self._expires_at_monotonic:
            self._cached_resolution = resolve_runtime(self._data_root)
            self._expires_at_monotonic = now_monotonic + RESOLUTION_CACHE_TTL_S
        return self._cached_resolution


def _unavailable_status_payload(
    detail: str,
    *,
    source: str | None = None,
    airflow_web_url: str | None = None,
    dag_id: str | None = None,
) -> dict[str, object]:
    return {
        "available": False,
        "detail": detail,
        "source": source,
        "airflow_web_url": airflow_web_url,
        "dag_id": dag_id,
        "registered": None,
        "health": None,
    }


def _run_summary(run: dict[str, Any]) -> dict[str, object]:
    """One master dag run reduced to the fields the Runs page shows.

    The full ``conf`` rides along (it is the trigger's own input); everything
    else Airflow returns stays server-side.
    """
    conf = run.get("conf")
    return {
        "dag_run_id": run.get("dag_run_id"),
        "state": run.get("state"),
        "logical_date": run.get("logical_date"),
        "start_date": run.get("start_date"),
        "end_date": run.get("end_date"),
        "conf": conf if isinstance(conf, dict) else {},
    }


def _stage_run_summary(run: dict[str, Any]) -> dict[str, object]:
    return {
        "dag_run_id": run.get("dag_run_id"),
        "state": run.get("state"),
        "start_date": run.get("start_date"),
        "end_date": run.get("end_date"),
    }


def create_runtime_router(settings: UiSettings, resolver: RuntimeResolver) -> APIRouter:
    """Every M2 runs-monitor route, closed over one launch's settings.

    The resolver is passed in (rather than built here) so the run-graph routes
    in ``_graph`` share one addressing cache with this router.
    """
    router = APIRouter(prefix="/api/v1")

    def resolved_or_refuse() -> ResolvedRuntime:
        resolution = resolver.resolve()
        if isinstance(resolution, RuntimeUnavailable):
            # 409: the workspace's runtime state conflicts with the request,
            # same mapping as an unconfigured pipeline.
            raise HTTPException(status_code=409, detail=resolution.detail)
        return resolution

    def refuse_when_read_only() -> None:
        if settings.read_only:
            raise HTTPException(
                status_code=403,
                detail="this workspace UI is running read-only; triggering ingest runs is disabled",
            )

    @router.get("/runtime/status")
    def read_runtime_status() -> JSONResponse:
        resolution = resolver.resolve()
        if isinstance(resolution, RuntimeUnavailable):
            return JSONResponse(_unavailable_status_payload(resolution.detail))
        try:
            health = resolution.client.health()
        except AirflowClientError as error:
            # Addressed but not answering (typical between `hflow up` runs):
            # still an available:false ANSWER, with the addressing facts.
            return JSONResponse(
                _unavailable_status_payload(
                    client_error_detail(error, source=resolution.source),
                    source=resolution.source,
                    airflow_web_url=resolution.airflow_web_url,
                    dag_id=resolution.dag_id,
                )
            )
        registered: bool | None
        try:
            resolution.client.dag(resolution.dag_id)
            registered = True
        except AirflowClientError as error:
            # 404 is the definitive "not registered (yet)"; anything else
            # (auth, transient) leaves registration unknown, not false.
            registered = False if error.status == 404 else None
        return JSONResponse(
            {
                "available": True,
                "detail": None,
                "source": resolution.source,
                "airflow_web_url": resolution.airflow_web_url,
                "dag_id": resolution.dag_id,
                "registered": registered,
                "health": {
                    component_name: health.components.get(component_name)
                    for component_name in _HEALTH_COMPONENT_NAMES
                },
            }
        )

    @router.get("/runtime/runs")
    def list_runtime_runs(
        limit: int = Query(default=25, ge=1, le=100),
    ) -> JSONResponse:
        runtime = resolved_or_refuse()
        try:
            # order_by="-id": Airflow truncates to `limit` in id order, so
            # newest-first is the only ordering that shows recent activity.
            master_runs = runtime.client.dag_runs(runtime.dag_id, limit=limit, order_by="-id")
        except AirflowClientError as error:
            raise HTTPException(
                status_code=502, detail=client_error_detail(error, source=runtime.source)
            ) from error
        stages: list[dict[str, object]] | None = None
        if runtime.stage_dag_ids is not None:
            stages = []
            for stage_name, stage_dag_id in runtime.stage_dag_ids:
                try:
                    recent_runs = runtime.client.dag_runs(
                        stage_dag_id, limit=_RECENT_STAGE_RUN_LIMIT, order_by="-id"
                    )
                except AirflowClientError:
                    # A stage sub-DAG that has not registered (or errored) is
                    # an empty strip, not a failed page.
                    recent_runs = []
                stages.append(
                    {
                        "stage": stage_name,
                        "dag_id": stage_dag_id,
                        "recent": [_stage_run_summary(run) for run in recent_runs],
                    }
                )
        return JSONResponse({"runs": [_run_summary(run) for run in master_runs], "stages": stages})

    @router.post("/runtime/ingest")
    def trigger_ingest(request: IngestRequest) -> JSONResponse:
        refuse_when_read_only()
        uris = [uri.strip() for uri in request.uris]
        if any(not uri for uri in uris):
            raise HTTPException(status_code=400, detail="every uri must be a non-empty string")
        # URIs resolve against the runtime's data root; absolute host paths and
        # ../ escapes cannot work there, so refuse them before triggering --
        # the same guard `hflow ingest` enforces (src/hflow/cli.py).
        for uri in uris:
            if uri.startswith("/") or normpath(uri).startswith(".."):
                raise HTTPException(
                    status_code=400,
                    detail=f"{uri!r} is not relative to the data root -- URIs are resolved "
                    "against the runtime's configured data root (e.g. "
                    "`episodes-in/run_0001.mcap`)",
                )
        if request.profile not in RUN_PROFILES:
            raise HTTPException(
                status_code=400,
                detail=f"unknown run profile {request.profile!r}; "
                f"valid profiles: {sorted(RUN_PROFILES)}",
            )
        try:
            mode = IngestMode(request.mode)
        except ValueError as error:
            raise HTTPException(
                status_code=400,
                detail=f"unknown ingest mode {request.mode!r}; "
                f"valid modes: {[known_mode.value for known_mode in IngestMode]}",
            ) from error
        runtime = resolved_or_refuse()
        try:
            # AirflowClient.ingest owns the trigger conf's shape (uris/profile/
            # mode/batch_count) for every caller -- CLI, UI, control plane --
            # so the UI never rebuilds the dict itself.
            trigger_response = runtime.client.ingest(
                runtime.dag_id,
                uris,
                profile=request.profile,
                online=mode is IngestMode.ONLINE,
                batch_count=request.batch_count,
            )
        except AirflowClientError as error:
            raise HTTPException(
                status_code=502, detail=client_error_detail(error, source=runtime.source)
            ) from error
        return JSONResponse(
            {
                "dag_run_id": trigger_response.get("dag_run_id"),
                "state": trigger_response.get("state"),
            }
        )

    return router
