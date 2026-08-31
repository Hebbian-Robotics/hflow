"""Remote runtime endpoints: address a workspace by URL, not a bundle dir.

The Compose flow reconstructs everything (URL, credentials, dag id) from a
local bundle directory; a hosted workspace has no such directory -- the
customer is handed an Airflow API base URL, a dag id, and a credential by
whoever operates the workspace. This module resolves that triple from flags
plus the environment, so ``hflow ingest``/``hflow status`` drive a remote
workspace exactly like a local one.

Credentials come from the environment only, never argv: command lines leak
into process listings and shell history.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

from hflow.runtime._client import (
    AirflowAuth,
    AirflowClient,
    AirflowClientError,
    BearerToken,
    PasswordCredentials,
)

AIRFLOW_URL_ENVIRONMENT_VARIABLE = "HFLOW_AIRFLOW_URL"
AIRFLOW_DAG_ID_ENVIRONMENT_VARIABLE = "HFLOW_AIRFLOW_DAG_ID"
AIRFLOW_TOKEN_ENVIRONMENT_VARIABLE = "HFLOW_AIRFLOW_TOKEN"
AIRFLOW_USERNAME_ENVIRONMENT_VARIABLE = "HFLOW_AIRFLOW_USERNAME"
AIRFLOW_PASSWORD_ENVIRONMENT_VARIABLE = "HFLOW_AIRFLOW_PASSWORD"


@dataclass(frozen=True)
class RemoteRuntimeEndpoint:
    """One addressable ingest runtime: where, which DAG, and how to auth."""

    base_url: str
    dag_id: str
    auth: AirflowAuth


def resolve_remote_endpoint(
    *,
    airflow_url: str | None = None,
    dag_id: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> RemoteRuntimeEndpoint | None:
    """Resolve a remote endpoint from explicit values plus the environment.

    Returns ``None`` when no URL was supplied anywhere -- the caller falls
    back to the local bundle flow. When a URL exists, an incomplete
    resolution (no dag id, no credentials) is a loud ``ValueError`` naming
    exactly what to set.
    """
    environment = os.environ if environ is None else environ
    base_url = airflow_url or environment.get(AIRFLOW_URL_ENVIRONMENT_VARIABLE)
    if not base_url:
        return None
    if urlsplit(base_url).scheme not in ("http", "https"):
        # Parse at the boundary: a scheme-less URL would otherwise surface
        # as a raw urllib ValueError traceback deep inside the first request.
        raise ValueError(
            f"remote Airflow URL {base_url!r} needs an http:// or https:// scheme "
            f"(from --airflow-url or {AIRFLOW_URL_ENVIRONMENT_VARIABLE})"
        )

    resolved_dag_id = dag_id or environment.get(AIRFLOW_DAG_ID_ENVIRONMENT_VARIABLE)
    if not resolved_dag_id:
        raise ValueError(
            f"remote Airflow at {base_url} needs a dag id: pass --dag-id or export "
            f"{AIRFLOW_DAG_ID_ENVIRONMENT_VARIABLE}"
        )

    token = environment.get(AIRFLOW_TOKEN_ENVIRONMENT_VARIABLE)
    username = environment.get(AIRFLOW_USERNAME_ENVIRONMENT_VARIABLE)
    password = environment.get(AIRFLOW_PASSWORD_ENVIRONMENT_VARIABLE)
    auth: AirflowAuth
    if token:
        auth = BearerToken(token=token)
    elif username and password:
        auth = PasswordCredentials(username=username, password=password)
    else:
        raise ValueError(
            f"remote Airflow at {base_url} needs credentials from the environment: export "
            f"{AIRFLOW_TOKEN_ENVIRONMENT_VARIABLE}, or "
            f"{AIRFLOW_USERNAME_ENVIRONMENT_VARIABLE} and "
            f"{AIRFLOW_PASSWORD_ENVIRONMENT_VARIABLE}"
        )
    return RemoteRuntimeEndpoint(base_url=base_url, dag_id=resolved_dag_id, auth=auth)


def client_for_endpoint(endpoint: RemoteRuntimeEndpoint) -> AirflowClient:
    """An :class:`AirflowClient` speaking to this endpoint."""
    return AirflowClient(endpoint.base_url, auth=endpoint.auth)


def describe_remote_status(endpoint: RemoteRuntimeEndpoint, *, run_limit: int = 5) -> str:
    """Health, DAG registration, and recent run states over the REST API.

    The backend-neutral half of ``hflow status``: no docker, no local files
    -- everything here works against any reachable Airflow, hosted or local.
    """
    client = client_for_endpoint(endpoint)
    lines = [f"endpoint: {endpoint.base_url}", f"dag:      {endpoint.dag_id}"]
    try:
        health = client.health()
    except AirflowClientError as error:
        lines.append(f"health:   unreachable ({error})")
        return "\n".join(lines)
    overall = "healthy" if health.healthy else "UNHEALTHY"
    lines.append(f"health:   {overall} ({health.summary()})")
    try:
        client.dag(endpoint.dag_id)
        # order_by="-id": the server truncates to `limit`, so it must sort
        # newest-first or a busy DAG would show only its oldest history.
        recent_runs = client.dag_runs(endpoint.dag_id, limit=run_limit, order_by="-id")
    except AirflowClientError as error:
        lines.append(f"dag:      unavailable ({error})")
        return "\n".join(lines)
    if not recent_runs:
        lines.append("runs:     none recorded")
    else:
        for run in reversed(recent_runs):  # print oldest-to-newest of the window
            run_id = run.dag_run_id or "<unknown>"
            run_state = run.state or "<unknown>"
            lines.append(f"run:      {run_id} [{run_state}]")
    return "\n".join(lines)
