"""Airflow REST API v2 client (stdlib-only; the SDK stays dependency-light).

Facts this encodes (references/airflow3-notes.md):

- Every ``/api/v2`` request carries ``Authorization: Bearer <JWT>``; the token
  comes from ``POST {base}/auth/token`` (NOT under ``/api/v2``) with
  ``{"username", "password"}`` and expires -- a 401 means refresh once and
  retry.
- ``GET /api/v2/monitor/health`` returns HTTP 200 even when unhealthy: the
  BODY is the signal (per-component ``status`` of healthy/unhealthy/null).
  It is also unauthenticated (the official compose healthcheck curls it).
- Dag runs trigger via ``POST /api/v2/dags/{dag_id}/dagRuns`` with a required
  (but nullable) ``logical_date``.
"""

import http.client
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class AirflowClientError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass(frozen=True)
class AirflowHealth:
    """Parsed ``/api/v2/monitor/health`` body."""

    components: dict[str, str | None]

    @property
    def healthy(self) -> bool:
        # metadatabase and scheduler must be healthy; triggerer/dag_processor
        # may legitimately be absent (null) in minimal deployments.
        required = ("metadatabase", "scheduler", "dag_processor")
        return all(self.components.get(name) == "healthy" for name in required)

    def summary(self) -> str:
        parts = [f"{name}={status or 'absent'}" for name, status in sorted(self.components.items())]
        return ", ".join(parts)


class AirflowClient:
    """Minimal typed client for the deployment endpoints the SDK needs."""

    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        *,
        request_timeout_s: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._username = username
        self._password = password
        self._request_timeout_s = request_timeout_s
        self._token: str | None = None

    def _http_json(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        *,
        bearer_token: str | None = None,
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(url, data=body, method=method)
        request.add_header("Content-Type", "application/json")
        request.add_header("Accept", "application/json")
        if bearer_token is not None:
            request.add_header("Authorization", f"Bearer {bearer_token}")
        try:
            with urllib.request.urlopen(request, timeout=self._request_timeout_s) as response:
                response_text = response.read().decode()
        except urllib.error.HTTPError as error:
            error_body = error.read().decode(errors="replace")
            # Airflow puts the useful part ("detail") in the body; surface a
            # truncated copy so failures are diagnosable from the message.
            body_excerpt = " ".join(error_body.split())[:200]
            raise AirflowClientError(
                f"{method} {url} failed with HTTP {error.code}"
                + (f": {body_excerpt}" if body_excerpt else ""),
                status=error.code,
                body=error_body,
            ) from error
        except urllib.error.URLError as error:
            raise AirflowClientError(f"{method} {url} unreachable: {error.reason}") from error
        except (OSError, http.client.HTTPException) as error:
            # Connection-level failures outside urllib's wrapping (e.g. a
            # boot-time ConnectionResetError from docker-proxy accepting the
            # published port before the api-server listens) must also become
            # AirflowClientError, or wait_until_healthy cannot retry them.
            raise AirflowClientError(f"{method} {url} failed: {error!r}") from error
        if not response_text:
            return {}
        parsed = json.loads(response_text)
        if not isinstance(parsed, dict):
            raise AirflowClientError(f"{method} {url} returned non-object JSON: {parsed!r}")
        return parsed

    def _fetch_token(self) -> str:
        response = self._http_json(
            "POST",
            f"{self.base_url}/auth/token",
            {"username": self._username, "password": self._password},
        )
        token = response.get("access_token")
        if not isinstance(token, str) or not token:
            raise AirflowClientError(f"/auth/token returned no access_token: {response!r}")
        return token

    def _authenticated(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if self._token is None:
            self._token = self._fetch_token()
        url = f"{self.base_url}{path}"
        try:
            return self._http_json(method, url, payload, bearer_token=self._token)
        except AirflowClientError as error:
            if error.status != 401:
                raise
            # Expired token: refresh exactly once and retry.
            self._token = self._fetch_token()
            return self._http_json(method, url, payload, bearer_token=self._token)

    def health(self) -> AirflowHealth:
        response = self._http_json("GET", f"{self.base_url}/api/v2/monitor/health")
        components: dict[str, str | None] = {}
        for component_name, component_body in response.items():
            if isinstance(component_body, dict):
                status = component_body.get("status")
                components[component_name] = status if isinstance(status, str) else None
        return AirflowHealth(components=components)

    def wait_until_healthy(
        self,
        *,
        timeout_s: float = 300.0,
        poll_interval_s: float = 3.0,
        on_poll: Callable[[str], None] | None = None,
    ) -> AirflowHealth:
        """Poll health until every required component reports healthy.

        Connection errors while services boot are expected and retried.
        ``on_poll`` (if given) receives the latest status summary after each
        unsuccessful poll, so callers can narrate a long wait without owning
        a second copy of this loop.
        """
        deadline = time.monotonic() + timeout_s
        last_status = "unreachable"
        while time.monotonic() < deadline:
            try:
                health = self.health()
            except AirflowClientError as error:
                last_status = str(error)
            else:
                if health.healthy:
                    return health
                last_status = health.summary()
            if on_poll is not None:
                on_poll(last_status)
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                break
            time.sleep(min(poll_interval_s, remaining_s))
        raise TimeoutError(
            f"Airflow at {self.base_url} not healthy after {timeout_s:.0f}s (last: {last_status})"
        )

    def trigger_dag_run(
        self,
        dag_id: str,
        conf: dict[str, Any] | None = None,
        *,
        dag_run_id: str | None = None,
    ) -> dict[str, Any]:
        """Trigger a run; pass ``dag_run_id`` to make retries idempotent.

        A caller-generated id decided BEFORE the POST means a retry after a
        dropped response cannot double-trigger: Airflow answers 409 for the
        existing id, and this method returns that run instead of failing.
        """
        payload: dict[str, Any] = {"logical_date": None, "conf": conf or {}}
        if dag_run_id is not None:
            payload["dag_run_id"] = dag_run_id
        try:
            return self._authenticated("POST", f"/api/v2/dags/{dag_id}/dagRuns", payload)
        except AirflowClientError as error:
            if error.status == 409 and dag_run_id is not None:
                return self.dag_run(dag_id, dag_run_id)
            raise

    def dag(self, dag_id: str) -> dict[str, Any]:
        """The DAG's details; 404s (as AirflowClientError) until it registers."""
        return self._authenticated("GET", f"/api/v2/dags/{dag_id}")

    def dag_run(self, dag_id: str, dag_run_id: str) -> dict[str, Any]:
        return self._authenticated("GET", f"/api/v2/dags/{dag_id}/dagRuns/{dag_run_id}")

    def dag_runs(self, dag_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        """The DAG's runs (up to ``limit``), as the API returns them."""
        response = self._authenticated("GET", f"/api/v2/dags/{dag_id}/dagRuns?limit={limit}")
        runs = response.get("dag_runs")
        return runs if isinstance(runs, list) else []

    def unpause_dag(self, dag_id: str) -> dict[str, Any]:
        return self._authenticated("PATCH", f"/api/v2/dags/{dag_id}", {"is_paused": False})

    def ingest(
        self,
        dag_id: str,
        uris: list[str],
        *,
        profile: str = "full",
        online: bool = False,
        dag_run_id: str | None = None,
    ) -> dict[str, Any]:
        """Trigger the MASTER ingest DAG over ``uris`` (the SDK/CLI entry point).

        ``profile`` names a run profile (the master validates it against the
        vocabulary baked from ``hflow.steps.RUN_PROFILES`` and triggers
        only the enabled stage sub-DAGs). ``online`` selects Figure 4's
        latency-first trigger lane -- the sub-DAGs process the uris as one
        immediate batch, no bin-packing, no stagger -- instead of the default
        staggered batch lane. Supply ``dag_run_id`` when the caller may retry
        (see :meth:`trigger_dag_run` for the idempotency contract).
        """
        conf = {
            "uris": uris,
            "profile": profile,
            "mode": "online" if online else "batch",
        }
        return self.trigger_dag_run(dag_id, conf=conf, dag_run_id=dag_run_id)
