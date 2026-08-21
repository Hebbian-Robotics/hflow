"""AirflowClient against a stub HTTP server (no Docker, no Airflow)."""

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar

import pytest

from hflow.runtime import AirflowClient, AirflowClientError


class _StubAirflowHandler(BaseHTTPRequestHandler):
    """Just enough of Airflow's surface: token, health, dagRuns."""

    issued_tokens: ClassVar[list[str]] = []
    requests_seen: ClassVar[list[tuple[str, str, dict[str, Any] | None, str | None]]] = []
    healthy: ClassVar[bool] = True
    expire_first_token: ClassVar[bool] = False

    def _read_json(self) -> dict[str, Any] | None:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return None
        return json.loads(self.rfile.read(length))

    def _respond(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _record(self, payload: dict[str, Any] | None) -> str | None:
        authorization = self.headers.get("Authorization")
        type(self).requests_seen.append((self.command, self.path, payload, authorization))
        return authorization

    def do_POST(self) -> None:
        payload = self._read_json()
        authorization = self._record(payload)
        if self.path == "/auth/token":
            assert payload is not None
            if payload.get("password") != "right-password":
                self._respond(401, {"detail": "bad credentials"})
                return
            token = f"token-{len(type(self).issued_tokens)}"
            type(self).issued_tokens.append(token)
            self._respond(201, {"access_token": token})
            return
        if self.path.startswith("/api/v2/dags/") and self.path.endswith("/dagRuns"):
            if not self._bearer_ok(authorization):
                self._respond(401, {"detail": "expired"})
                return
            requested_run_id = (payload or {}).get("dag_run_id")
            if requested_run_id == "already-exists" or (payload or {}).get("conf", {}).get(
                "force_conflict"
            ):
                self._respond(409, {"detail": "dag run already exists"})
                return
            self._respond(200, {"dag_run_id": requested_run_id or "manual__1", "state": "queued"})
            return
        self._respond(404, {"detail": self.path})

    def do_GET(self) -> None:
        authorization = self._record(None)
        if "/dagRuns/" in self.path:
            if not self._bearer_ok(authorization):
                self._respond(401, {"detail": "expired"})
                return
            existing_run_id = self.path.rsplit("/", 1)[-1]
            self._respond(200, {"dag_run_id": existing_run_id, "state": "running"})
            return
        if self.path == "/api/v2/monitor/health":
            status = "healthy" if type(self).healthy else "unhealthy"
            self._respond(
                200,  # always 200: the body is the signal
                {
                    "metadatabase": {"status": status},
                    "scheduler": {"status": status, "latest_scheduler_heartbeat": "now"},
                    "triggerer": {"status": None},
                    "dag_processor": {"status": status},
                },
            )
            return
        self._respond(404, {"detail": self.path})

    def _bearer_ok(self, authorization: str | None) -> bool:
        if authorization is None or not authorization.startswith("Bearer "):
            return False
        token = authorization.removeprefix("Bearer ")
        if type(self).expire_first_token and token == type(self).issued_tokens[0]:
            return False  # simulate an expired first token
        return token in type(self).issued_tokens

    def log_message(self, format: str, *args: Any) -> None:
        pass


def _reset_stub_airflow_state() -> None:
    _StubAirflowHandler.issued_tokens = []
    _StubAirflowHandler.requests_seen = []
    _StubAirflowHandler.healthy = True
    _StubAirflowHandler.expire_first_token = False


@pytest.fixture(scope="module")
def stub_server_base_url() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubAirflowHandler)
    server_thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.01},
        daemon=True,
    )
    server_thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()


@pytest.fixture()
def stub_server(stub_server_base_url: str) -> str:
    _reset_stub_airflow_state()
    return stub_server_base_url


def test_trigger_fetches_token_once_and_sends_bearer(stub_server: str) -> None:
    client = AirflowClient(stub_server, "airflow", "right-password")
    first = client.trigger_dag_run("pipeline_ingest", conf={"uris": ["a.mcap"]})
    second = client.ingest("pipeline_ingest", ["b.mcap"])
    assert first["state"] == "queued" and second["state"] == "queued"
    assert len(_StubAirflowHandler.issued_tokens) == 1  # token cached across calls

    trigger_requests = [
        entry for entry in _StubAirflowHandler.requests_seen if entry[1].endswith("/dagRuns")
    ]
    method, path, payload, authorization = trigger_requests[0]
    assert (method, path) == ("POST", "/api/v2/dags/pipeline_ingest/dagRuns")
    assert payload == {"logical_date": None, "conf": {"uris": ["a.mcap"]}}
    assert authorization == "Bearer token-0"


def test_expired_token_is_refreshed_once(stub_server: str) -> None:
    _StubAirflowHandler.expire_first_token = True
    client = AirflowClient(stub_server, "airflow", "right-password")
    client._token = None  # force initial fetch
    result = client.trigger_dag_run("pipeline_ingest")
    assert result["state"] == "queued"
    assert len(_StubAirflowHandler.issued_tokens) == 2  # refreshed exactly once


def test_caller_supplied_dag_run_id_makes_retries_idempotent(stub_server: str) -> None:
    client = AirflowClient(stub_server, "airflow", "right-password")
    created = client.ingest("pipeline_ingest", ["a.mcap"], dag_run_id="my-run-1")
    assert created == {"dag_run_id": "my-run-1", "state": "queued"}

    # A retry whose id already exists gets the EXISTING run back, not a 409.
    existing = client.trigger_dag_run("pipeline_ingest", dag_run_id="already-exists")
    assert existing == {"dag_run_id": "already-exists", "state": "running"}


def test_conflict_without_a_dag_run_id_still_raises(stub_server: str) -> None:
    # Without a caller id there is nothing to idempotently return; the 409
    # must surface.
    client = AirflowClient(stub_server, "airflow", "right-password")
    with pytest.raises(AirflowClientError) as error_info:
        client.trigger_dag_run("pipeline_ingest", conf={"force_conflict": True})
    assert error_info.value.status == 409


def test_bad_credentials_surface_clearly(stub_server: str) -> None:
    client = AirflowClient(stub_server, "airflow", "wrong-password")
    with pytest.raises(AirflowClientError) as error_info:
        client.trigger_dag_run("pipeline_ingest")
    assert error_info.value.status == 401


def test_health_parses_body_not_status(stub_server: str) -> None:
    client = AirflowClient(stub_server, "airflow", "right-password")
    assert client.health().healthy

    _StubAirflowHandler.healthy = False
    unhealthy = client.health()  # still HTTP 200: the body is the signal
    assert not unhealthy.healthy
    assert "scheduler=unhealthy" in unhealthy.summary()


def test_wait_until_healthy_times_out_with_last_status(stub_server: str) -> None:
    class InstantlyAdvancingClock:
        def __init__(self) -> None:
            self.current_time_s = 0.0

        def monotonic(self) -> float:
            return self.current_time_s

        def sleep(self, duration_s: float) -> None:
            self.current_time_s += duration_s

    _StubAirflowHandler.healthy = False
    client = AirflowClient(stub_server, "airflow", "right-password")
    instantly_advancing_clock = InstantlyAdvancingClock()
    with (
        pytest.MonkeyPatch.context() as monkeypatch,
        pytest.raises(TimeoutError, match="scheduler=unhealthy"),
    ):
        monkeypatch.setattr("hflow.runtime._client.time", instantly_advancing_clock)
        client.wait_until_healthy(timeout_s=0.3, poll_interval_s=10.0)
    assert instantly_advancing_clock.current_time_s == pytest.approx(0.3)
