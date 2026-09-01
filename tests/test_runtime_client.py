"""AirflowClient against a stub HTTP server (no Docker, no Airflow)."""

import json
import socket
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar

import pytest

from hflow.runtime import (
    AirflowClient,
    AirflowClientError,
    BearerToken,
    PasswordCredentials,
    RemoteRuntimeEndpoint,
    describe_remote_status,
    resolve_remote_endpoint,
)


class _StubAirflowHandler(BaseHTTPRequestHandler):
    """Just enough of Airflow's surface: token, health, dagRuns."""

    issued_tokens: ClassVar[list[str]] = []
    requests_seen: ClassVar[list[tuple[str, str, dict[str, Any] | None, str | None]]] = []
    healthy: ClassVar[bool] = True
    expire_first_token: ClassVar[bool] = False
    health_response_body: ClassVar[bytes | None] = None

    def _read_json(self) -> dict[str, Any] | None:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return None
        return json.loads(self.rfile.read(length))

    def _respond(self, status: int, payload: dict[str, Any]) -> None:
        self._respond_bytes(status, json.dumps(payload).encode())

    def _respond_bytes(self, status: int, body: bytes) -> None:
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
        if self.path.endswith("/taskInstances"):
            if not self._bearer_ok(authorization):
                self._respond(401, {"detail": "expired"})
                return
            self._respond(200, {"task_instances": [{"task_id": "plan", "map_index": -1}]})
            return
        if "/dagRuns/" in self.path:
            if not self._bearer_ok(authorization):
                self._respond(401, {"detail": "expired"})
                return
            existing_run_id = self.path.rsplit("/", 1)[-1]
            self._respond(200, {"dag_run_id": existing_run_id, "state": "running"})
            return
        if "/dagRuns" in self.path:  # the run LIST (with or without ?limit=)
            if not self._bearer_ok(authorization):
                self._respond(401, {"detail": "expired"})
                return
            self._respond(200, {"dag_runs": [{"dag_run_id": "manual__1", "state": "success"}]})
            return
        if self.path.startswith("/api/v2/dags/"):
            if not self._bearer_ok(authorization):
                self._respond(401, {"detail": "expired"})
                return
            requested_dag_id = self.path.rsplit("/", 1)[-1]
            if requested_dag_id == "missing_dag":
                self._respond(404, {"detail": "DAG not found"})
                return
            self._respond(200, {"dag_id": requested_dag_id})
            return
        if self.path == "/api/v2/monitor/health":
            health_response_body = type(self).health_response_body
            if health_response_body is not None:
                self._respond_bytes(200, health_response_body)
                return
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
    _StubAirflowHandler.health_response_body = None


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


def test_per_run_endpoints_address_the_same_run(stub_server: str) -> None:
    """One run id, one URL rule -- whichever per-run endpoint asks for it.

    Airflow's own ids carry ':' and '+', and a caller-supplied idempotency id
    may carry '#', which urllib reads as a fragment and drops from the path.
    Left raw, the run detail and its task instances would describe two
    different runs (and the 409 retry in trigger_dag_run would 404).
    """
    client = AirflowClient(stub_server, "airflow", "right-password")
    dag_run_id = "manual__2026-08-22T03:06:55+00:00#retry-1"
    client.dag_run("pipeline_ingest", dag_run_id)
    client.task_instances("pipeline_ingest", dag_run_id)

    encoded_run = "manual__2026-08-22T03%3A06%3A55%2B00%3A00%23retry-1"
    per_run_paths = [
        path
        for method, path, _payload, _authorization in _StubAirflowHandler.requests_seen
        if method == "GET" and "/dagRuns/" in path
    ]
    assert per_run_paths == [
        f"/api/v2/dags/pipeline_ingest/dagRuns/{encoded_run}",
        f"/api/v2/dags/pipeline_ingest/dagRuns/{encoded_run}/taskInstances",
    ]


def test_conflict_without_a_dag_run_id_still_raises(stub_server: str) -> None:
    # Without a caller id there is nothing to idempotently return; the 409
    # must surface.
    client = AirflowClient(stub_server, "airflow", "right-password")
    with pytest.raises(AirflowClientError) as error_info:
        client.trigger_dag_run("pipeline_ingest", conf={"force_conflict": True})
    assert error_info.value.status == 409


def test_ingest_refuses_a_batch_count_the_run_could_not_honour(stub_server: str) -> None:
    """The conf's owner refuses it here, before a run exists to fail.

    ``plan_batches`` enforces ``>= 1`` inside the sync sub-DAG's plan task, so
    without this the SDK reports a triggered run and the operator's history
    collects a failure for a value the client could have named.
    """
    client = AirflowClient(stub_server, "airflow", "right-password")
    with pytest.raises(ValueError, match="batch_count must be >= 1, got 0"):
        client.ingest("pipeline_ingest", ["a.mcap"], batch_count=0)
    assert _StubAirflowHandler.requests_seen == []


def test_ingest_serializes_selected_step_names(stub_server: str) -> None:
    client = AirflowClient(stub_server, "airflow", "right-password")

    client.ingest(
        "pipeline_ingest",
        ["a.mcap"],
        step_names=("camera_integrity", "hand_activity"),
    )

    trigger_request = next(
        entry for entry in _StubAirflowHandler.requests_seen if entry[1].endswith("/dagRuns")
    )
    assert trigger_request[2] == {
        "logical_date": None,
        "conf": {
            "uris": ["a.mcap"],
            "profile": "full",
            "mode": "batch",
            "step_names": ["camera_integrity", "hand_activity"],
        },
    }


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


def test_malformed_success_response_raises_typed_client_error(stub_server: str) -> None:
    _StubAirflowHandler.health_response_body = (
        b"<html>\n  <body>proxy returned an invalid API response "
        + b"x" * 250
        + b" end-of-response</body>\n</html>"
    )
    url = f"{stub_server}/api/v2/monitor/health"
    client = AirflowClient(stub_server, "airflow", "right-password")

    with pytest.raises(AirflowClientError) as error_info:
        client.health()

    message = str(error_info.value)
    assert message.startswith(f"GET {url} returned malformed JSON: <html> <body>")
    assert "proxy returned an invalid API response" in message
    assert "end-of-response" not in message
    assert "\n" not in message


def test_invalid_utf8_success_response_raises_typed_client_error(stub_server: str) -> None:
    _StubAirflowHandler.health_response_body = (
        b'{"detail":"upstream returned invalid bytes \xff' + b"x" * 250 + b'end-of-response"}'
    )
    url = f"{stub_server}/api/v2/monitor/health"
    client = AirflowClient(stub_server, "airflow", "right-password")

    with pytest.raises(AirflowClientError) as error_info:
        client.health()

    message = str(error_info.value)
    assert message.startswith(f"GET {url} returned invalid UTF-8: ")
    assert "upstream returned invalid bytes" in message
    assert "end-of-response" not in message


def test_empty_success_response_still_returns_an_empty_object(stub_server: str) -> None:
    _StubAirflowHandler.health_response_body = b""
    client = AirflowClient(stub_server, "airflow", "right-password")

    assert client.health().components == {}


def test_non_object_success_response_is_still_refused(stub_server: str) -> None:
    _StubAirflowHandler.health_response_body = b'["healthy"]'
    client = AirflowClient(stub_server, "airflow", "right-password")

    with pytest.raises(AirflowClientError, match="returned non-object JSON"):
        client.health()


def test_bearer_token_auth_never_calls_the_token_endpoint(stub_server: str) -> None:
    """A pre-issued token (control-plane-minted) is used as-is: no username,
    no password, and no POST /auth/token."""
    _StubAirflowHandler.issued_tokens.append("pre-issued-token")
    client = AirflowClient(stub_server, auth=BearerToken("pre-issued-token"))
    result = client.trigger_dag_run("pipeline_ingest")
    assert result["state"] == "queued"
    token_endpoint_requests = [
        entry for entry in _StubAirflowHandler.requests_seen if entry[1] == "/auth/token"
    ]
    assert token_endpoint_requests == []


def test_rejected_bearer_token_fails_without_a_refresh_loop(stub_server: str) -> None:
    client = AirflowClient(stub_server, auth=BearerToken("never-issued"))
    with pytest.raises(AirflowClientError, match="expired or invalid"):
        client.trigger_dag_run("pipeline_ingest")
    # Exactly one request went out: retrying the same token bytes is useless.
    trigger_requests = [
        entry for entry in _StubAirflowHandler.requests_seen if entry[1].endswith("/dagRuns")
    ]
    assert len(trigger_requests) == 1


def test_client_requires_exactly_one_credential_form() -> None:
    with pytest.raises(ValueError, match="not both"):
        AirflowClient("http://x", "user", "password", auth=BearerToken("token"))
    with pytest.raises(ValueError, match="needs credentials"):
        AirflowClient("http://x")


class TestResolveRemoteEndpoint:
    def test_no_url_anywhere_means_local(self) -> None:
        assert resolve_remote_endpoint(environ={}) is None

    def test_flags_win_and_token_beats_password(self) -> None:
        endpoint = resolve_remote_endpoint(
            airflow_url="https://workspace.example.com",
            dag_id="kitchen_ingest",
            environ={
                "HFLOW_AIRFLOW_URL": "https://ignored.example.com",
                "HFLOW_AIRFLOW_TOKEN": "minted-token",
                "HFLOW_AIRFLOW_USERNAME": "also-ignored",
                "HFLOW_AIRFLOW_PASSWORD": "also-ignored",
            },
        )
        assert endpoint == RemoteRuntimeEndpoint(
            base_url="https://workspace.example.com",
            dag_id="kitchen_ingest",
            auth=BearerToken("minted-token"),
        )

    def test_environment_only_resolution_with_password_credentials(self) -> None:
        endpoint = resolve_remote_endpoint(
            environ={
                "HFLOW_AIRFLOW_URL": "https://workspace.example.com",
                "HFLOW_AIRFLOW_DAG_ID": "kitchen_ingest",
                "HFLOW_AIRFLOW_USERNAME": "svc",
                "HFLOW_AIRFLOW_PASSWORD": "secret",
            }
        )
        assert endpoint is not None
        assert endpoint.auth == PasswordCredentials(username="svc", password="secret")

    def test_missing_dag_id_and_missing_credentials_name_the_fix(self) -> None:
        with pytest.raises(ValueError, match="HFLOW_AIRFLOW_DAG_ID"):
            resolve_remote_endpoint(
                airflow_url="https://workspace.example.com",
                environ={"HFLOW_AIRFLOW_TOKEN": "minted-token"},
            )
        with pytest.raises(ValueError, match="HFLOW_AIRFLOW_TOKEN"):
            resolve_remote_endpoint(
                airflow_url="https://workspace.example.com",
                dag_id="kitchen_ingest",
                environ={},
            )

    def test_scheme_less_url_is_refused_at_the_boundary(self) -> None:
        # urllib would otherwise crash with a raw ValueError deep inside the
        # first request; the resolution boundary names the fix instead.
        with pytest.raises(ValueError, match="http:// or https://"):
            resolve_remote_endpoint(
                airflow_url="workspace.example.com:8080",
                dag_id="kitchen_ingest",
                environ={"HFLOW_AIRFLOW_TOKEN": "minted-token"},
            )

    @pytest.mark.parametrize(
        "hostless_url",
        [
            "http://",
            "https://",
            "https:///path",
            # A port or userinfo alone still names no host, and these are the
            # shapes that separate `hostname` from the truthy `netloc`.
            "http://:8080",
            "http://user@",
        ],
    )
    def test_hostless_url_is_refused_at_the_boundary(self, hostless_url: str) -> None:
        with pytest.raises(ValueError, match="needs a host"):
            resolve_remote_endpoint(
                airflow_url=hostless_url,
                dag_id="kitchen_ingest",
                environ={"HFLOW_AIRFLOW_TOKEN": "minted-token"},
            )

    def test_hostless_url_from_environment_is_refused(self) -> None:
        with pytest.raises(ValueError, match="needs a host"):
            resolve_remote_endpoint(
                dag_id="kitchen_ingest",
                environ={
                    "HFLOW_AIRFLOW_URL": "https://",
                    "HFLOW_AIRFLOW_TOKEN": "minted-token",
                },
            )

    @pytest.mark.parametrize(
        "hosted_url",
        [
            "http://127.0.0.1:8080",
            "http://127.0.0.1:8080/airflow",
            "https://workspace.example.com:8443/prefix/path",
            "http://[::1]:8080",
        ],
    )
    def test_explicit_port_and_path_prefix_stay_accepted(self, hosted_url: str) -> None:
        # The host requirement must not narrow the shapes a real deployment
        # uses: loopback addresses, explicit ports, and path-prefixed bases.
        endpoint = resolve_remote_endpoint(
            airflow_url=hosted_url,
            dag_id="kitchen_ingest",
            environ={"HFLOW_AIRFLOW_TOKEN": "minted-token"},
        )
        assert endpoint is not None
        assert endpoint.base_url == hosted_url


def test_describe_remote_status_reports_health_dag_and_recent_runs(stub_server: str) -> None:
    _StubAirflowHandler.issued_tokens.append("pre-issued-token")
    endpoint = RemoteRuntimeEndpoint(
        base_url=stub_server,
        dag_id="pipeline_ingest",
        auth=BearerToken("pre-issued-token"),
    )
    status_text = describe_remote_status(endpoint)
    assert "healthy" in status_text
    assert "pipeline_ingest" in status_text
    assert "manual__1 [success]" in status_text
    # The server truncates the run list to `limit` in id-ASCENDING order by
    # default, which would show a busy DAG's OLDEST history; the request must
    # ask for the newest runs explicitly.
    run_list_requests = [
        entry
        for entry in _StubAirflowHandler.requests_seen
        if "/dagRuns?" in entry[1] and entry[0] == "GET"
    ]
    assert run_list_requests, "no dag-run list request was made"
    assert "order_by=-id" in run_list_requests[-1][1]


def test_describe_remote_status_reports_an_unreachable_endpoint() -> None:
    """The primary operator-facing failure: the workspace is down or the URL
    is wrong. Status must report it, never raise."""
    with socket.socket() as port_probe:  # find a port with no listener
        port_probe.bind(("127.0.0.1", 0))
        unused_port = port_probe.getsockname()[1]
    endpoint = RemoteRuntimeEndpoint(
        base_url=f"http://127.0.0.1:{unused_port}",
        dag_id="pipeline_ingest",
        auth=BearerToken("irrelevant"),
    )
    status_text = describe_remote_status(endpoint)
    assert "unreachable" in status_text


def test_describe_remote_status_reports_an_unavailable_dag(stub_server: str) -> None:
    """A healthy endpoint whose DAG id is wrong (or not yet registered) must
    be reported as such, with the health still shown."""
    _StubAirflowHandler.issued_tokens.append("pre-issued-token")
    endpoint = RemoteRuntimeEndpoint(
        base_url=stub_server,
        dag_id="missing_dag",
        auth=BearerToken("pre-issued-token"),
    )
    status_text = describe_remote_status(endpoint)
    assert "healthy" in status_text
    assert "unavailable" in status_text


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
