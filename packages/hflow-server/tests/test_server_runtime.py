"""/api/v1/runtime/*: bundle/remote addressing with stubbed AirflowClient.

No Docker and no live Airflow anywhere: bundles are rendered with
``render_bundle`` (plain file writing) and every Airflow API call is stubbed
at the AirflowClient method level, the same idiom as the repository's
``tests/test_runtime_cli.py``.
"""

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from hflow_server import ServerSettings, create_app

from hflow.runtime import RuntimeConfig, render_bundle
from hflow.runtime._client import AirflowClient, AirflowClientError, AirflowDagRun, AirflowHealth

HEALTHY = AirflowHealth(
    components={
        "metadatabase": "healthy",
        "scheduler": "healthy",
        "dag_processor": "healthy",
        "triggerer": "healthy",
    }
)

PIPELINE_SOURCE = "import hflow\n\napp = hflow.App('demo', data_root='/opt/airflow/data')\n"


def _client_over(data_root: Path, assets_dir: Path, *, read_only: bool = False) -> TestClient:
    settings = ServerSettings(data_root=str(data_root), assets_dir=assets_dir, read_only=read_only)
    return TestClient(create_app(settings))


@pytest.fixture()
def runtime_free_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A cwd with no ./runtime fallback and no remote environment exported."""
    working_directory = tmp_path / "cwd"
    working_directory.mkdir()
    monkeypatch.chdir(working_directory)
    for variable in ("HFLOW_AIRFLOW_URL", "HFLOW_AIRFLOW_DAG_ID", "HFLOW_AIRFLOW_TOKEN"):
        monkeypatch.delenv(variable, raising=False)
    return working_directory


@pytest.fixture()
def bundle_workspace(tmp_path: Path) -> Path:
    """A data root whose ``runtime/`` holds a really rendered Compose bundle."""
    data_root = tmp_path / "data"
    pipeline_file = tmp_path / "demo_pipeline.py"
    pipeline_file.write_text(PIPELINE_SOURCE)
    render_bundle(
        RuntimeConfig(pipeline_file=pipeline_file, data_root=data_root), data_root / "runtime"
    )
    return data_root


@pytest.fixture()
def bundle_api(bundle_workspace: Path, unbuilt_assets_dir: Path) -> TestClient:
    return _client_over(bundle_workspace, unbuilt_assets_dir)


def test_status_without_any_runtime_is_available_false_not_a_traceback(
    runtime_free_cwd: Path, tmp_path: Path, unbuilt_assets_dir: Path
) -> None:
    data_root = tmp_path / "bare-root"
    data_root.mkdir()
    response = _client_over(data_root, unbuilt_assets_dir).get("/api/v1/runtime/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is False
    assert "hflow up" in payload["detail"]
    assert "HFLOW_AIRFLOW_URL" in payload["detail"]
    assert payload["source"] is None
    assert payload["dag_id"] is None
    assert payload["health"] is None
    assert "Traceback" not in response.text


def test_status_reports_a_healthy_bundle_runtime(
    bundle_api: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(AirflowClient, "health", lambda self: HEALTHY)
    monkeypatch.setattr(AirflowClient, "dag", lambda self, dag_id: {"dag_id": dag_id})
    payload = bundle_api.get("/api/v1/runtime/status").json()
    assert payload == {
        "available": True,
        "detail": None,
        "source": "bundle",
        "airflow_web_url": "http://127.0.0.1:8080",
        # A rendered bundle binds its api-server to loopback, so the address
        # is only followable on the workspace host -- a browser elsewhere
        # would aim it at its own machine.
        "airflow_web_url_host_only": True,
        "dag_id": "demo_pipeline_ingest",
        "registered": True,
        "health": {
            "metadatabase": "healthy",
            "scheduler": "healthy",
            "triggerer": "healthy",
            "dag_processor": "healthy",
        },
    }


def test_status_reports_an_unregistered_dag(
    bundle_api: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def dag_not_found(self: AirflowClient, dag_id: str) -> dict[str, Any]:
        raise AirflowClientError(f"GET /dags/{dag_id} failed with HTTP 404", status=404)

    monkeypatch.setattr(AirflowClient, "health", lambda self: HEALTHY)
    monkeypatch.setattr(AirflowClient, "dag", dag_not_found)
    payload = bundle_api.get("/api/v1/runtime/status").json()
    assert payload["available"] is True
    assert payload["registered"] is False


def test_status_with_unreachable_bundle_runtime_stays_a_calm_answer(
    bundle_api: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unreachable_health(self: AirflowClient) -> AirflowHealth:
        raise AirflowClientError("GET http://127.0.0.1:8080 unreachable: connection refused")

    monkeypatch.setattr(AirflowClient, "health", unreachable_health)
    response = bundle_api.get("/api/v1/runtime/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is False
    assert "unreachable" in payload["detail"]
    # The addressing facts still ride along: the bundle IS configured.
    assert payload["source"] == "bundle"
    assert payload["dag_id"] == "demo_pipeline_ingest"
    assert payload["airflow_web_url"] == "http://127.0.0.1:8080"
    assert "Traceback" not in response.text


def test_status_addresses_the_remote_environment(
    runtime_free_cwd: Path,
    tmp_path: Path,
    unbuilt_assets_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HFLOW_AIRFLOW_URL", "https://workspace.example.com")
    monkeypatch.setenv("HFLOW_AIRFLOW_DAG_ID", "kitchen_ingest")
    monkeypatch.setenv("HFLOW_AIRFLOW_TOKEN", "minted-token")
    monkeypatch.setattr(AirflowClient, "health", lambda self: HEALTHY)
    monkeypatch.setattr(AirflowClient, "dag", lambda self, dag_id: {"dag_id": dag_id})
    data_root = tmp_path / "bare-root"
    data_root.mkdir()
    payload = _client_over(data_root, unbuilt_assets_dir).get("/api/v1/runtime/status").json()
    assert payload["available"] is True
    assert payload["source"] == "remote"
    assert payload["dag_id"] == "kitchen_ingest"
    # Only a bundle records its own web address; a remote endpoint's is unknown.
    assert payload["airflow_web_url"] is None


def test_status_remote_failure_does_not_leak_the_base_url(
    runtime_free_cwd: Path,
    tmp_path: Path,
    unbuilt_assets_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HFLOW_AIRFLOW_URL", "https://airflow.internal.corp:8443")
    monkeypatch.setenv("HFLOW_AIRFLOW_DAG_ID", "kitchen_ingest")
    monkeypatch.setenv("HFLOW_AIRFLOW_TOKEN", "minted-token")

    def unreachable_health(self: AirflowClient) -> AirflowHealth:
        raise AirflowClientError(
            "GET https://airflow.internal.corp:8443/api/v2/monitor/health "
            "unreachable: [Errno 111] Connection refused"
        )

    monkeypatch.setattr(AirflowClient, "health", unreachable_health)
    data_root = tmp_path / "bare-root"
    data_root.mkdir()
    payload = _client_over(data_root, unbuilt_assets_dir).get("/api/v1/runtime/status").json()
    assert payload["available"] is False
    # The remote base URL the success path withholds must not leak on failure.
    assert "airflow.internal.corp" not in payload["detail"]
    assert payload["airflow_web_url"] is None
    # A stable machine-readable reason still rides along.
    assert "unreachable" in payload["detail"]


def test_status_remote_incomplete_environment_names_the_missing_variable(
    runtime_free_cwd: Path,
    tmp_path: Path,
    unbuilt_assets_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HFLOW_AIRFLOW_URL", "https://workspace.example.com")
    data_root = tmp_path / "bare-root"
    data_root.mkdir()
    response = _client_over(data_root, unbuilt_assets_dir).get("/api/v1/runtime/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is False
    assert "HFLOW_AIRFLOW_DAG_ID" in payload["detail"]
    assert "Traceback" not in response.text


def test_runs_shape_over_a_bundle_with_stage_strips(
    bundle_api: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    dag_runs_calls: list[tuple[str, int, str | None]] = []

    def fake_dag_runs(
        self: AirflowClient, dag_id: str, *, limit: int = 100, order_by: str | None = None
    ) -> list[AirflowDagRun]:
        dag_runs_calls.append((dag_id, limit, order_by))
        if dag_id == "demo_pipeline_ingest":
            return [
                AirflowDagRun(
                    dag_run_id="manual__1",
                    state="success",
                    logical_date=None,
                    start_date="2026-08-21T00:00:00Z",
                    end_date="2026-08-21T00:05:00Z",
                    conf={"uris": ["a.mcap"], "profile": "full", "mode": "batch"},
                )
            ]
        return [
            AirflowDagRun(
                dag_run_id=f"{dag_id}__r1",
                state="running",
                logical_date=None,
                start_date=None,
                end_date=None,
                conf={},
            )
        ]

    monkeypatch.setattr(AirflowClient, "dag_runs", fake_dag_runs)
    response = bundle_api.get("/api/v1/runtime/runs", params={"limit": 5})
    assert response.status_code == 200
    payload = response.json()
    assert payload["runs"] == [
        {
            "dag_run_id": "manual__1",
            "state": "success",
            "logical_date": None,
            "start_date": "2026-08-21T00:00:00Z",
            "end_date": "2026-08-21T00:05:00Z",
            "conf": {"uris": ["a.mcap"], "profile": "full", "mode": "batch"},
        }
    ]
    assert [stage["stage"] for stage in payload["stages"]] == ["sync", "meta", "labels", "media"]
    assert [stage["dag_id"] for stage in payload["stages"]] == [
        "demo_pipeline_sync",
        "demo_pipeline_meta",
        "demo_pipeline_labels",
        "demo_pipeline_media",
    ]
    assert payload["stages"][0]["recent"] == [
        {
            "dag_run_id": "demo_pipeline_sync__r1",
            "state": "running",
            "start_date": None,
            "end_date": None,
        }
    ]
    # Newest-first ordering is requested explicitly (Airflow truncates in id
    # order otherwise), and the master honors the caller's limit.
    assert dag_runs_calls[0] == ("demo_pipeline_ingest", 5, "-id")


def test_runs_tolerates_an_unregistered_stage_dag(
    bundle_api: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_dag_runs(
        self: AirflowClient, dag_id: str, *, limit: int = 100, order_by: str | None = None
    ) -> list[dict[str, Any]]:
        if dag_id == "demo_pipeline_ingest":
            return []
        raise AirflowClientError(f"GET /dags/{dag_id}/dagRuns failed with HTTP 404", status=404)

    monkeypatch.setattr(AirflowClient, "dag_runs", fake_dag_runs)
    payload = bundle_api.get("/api/v1/runtime/runs").json()
    assert payload["runs"] == []
    assert all(stage["recent"] == [] for stage in payload["stages"])


def test_runs_without_a_runtime_is_a_clear_409(
    runtime_free_cwd: Path, tmp_path: Path, unbuilt_assets_dir: Path
) -> None:
    data_root = tmp_path / "bare-root"
    data_root.mkdir()
    response = _client_over(data_root, unbuilt_assets_dir).get("/api/v1/runtime/runs")
    assert response.status_code == 409
    assert "hflow up" in response.json()["detail"]


def test_ingest_is_refused_read_only(bundle_workspace: Path, unbuilt_assets_dir: Path) -> None:
    client = _client_over(bundle_workspace, unbuilt_assets_dir, read_only=True)
    response = client.post(
        "/api/v1/runtime/ingest", json={"uris": ["a.mcap"], "profile": "full", "mode": "batch"}
    )
    assert response.status_code == 403
    assert "read-only" in response.json()["detail"]


def test_ingest_validates_profile_and_mode_before_touching_any_runtime(
    runtime_free_cwd: Path, tmp_path: Path, unbuilt_assets_dir: Path
) -> None:
    data_root = tmp_path / "bare-root"
    data_root.mkdir()
    client = _client_over(data_root, unbuilt_assets_dir)

    bad_profile = client.post(
        "/api/v1/runtime/ingest",
        json={"uris": ["a.mcap"], "profile": "everything", "mode": "batch"},
    )
    assert bad_profile.status_code == 400
    assert "full" in bad_profile.json()["detail"]

    bad_mode = client.post(
        "/api/v1/runtime/ingest",
        json={"uris": ["a.mcap"], "profile": "full", "mode": "streaming"},
    )
    assert bad_mode.status_code == 400
    assert "online" in bad_mode.json()["detail"]

    blank_uri = client.post(
        "/api/v1/runtime/ingest", json={"uris": ["   "], "profile": "full", "mode": "batch"}
    )
    assert blank_uri.status_code == 400

    no_uris = client.post(
        "/api/v1/runtime/ingest", json={"uris": [], "profile": "full", "mode": "batch"}
    )
    assert no_uris.status_code == 422  # pydantic: the list itself must be non-empty


def test_ingest_rejects_absolute_and_escaping_uris_before_any_runtime(
    runtime_free_cwd: Path, tmp_path: Path, unbuilt_assets_dir: Path
) -> None:
    # URIs resolve against the runtime's data root; absolute paths and ../
    # escapes cannot work there, so they are refused with a 400 before the
    # runtime is even resolved -- the same guard `hflow ingest` enforces.
    data_root = tmp_path / "bare-root"
    data_root.mkdir()
    client = _client_over(data_root, unbuilt_assets_dir)
    for hostile_uri in ("/etc/passwd", "../../etc/shadow", "sub/../../escape.mcap"):
        response = client.post(
            "/api/v1/runtime/ingest",
            json={"uris": [hostile_uri], "profile": "full", "mode": "batch"},
        )
        assert response.status_code == 400, hostile_uri
        assert "not relative to the data root" in response.json()["detail"]


def test_ingest_without_a_runtime_is_a_clear_409(
    runtime_free_cwd: Path, tmp_path: Path, unbuilt_assets_dir: Path
) -> None:
    data_root = tmp_path / "bare-root"
    data_root.mkdir()
    response = _client_over(data_root, unbuilt_assets_dir).post(
        "/api/v1/runtime/ingest", json={"uris": ["a.mcap"], "profile": "full", "mode": "batch"}
    )
    assert response.status_code == 409
    assert "Traceback" not in response.text


def test_ingest_triggers_the_master_dag(
    bundle_api: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_ingest(
        self: AirflowClient,
        dag_id: str,
        uris: list[str],
        *,
        profile: str = "full",
        online: bool = False,
        batch_count: int | None = None,
        dag_run_id: str | None = None,
    ) -> AirflowDagRun:
        captured["dag_id"] = dag_id
        captured["uris"] = uris
        captured["profile"] = profile
        captured["online"] = online
        captured["batch_count"] = batch_count
        return AirflowDagRun(
            dag_run_id="manual__ui",
            state="queued",
            logical_date=None,
            start_date=None,
            end_date=None,
            conf={},
        )

    monkeypatch.setattr(AirflowClient, "ingest", fake_ingest)
    response = bundle_api.post(
        "/api/v1/runtime/ingest",
        json={"uris": ["a.mcap", "sub/b.mcap"], "profile": "relabel", "mode": "online"},
    )
    assert response.status_code == 200
    assert response.json() == {"dag_run_id": "manual__ui", "state": "queued"}
    assert captured == {
        "dag_id": "demo_pipeline_ingest",
        "uris": ["a.mcap", "sub/b.mcap"],
        "profile": "relabel",
        "online": True,
        "batch_count": None,
    }


def test_ingest_batch_count_rides_the_trigger_conf(
    bundle_api: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_trigger(
        self: AirflowClient,
        dag_id: str,
        conf: dict[str, Any] | None = None,
        *,
        dag_run_id: str | None = None,
    ) -> AirflowDagRun:
        captured["dag_id"] = dag_id
        captured["conf"] = conf
        return AirflowDagRun(
            dag_run_id="manual__sharded",
            state="queued",
            logical_date=None,
            start_date=None,
            end_date=None,
            conf={},
        )

    monkeypatch.setattr(AirflowClient, "trigger_dag_run", fake_trigger)
    response = bundle_api.post(
        "/api/v1/runtime/ingest",
        json={"uris": ["a.mcap"], "profile": "full", "mode": "batch", "batch_count": 3},
    )
    assert response.status_code == 200
    assert captured["dag_id"] == "demo_pipeline_ingest"
    assert captured["conf"] == {
        "uris": ["a.mcap"],
        "profile": "full",
        "mode": "batch",
        "batch_count": 3,
    }


def test_ingest_maps_airflow_errors_to_502(
    bundle_api: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_ingest(
        self: AirflowClient,
        dag_id: str,
        uris: list[str],
        *,
        profile: str = "full",
        online: bool = False,
        batch_count: int | None = None,
        dag_run_id: str | None = None,
    ) -> dict[str, str]:
        raise AirflowClientError("POST /dagRuns failed with HTTP 503: scheduler down")

    monkeypatch.setattr(AirflowClient, "ingest", failing_ingest)
    response = bundle_api.post(
        "/api/v1/runtime/ingest", json={"uris": ["a.mcap"], "profile": "full", "mode": "batch"}
    )
    assert response.status_code == 502
    assert "scheduler down" in response.json()["detail"]


def test_loopback_classification_marks_only_host_local_addresses() -> None:
    """The Runs page needs to know when a deep link cannot be followed.

    A viewer on another machine reads http://127.0.0.1:8080 as their OWN
    laptop, so the payload states whether the address is host-local rather
    than leaving the browser to guess.
    """
    from hflow_server._runtime import is_loopback_web_url

    assert is_loopback_web_url("http://127.0.0.1:8080") is True
    assert is_loopback_web_url("http://localhost:8080") is True
    assert is_loopback_web_url("http://[::1]:8080") is True
    assert is_loopback_web_url("http://100.104.216.28:8080") is False
    assert is_loopback_web_url("https://airflow.example.com") is False
    assert is_loopback_web_url(None) is False


def test_an_unreachable_bundle_names_the_workspace_host_as_the_caller(
    bundle_api: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure text must not read as the viewer's own machine failing."""

    def refuse(self: AirflowClient) -> None:
        raise AirflowClientError("GET http://127.0.0.1:8080/api/v2/monitor/health unreachable")

    monkeypatch.setattr(AirflowClient, "health", refuse)
    payload = bundle_api.get("/api/v1/runtime/status").json()
    assert payload["available"] is False
    assert "the workspace host could not reach its own ingest runtime" in payload["detail"]
