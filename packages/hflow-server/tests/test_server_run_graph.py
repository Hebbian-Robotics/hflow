"""GET /api/v1/runtime/runs/{dag_run_id}/graph: one run's live state.

Every Airflow call is stubbed at the AirflowClient method level (the idiom of
``test_ui_runtime.py``): no Docker, no live Airflow, and the bundle under test
is a really rendered one so the dag ids are the real derived ones.
"""

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from hflow_server import ServerSettings, create_app

from hflow.runtime import RuntimeConfig, render_bundle
from hflow.runtime._client import AirflowClient, AirflowClientError

PIPELINE_SOURCE = "import hflow\n\napp = hflow.App('demo', data_root='/opt/airflow/data')\n"

MASTER_DAG_ID = "demo_pipeline_ingest"
MASTER_RUN_ID = "manual__2026-08-21T10:00:00+00:00"
MASTER_STARTED_AT = "2026-08-21T10:00:00+00:00"
YESTERDAYS_MASTER_RUN_ID = "manual__2026-08-20T09:00:00+00:00"


@pytest.fixture()
def runtime_free_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    working_directory = tmp_path / "cwd"
    working_directory.mkdir()
    monkeypatch.chdir(working_directory)
    for variable in ("HFLOW_AIRFLOW_URL", "HFLOW_AIRFLOW_DAG_ID", "HFLOW_AIRFLOW_TOKEN"):
        monkeypatch.delenv(variable, raising=False)
    return working_directory


@pytest.fixture()
def bundle_workspace(tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    pipeline_file = tmp_path / "demo_pipeline.py"
    pipeline_file.write_text(PIPELINE_SOURCE)
    render_bundle(
        RuntimeConfig(pipeline_file=pipeline_file, data_root=data_root), data_root / "runtime"
    )
    return data_root


def _client_over(data_root: Path, assets_dir: Path) -> TestClient:
    return TestClient(create_app(ServerSettings(data_root=str(data_root), assets_dir=assets_dir)))


@pytest.fixture()
def bundle_api(bundle_workspace: Path, unbuilt_assets_dir: Path) -> TestClient:
    return _client_over(bundle_workspace, unbuilt_assets_dir)


def _task_instance(
    task_id: str,
    state: str | None,
    *,
    map_index: int = -1,
    start_date: str | None = None,
    end_date: str | None = None,
    try_number: int = 1,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "state": state,
        "map_index": map_index,
        "start_date": start_date,
        "end_date": end_date,
        "try_number": try_number,
        "operator": "never surfaced",
    }


def _stubbed_airflow(
    monkeypatch: pytest.MonkeyPatch,
    *,
    master_run: dict[str, Any],
    stage_runs: dict[str, list[dict[str, Any]]],
    task_instances: dict[tuple[str, str], list[dict[str, Any]]],
) -> None:
    monkeypatch.setattr(AirflowClient, "dag_run", lambda self, dag_id, dag_run_id: dict(master_run))
    monkeypatch.setattr(
        AirflowClient,
        "dag_runs",
        lambda self, dag_id, *, limit=100, order_by=None: list(stage_runs.get(dag_id, [])),
    )
    monkeypatch.setattr(
        AirflowClient,
        "task_instances",
        lambda self, dag_id, dag_run_id: list(task_instances.get((dag_id, dag_run_id), [])),
    )


def test_run_graph_without_a_runtime_is_a_clear_409(
    runtime_free_cwd: Path, tmp_path: Path, unbuilt_assets_dir: Path
) -> None:
    data_root = tmp_path / "bare-root"
    data_root.mkdir()
    response = _client_over(data_root, unbuilt_assets_dir).get(
        "/api/v1/runtime/runs/manual__1/graph"
    )
    assert response.status_code == 409
    assert "hflow up" in response.json()["detail"]
    assert "Traceback" not in response.text


def test_run_graph_colours_the_master_and_matches_stage_runs(
    bundle_api: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stubbed_airflow(
        monkeypatch,
        master_run={
            "dag_run_id": MASTER_RUN_ID,
            "state": "running",
            "start_date": MASTER_STARTED_AT,
        },
        stage_runs={
            # Two runs of the sync sub-DAG: one from BEFORE this master run
            # (a previous ingest) and the one this run triggered.
            "demo_pipeline_sync": [
                {
                    "dag_run_id": "sync__new",
                    "state": "success",
                    "start_date": "2026-08-21T10:00:05+00:00",
                },
                {
                    "dag_run_id": "sync__old",
                    "state": "failed",
                    "start_date": "2026-08-20T09:00:00+00:00",
                },
            ],
            "demo_pipeline_meta": [
                {
                    "dag_run_id": "meta__new",
                    "state": "running",
                    "start_date": "2026-08-21T10:02:00+00:00",
                }
            ],
        },
        task_instances={
            (MASTER_DAG_ID, MASTER_RUN_ID): [
                _task_instance(
                    "trigger_sync",
                    "success",
                    start_date="2026-08-21T10:00:02+00:00",
                    end_date="2026-08-21T10:01:32+00:00",
                ),
                _task_instance("resolve_profile", "success"),
                _task_instance("enabled_sync", "success"),
                _task_instance("trigger_meta", "deferred"),
            ],
            ("demo_pipeline_sync", "sync__new"): [
                _task_instance("plan", "success"),
                _task_instance("process_batch", "success", map_index=0),
                _task_instance("process_batch", "failed", map_index=1),
                _task_instance("process_batch", "success", map_index=2),
                _task_instance("error_budget_gate", "success"),
            ],
            ("demo_pipeline_meta", "meta__new"): [
                _task_instance("plan", "success"),
                _task_instance("process_batch", "running", map_index=0),
                _task_instance("process_batch", None, map_index=1),
            ],
        },
    )
    response = bundle_api.get(f"/api/v1/runtime/runs/{MASTER_RUN_ID}/graph")
    assert response.status_code == 200
    payload = response.json()

    assert payload["master"]["dag_run_id"] == MASTER_RUN_ID
    assert payload["master"]["state"] == "running"
    # Task instances come back in TOPOLOGY order, not the API's order.
    assert [task["task_id"] for task in payload["master"]["tasks"]] == [
        "resolve_profile",
        "enabled_sync",
        "trigger_sync",
        "trigger_meta",
    ]
    trigger_sync = payload["master"]["tasks"][2]
    assert trigger_sync["duration_s"] == 90.0
    assert trigger_sync["try_number"] == 1
    assert trigger_sync["map_index"] == -1
    # Airflow's own extra fields never reach the browser.
    assert "operator" not in trigger_sync

    stages_by_name = {stage["stage"]: stage for stage in payload["stages"]}
    sync_stage = stages_by_name["sync"]
    # The heuristic picks the run that started after the master run, never
    # the older one, and says out loud that it is a heuristic.
    assert sync_stage["dag_run_id"] == "sync__new"
    assert sync_stage["state"] == "success"
    assert sync_stage["match"] == "heuristic"
    assert sync_stage["mapped_summary"] == {
        "task_id": "process_batch",
        "total": 3,
        "by_state": {"failed": 1, "success": 2},
    }
    assert stages_by_name["meta"]["mapped_summary"] == {
        "task_id": "process_batch",
        "total": 2,
        # A task instance Airflow has not scheduled yet has a null state.
        "by_state": {"no_status": 1, "running": 1},
    }
    # Stages with no run of their own are explicit nulls, not omissions.
    for never_ran in ("labels", "media"):
        assert stages_by_name[never_ran]["dag_run_id"] is None
        assert stages_by_name[never_ran]["state"] is None
        assert stages_by_name[never_ran]["match"] is None
        assert stages_by_name[never_ran]["tasks"] == []
        assert stages_by_name[never_ran]["mapped_summary"] is None
        assert stages_by_name[never_ran]["dag_id"] == f"demo_pipeline_{never_ran}"


def test_run_graph_passes_the_run_id_through_verbatim(
    bundle_api: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Airflow run ids carry ':' and '+'; the path must not mangle them."""
    requested: list[tuple[str, str]] = []

    def capturing_dag_run(self: AirflowClient, dag_id: str, dag_run_id: str) -> dict[str, Any]:
        requested.append((dag_id, dag_run_id))
        return {"dag_run_id": dag_run_id, "state": "success", "start_date": None}

    monkeypatch.setattr(AirflowClient, "dag_run", capturing_dag_run)
    monkeypatch.setattr(AirflowClient, "task_instances", lambda self, dag_id, dag_run_id: [])
    monkeypatch.setattr(
        AirflowClient, "dag_runs", lambda self, dag_id, *, limit=100, order_by=None: []
    )
    response = bundle_api.get(f"/api/v1/runtime/runs/{MASTER_RUN_ID}/graph")
    assert response.status_code == 200
    assert requested == [(MASTER_DAG_ID, MASTER_RUN_ID)]
    assert response.json()["master"]["dag_run_id"] == MASTER_RUN_ID


def test_run_graph_mapped_summary_counts_the_unexpanded_placeholder(
    bundle_api: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stubbed_airflow(
        monkeypatch,
        master_run={
            "dag_run_id": MASTER_RUN_ID,
            "state": "running",
            "start_date": MASTER_STARTED_AT,
        },
        stage_runs={
            "demo_pipeline_sync": [
                {
                    "dag_run_id": "sync__new",
                    "state": "running",
                    "start_date": "2026-08-21T10:00:05+00:00",
                }
            ]
        },
        task_instances={
            ("demo_pipeline_sync", "sync__new"): [
                _task_instance("plan", "running"),
                # Before the fan-out expands, Airflow reports ONE instance.
                _task_instance("process_batch", None, map_index=-1),
            ]
        },
    )
    payload = bundle_api.get(f"/api/v1/runtime/runs/{MASTER_RUN_ID}/graph").json()
    sync_stage = next(stage for stage in payload["stages"] if stage["stage"] == "sync")
    assert sync_stage["mapped_summary"] == {
        "task_id": "process_batch",
        "total": 1,
        "by_state": {"no_status": 1},
    }


def test_run_graph_without_a_master_start_matches_no_stage_run(
    bundle_api: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A queued master run cannot be attributed stage runs -- and says so."""
    _stubbed_airflow(
        monkeypatch,
        master_run={"dag_run_id": MASTER_RUN_ID, "state": "queued", "start_date": None},
        stage_runs={
            "demo_pipeline_sync": [
                {
                    "dag_run_id": "sync__unrelated",
                    "state": "success",
                    "start_date": "2026-08-20T10:00:05+00:00",
                }
            ]
        },
        task_instances={},
    )
    payload = bundle_api.get(f"/api/v1/runtime/runs/{MASTER_RUN_ID}/graph").json()
    assert payload["master"]["state"] == "queued"
    assert all(stage["dag_run_id"] is None for stage in payload["stages"])


def test_an_ended_master_run_never_adopts_a_later_unrelated_stage_run(
    bundle_api: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two ingests over one lane: each run shows the stage runs it triggered.

    The stage lanes only look back a fixed number of runs, so an unbounded
    "started at/after the master" filter matched every candidate for any
    master run that was not the latest -- and keeping the newest handed an old
    run today's unrelated stage run, labelled as an attribution.
    """
    master_runs = {
        YESTERDAYS_MASTER_RUN_ID: {
            "dag_run_id": YESTERDAYS_MASTER_RUN_ID,
            "state": "success",
            "start_date": "2026-08-20T09:00:00+00:00",
            "end_date": "2026-08-20T09:04:00+00:00",
        },
        MASTER_RUN_ID: {
            "dag_run_id": MASTER_RUN_ID,
            "state": "running",
            "start_date": MASTER_STARTED_AT,
            "end_date": None,
        },
    }
    # Newest first, exactly as the runs are fetched (order_by="-id").
    interleaved_stage_runs = {
        "demo_pipeline_sync": [
            {
                "dag_run_id": "sync__today",
                "state": "running",
                "start_date": "2026-08-21T10:00:05+00:00",
            },
            {
                "dag_run_id": "sync__yesterday",
                "state": "success",
                "start_date": "2026-08-20T09:00:05+00:00",
            },
        ],
        # Today's ingest reached meta; yesterday's never did.
        "demo_pipeline_meta": [
            {
                "dag_run_id": "meta__today",
                "state": "running",
                "start_date": "2026-08-21T10:02:00+00:00",
            }
        ],
    }
    monkeypatch.setattr(
        AirflowClient, "dag_run", lambda self, dag_id, dag_run_id: dict(master_runs[dag_run_id])
    )
    monkeypatch.setattr(
        AirflowClient,
        "dag_runs",
        lambda self, dag_id, *, limit=100, order_by=None: list(
            interleaved_stage_runs.get(dag_id, [])
        ),
    )
    monkeypatch.setattr(AirflowClient, "task_instances", lambda self, dag_id, dag_run_id: [])

    yesterday = {
        stage["stage"]: stage
        for stage in bundle_api.get(
            f"/api/v1/runtime/runs/{YESTERDAYS_MASTER_RUN_ID}/graph"
        ).json()["stages"]
    }
    assert yesterday["sync"]["dag_run_id"] == "sync__yesterday"
    assert yesterday["sync"]["state"] == "success"
    assert yesterday["sync"]["match"] == "heuristic"
    # Nothing ran in that master run's window, so the lane stays empty rather
    # than borrowing today's still-running one.
    assert yesterday["meta"]["dag_run_id"] is None
    assert yesterday["meta"]["match"] is None

    today = {
        stage["stage"]: stage
        for stage in bundle_api.get(f"/api/v1/runtime/runs/{MASTER_RUN_ID}/graph").json()["stages"]
    }
    assert today["sync"]["dag_run_id"] == "sync__today"
    assert today["meta"]["dag_run_id"] == "meta__today"


def test_a_stage_run_starting_just_after_the_master_ended_still_matches(
    bundle_api: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A master that dies right after firing a trigger keeps its stage run.

    The two timestamps come from different components, and a failing master
    can end before the run it just caused appears, so the window carries a
    grace period rather than a hard edge.
    """
    _stubbed_airflow(
        monkeypatch,
        master_run={
            "dag_run_id": MASTER_RUN_ID,
            "state": "failed",
            "start_date": MASTER_STARTED_AT,
            "end_date": "2026-08-21T10:00:04+00:00",
        },
        stage_runs={
            "demo_pipeline_sync": [
                {
                    "dag_run_id": "sync__just_after",
                    "state": "running",
                    "start_date": "2026-08-21T10:00:06+00:00",
                }
            ]
        },
        task_instances={},
    )
    payload = bundle_api.get(f"/api/v1/runtime/runs/{MASTER_RUN_ID}/graph").json()
    sync_stage = next(stage for stage in payload["stages"] if stage["stage"] == "sync")
    assert sync_stage["dag_run_id"] == "sync__just_after"


def test_run_graph_tolerates_unregistered_stage_sub_dags(
    bundle_api: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def stage_runs_404(
        self: AirflowClient, dag_id: str, *, limit: int = 100, order_by: str | None = None
    ) -> list[dict[str, Any]]:
        raise AirflowClientError(f"GET /dags/{dag_id}/dagRuns failed with HTTP 404", status=404)

    monkeypatch.setattr(
        AirflowClient,
        "dag_run",
        lambda self, dag_id, dag_run_id: {
            "dag_run_id": dag_run_id,
            "state": "success",
            "start_date": MASTER_STARTED_AT,
        },
    )
    monkeypatch.setattr(AirflowClient, "dag_runs", stage_runs_404)
    monkeypatch.setattr(AirflowClient, "task_instances", lambda self, dag_id, dag_run_id: [])
    response = bundle_api.get(f"/api/v1/runtime/runs/{MASTER_RUN_ID}/graph")
    assert response.status_code == 200
    payload = response.json()
    assert payload["master"]["state"] == "success"
    assert all(stage["dag_run_id"] is None for stage in payload["stages"])


def test_run_graph_maps_stage_dag_listing_failures_to_502(
    bundle_api: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stubbed_airflow(
        monkeypatch,
        master_run={
            "dag_run_id": MASTER_RUN_ID,
            "state": "success",
            "start_date": MASTER_STARTED_AT,
        },
        stage_runs={},
        task_instances={(MASTER_DAG_ID, MASTER_RUN_ID): []},
    )

    def failing_stage_runs(
        self: AirflowClient, dag_id: str, *, limit: int = 100, order_by: str | None = None
    ) -> list[dict[str, Any]]:
        if dag_id != MASTER_DAG_ID:
            raise AirflowClientError(
                "GET /dagRuns failed with HTTP 503: scheduler down", status=503
            )
        return []

    monkeypatch.setattr(AirflowClient, "dag_runs", failing_stage_runs)
    response = bundle_api.get(f"/api/v1/runtime/runs/{MASTER_RUN_ID}/graph")
    assert response.status_code == 502
    assert "scheduler down" in response.json()["detail"]


def test_run_graph_maps_stage_task_listing_failures_to_502(
    bundle_api: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stubbed_airflow(
        monkeypatch,
        master_run={
            "dag_run_id": MASTER_RUN_ID,
            "state": "success",
            "start_date": MASTER_STARTED_AT,
        },
        stage_runs={
            "demo_pipeline_sync": [
                {
                    "dag_run_id": "sync__run",
                    "state": "success",
                    "start_date": "2026-08-21T10:00:05+00:00",
                }
            ]
        },
        task_instances={(MASTER_DAG_ID, MASTER_RUN_ID): []},
    )

    def failing_task_instances(
        self: AirflowClient, dag_id: str, dag_run_id: str
    ) -> list[dict[str, Any]]:
        if dag_id != MASTER_DAG_ID:
            raise AirflowClientError(
                "GET /taskInstances failed with HTTP 401: unauthorized", status=401
            )
        return []

    monkeypatch.setattr(AirflowClient, "task_instances", failing_task_instances)
    response = bundle_api.get(f"/api/v1/runtime/runs/{MASTER_RUN_ID}/graph")
    assert response.status_code == 502
    assert "unauthorized" in response.json()["detail"]


def test_run_graph_unknown_run_is_a_404(
    bundle_api: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing_run(self: AirflowClient, dag_id: str, dag_run_id: str) -> dict[str, Any]:
        raise AirflowClientError(
            f"GET /dags/{dag_id}/dagRuns/{dag_run_id} failed with HTTP 404", status=404
        )

    monkeypatch.setattr(AirflowClient, "dag_run", missing_run)
    response = bundle_api.get("/api/v1/runtime/runs/manual__nope/graph")
    assert response.status_code == 404
    assert "manual__nope" in response.json()["detail"]
    assert "Traceback" not in response.text


def test_run_graph_maps_airflow_failures_to_502(
    bundle_api: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_run(self: AirflowClient, dag_id: str, dag_run_id: str) -> dict[str, Any]:
        raise AirflowClientError("GET /dagRuns failed with HTTP 503: scheduler down", status=503)

    monkeypatch.setattr(AirflowClient, "dag_run", failing_run)
    response = bundle_api.get(f"/api/v1/runtime/runs/{MASTER_RUN_ID}/graph")
    assert response.status_code == 502
    assert "scheduler down" in response.json()["detail"]


def test_run_graph_remote_failure_does_not_leak_the_base_url(
    runtime_free_cwd: Path,
    tmp_path: Path,
    unbuilt_assets_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HFLOW_AIRFLOW_URL", "https://airflow.internal.corp:8443")
    monkeypatch.setenv("HFLOW_AIRFLOW_DAG_ID", "kitchen_ingest")
    monkeypatch.setenv("HFLOW_AIRFLOW_TOKEN", "minted-token")

    def failing_run(self: AirflowClient, dag_id: str, dag_run_id: str) -> dict[str, Any]:
        raise AirflowClientError(
            "GET https://airflow.internal.corp:8443/api/v2/dags/kitchen_ingest/dagRuns/x "
            "failed with HTTP 500: internal"
        )

    monkeypatch.setattr(AirflowClient, "dag_run", failing_run)
    data_root = tmp_path / "bare-root"
    data_root.mkdir()
    response = _client_over(data_root, unbuilt_assets_dir).get("/api/v1/runtime/runs/x/graph")
    assert response.status_code == 502
    assert "airflow.internal.corp" not in response.text


def test_run_graph_over_a_remote_runtime_derives_the_stage_dag_ids(
    runtime_free_cwd: Path,
    tmp_path: Path,
    unbuilt_assets_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A remote master id derives its sub-DAG ids the same way the renderer did."""
    monkeypatch.setenv("HFLOW_AIRFLOW_URL", "https://workspace.example.com")
    monkeypatch.setenv("HFLOW_AIRFLOW_DAG_ID", "kitchen_ingest")
    monkeypatch.setenv("HFLOW_AIRFLOW_TOKEN", "minted-token")
    _stubbed_airflow(
        monkeypatch,
        master_run={"dag_run_id": "r1", "state": "success", "start_date": MASTER_STARTED_AT},
        stage_runs={},
        task_instances={},
    )
    data_root = tmp_path / "bare-root"
    data_root.mkdir()
    payload = (
        _client_over(data_root, unbuilt_assets_dir).get("/api/v1/runtime/runs/r1/graph").json()
    )
    assert [stage["dag_id"] for stage in payload["stages"]] == [
        "kitchen_sync",
        "kitchen_meta",
        "kitchen_labels",
        "kitchen_media",
    ]
