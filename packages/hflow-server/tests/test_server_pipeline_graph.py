"""GET /api/v1/pipeline/graph: the DAG topology merged with the user's steps.

Nothing here needs Docker: the bundle fixtures render real files with
``render_bundle`` (plain file writing) and no Airflow call is made -- the
pipeline graph is pure description, and its only runtime input is whether a
bundle is ADDRESSED.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from hflow_server import ServerSettings, create_app

from hflow.runtime import RuntimeConfig, render_bundle
from hflow.steps import RUN_PROFILES, Stage

# Mixed tiers on purpose: `cheap_check` declares neither requires nor uses
# (tier 1), `needs_channel` declares requires, `needs_endpoint` declares uses
# -- and both are registered BEFORE the cheap one, so a payload that merely
# echoed registration order would fail the ordering assertions.
TIERED_PIPELINE_SOURCE = """import hflow

app = hflow.App("tiered-demo", endpoints={"vlm": "http://vlm.invalid"})


@app.check(name="needs_channel", requires=["/camera/wrist"], critical=True)
def needs_channel(episode):
    return hflow.CheckResult(measurements={"frames": 1.0})


@app.check(name="needs_endpoint", uses="vlm")
def needs_endpoint(episode):
    return hflow.CheckResult(measurements={"score": 1.0})


@app.check(name="cheap_check", critical=True)
def cheap_check(episode):
    return hflow.CheckResult(verdict=True)


@app.enrich(name="rich_caption", uses="vlm")
def rich_caption(episode):
    return hflow.EnrichmentResult(labels={"caption": "x"})


@app.enrich(name="cheap_label")
def cheap_label(episode):
    return hflow.EnrichmentResult(labels={"ok": True})
"""

NO_CRITICAL_PIPELINE_SOURCE = """import hflow

app = hflow.App("no-critical-demo")


@app.check(name="just_evidence")
def just_evidence(episode):
    return hflow.CheckResult(measurements={"value": 1.0})
"""

BUNDLE_PIPELINE_SOURCE = "import hflow\n\napp = hflow.App('demo', data_root='/opt/airflow/data')\n"


@pytest.fixture()
def runtime_free_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A cwd with no ./runtime fallback and no remote environment exported."""
    working_directory = tmp_path / "cwd"
    working_directory.mkdir()
    monkeypatch.chdir(working_directory)
    for variable in ("HFLOW_AIRFLOW_URL", "HFLOW_AIRFLOW_DAG_ID", "HFLOW_AIRFLOW_TOKEN"):
        monkeypatch.delenv(variable, raising=False)
    return working_directory


def _client_over(data_root: Path, assets_dir: Path, *, pipeline: str | None = None) -> TestClient:
    settings = ServerSettings(data_root=str(data_root), assets_dir=assets_dir, pipeline=pipeline)
    return TestClient(create_app(settings))


def _written_pipeline_file(tmp_path: Path, source: str, name: str = "graph_pipeline.py") -> Path:
    pipeline_file = tmp_path / name
    pipeline_file.write_text(source)
    return pipeline_file


def _rendered_bundle_root(tmp_path: Path) -> Path:
    data_root = tmp_path / "bundle-data"
    bundle_pipeline_file = tmp_path / "demo_pipeline.py"
    bundle_pipeline_file.write_text(BUNDLE_PIPELINE_SOURCE)
    render_bundle(
        RuntimeConfig(pipeline_file=bundle_pipeline_file, data_root=data_root),
        data_root / "runtime",
    )
    return data_root


def test_graph_with_neither_runtime_nor_pipeline_is_explicit_not_an_error(
    runtime_free_cwd: Path, tmp_path: Path, unbuilt_assets_dir: Path
) -> None:
    data_root = tmp_path / "bare-root"
    data_root.mkdir()
    response = _client_over(data_root, unbuilt_assets_dir).get("/api/v1/pipeline/graph")
    assert response.status_code == 200
    payload = response.json()
    assert payload["dag_ids_known"] is False
    assert payload["steps_known"] is False
    # The shape is still fully drawable under a display-only master id.
    assert payload["master"]["dag_id"] == "ingest"
    assert [task["task_id"] for task in payload["master"]["tasks"]][:3] == [
        "resolve_profile",
        "enabled_sync",
        "trigger_sync",
    ]
    assert [stage["stage"] for stage in payload["stages"]] == ["sync", "meta", "labels", "media"]
    assert all(stage["user_steps"] == [] for stage in payload["stages"])
    assert payload["quarantine_gate"] is None
    assert "Traceback" not in response.text


def test_graph_without_a_runtime_uses_the_pipeline_name_as_a_display_only_id(
    runtime_free_cwd: Path, tmp_path: Path, unbuilt_assets_dir: Path
) -> None:
    pipeline_file = _written_pipeline_file(tmp_path, TIERED_PIPELINE_SOURCE)
    data_root = tmp_path / "bare-root"
    data_root.mkdir()
    payload = (
        _client_over(data_root, unbuilt_assets_dir, pipeline=str(pipeline_file))
        .get("/api/v1/pipeline/graph")
        .json()
    )
    assert payload["dag_ids_known"] is False
    assert payload["steps_known"] is True
    assert payload["master"]["dag_id"] == "tiered-demo"
    stages_by_name = {stage["stage"]: stage for stage in payload["stages"]}
    assert stages_by_name["meta"]["dag"]["dag_id"] == "tiered-demo_meta"


def test_graph_over_a_rendered_bundle_serves_the_real_dag_ids(
    runtime_free_cwd: Path, tmp_path: Path, unbuilt_assets_dir: Path
) -> None:
    data_root = _rendered_bundle_root(tmp_path)
    payload = _client_over(data_root, unbuilt_assets_dir).get("/api/v1/pipeline/graph").json()
    assert payload["dag_ids_known"] is True
    assert payload["steps_known"] is False
    assert payload["master"]["dag_id"] == "demo_pipeline_ingest"
    assert [stage["dag"]["dag_id"] for stage in payload["stages"]] == [
        "demo_pipeline_sync",
        "demo_pipeline_meta",
        "demo_pipeline_labels",
        "demo_pipeline_media",
    ]
    # No pipeline imported: the steps are unknown, but the engine's own work
    # is a fact of the ENGINE, so every stage still describes it.
    assert payload["quarantine_gate"] is None
    engine_step_names = {
        stage["stage"]: [step["name"] for step in stage["engine_steps"]]
        for stage in payload["stages"]
    }
    assert engine_step_names == {
        "sync": ["canonical transform"],
        "meta": ["catalog registration"],
        "labels": [],
        "media": ["media/contact_sheet"],
    }


def test_graph_master_edges_chain_the_stages(
    runtime_free_cwd: Path, tmp_path: Path, unbuilt_assets_dir: Path
) -> None:
    data_root = tmp_path / "bare-root"
    data_root.mkdir()
    payload = _client_over(data_root, unbuilt_assets_dir).get("/api/v1/pipeline/graph").json()
    edges = {(edge[0], edge[1]) for edge in payload["master"]["edges"]}
    for stage in Stage:
        assert ("resolve_profile", f"enabled_{stage.value}") in edges
        assert (f"enabled_{stage.value}", f"trigger_{stage.value}") in edges
    # The chain: each stage's gate waits for the previous stage's trigger.
    assert ("trigger_sync", "enabled_meta") in edges
    assert ("trigger_meta", "enabled_labels") in edges
    assert ("trigger_labels", "enabled_media") in edges
    tasks_by_id = {task["task_id"]: task for task in payload["master"]["tasks"]}
    assert tasks_by_id["trigger_meta"]["deferred"] is True
    assert tasks_by_id["enabled_meta"]["deferred"] is False


def test_graph_sub_dag_shape_is_plan_fan_out_gate(
    runtime_free_cwd: Path, tmp_path: Path, unbuilt_assets_dir: Path
) -> None:
    data_root = tmp_path / "bare-root"
    data_root.mkdir()
    payload = _client_over(data_root, unbuilt_assets_dir).get("/api/v1/pipeline/graph").json()
    stages_by_name = {stage["stage"]: stage for stage in payload["stages"]}
    meta = stages_by_name["meta"]
    assert [task["task_id"] for task in meta["dag"]["tasks"]] == [
        "plan",
        "process_batch",
        "quarantine_budget_gate",
    ]
    assert meta["dag"]["edges"] == [
        ["plan", "process_batch"],
        ["process_batch", "quarantine_budget_gate"],
    ]
    mapped_tasks = [task["task_id"] for task in meta["dag"]["tasks"] if task["mapped"]]
    assert mapped_tasks == ["process_batch"]
    # Only meta gates on the quarantine budget; the others on errors alone.
    assert stages_by_name["labels"]["gate_task_id"] == "enabled_labels"
    assert [task["task_id"] for task in stages_by_name["labels"]["dag"]["tasks"]][-1] == (
        "error_budget_gate"
    )
    # The master's gate/trigger ids are named per stage, and the profiles that
    # enable each stage ride along for the lane header.
    assert stages_by_name["labels"]["trigger_task_id"] == "trigger_labels"
    assert set(stages_by_name["labels"]["enabling_profiles"]) == {
        name for name, stages in RUN_PROFILES.items() if Stage.LABELS in stages
    }


def test_graph_user_step_tiers_match_the_app_ordering(
    runtime_free_cwd: Path, tmp_path: Path, unbuilt_assets_dir: Path
) -> None:
    pipeline_file = _written_pipeline_file(tmp_path, TIERED_PIPELINE_SOURCE)
    data_root = tmp_path / "bare-root"
    data_root.mkdir()
    payload = (
        _client_over(data_root, unbuilt_assets_dir, pipeline=str(pipeline_file))
        .get("/api/v1/pipeline/graph")
        .json()
    )
    stages_by_name = {stage["stage"]: stage for stage in payload["stages"]}
    meta_steps = stages_by_name["meta"]["user_steps"]
    # Cheap-first: the tier-1 check leads even though it registered last, and
    # tier 2 keeps registration order (requires before uses).
    assert [(step["name"], step["tier"]) for step in meta_steps] == [
        ("cheap_check", 1),
        ("needs_channel", 2),
        ("needs_endpoint", 2),
    ]
    needs_channel = next(step for step in meta_steps if step["name"] == "needs_channel")
    assert needs_channel["requires"] == ["/camera/wrist"]
    assert needs_channel["uses"] is None
    assert needs_channel["critical"] is True
    assert needs_channel["kind"] == "check"
    assert needs_channel["version"]

    labels_steps = stages_by_name["labels"]["user_steps"]
    assert [(step["name"], step["tier"]) for step in labels_steps] == [
        ("cheap_label", 1),
        ("rich_caption", 2),
    ]
    assert next(step for step in labels_steps if step["name"] == "rich_caption")["uses"] == "vlm"
    # Enrichments are never critical (only checks carry the gate flag).
    assert all(step["critical"] is False for step in labels_steps)
    # Sync and media carry no user-registered steps at all.
    assert stages_by_name["sync"]["user_steps"] == []
    assert stages_by_name["media"]["user_steps"] == []


def test_graph_tier_derivation_matches_the_engines_own_ordering(
    runtime_free_cwd: Path, tmp_path: Path, unbuilt_assets_dir: Path
) -> None:
    """The payload's order IS App._ordered_checks' order, not a lookalike."""
    from hflow import import_pipeline_application

    pipeline_file = _written_pipeline_file(tmp_path, TIERED_PIPELINE_SOURCE)
    data_root = tmp_path / "bare-root"
    data_root.mkdir()
    payload = (
        _client_over(data_root, unbuilt_assets_dir, pipeline=str(pipeline_file))
        .get("/api/v1/pipeline/graph")
        .json()
    )
    application = import_pipeline_application(str(pipeline_file))
    stages_by_name = {stage["stage"]: stage for stage in payload["stages"]}
    assert [step["name"] for step in stages_by_name["meta"]["user_steps"]] == [
        registered.name for registered in application._ordered_checks()
    ]
    assert [step["name"] for step in stages_by_name["labels"]["user_steps"]] == [
        registered.name for registered in application._ordered_enrichments()
    ]


def test_the_pipeline_page_and_the_graph_describe_one_pipeline_the_same_way(
    runtime_free_cwd: Path, tmp_path: Path, unbuilt_assets_dir: Path
) -> None:
    """Steps are served in EXECUTION order, not registration order.

    The graph is the one owner of stage grouping (the pipeline page serves no
    lanes), so this pins the property that owner must hold: the cheap tier
    runs first. The fixture registers its tier-2 checks FIRST, so an endpoint
    that echoed registration order would fail here.
    """
    pipeline_file = _written_pipeline_file(tmp_path, TIERED_PIPELINE_SOURCE)
    data_root = tmp_path / "bare-root"
    data_root.mkdir()
    client = _client_over(data_root, unbuilt_assets_dir, pipeline=str(pipeline_file))
    graph_steps_by_stage = {
        stage["stage"]: stage["user_steps"]
        for stage in client.get("/api/v1/pipeline/graph").json()["stages"]
    }
    meta_steps = graph_steps_by_stage["meta"]
    assert [step["name"] for step in meta_steps] == [
        "cheap_check",
        "needs_channel",
        "needs_endpoint",
    ]
    assert [step["tier"] for step in meta_steps] == [1, 2, 2]


def test_graph_quarantine_gate_lists_exactly_the_critical_checks(
    runtime_free_cwd: Path, tmp_path: Path, unbuilt_assets_dir: Path
) -> None:
    pipeline_file = _written_pipeline_file(tmp_path, TIERED_PIPELINE_SOURCE)
    data_root = tmp_path / "bare-root"
    data_root.mkdir()
    payload = (
        _client_over(data_root, unbuilt_assets_dir, pipeline=str(pipeline_file))
        .get("/api/v1/pipeline/graph")
        .json()
    )
    gate = payload["quarantine_gate"]
    assert gate["from_stage"] == "meta"
    assert gate["to_stages"] == ["labels", "media"]
    assert sorted(gate["critical_step_names"]) == ["cheap_check", "needs_channel"]
    # The explanation must describe what App.process actually does.
    assert "quarantines the episode" in gate["explanation"]
    assert "skipped" in gate["explanation"]
    assert "never a deletion" in gate["explanation"]


def test_graph_quarantine_gate_is_honest_when_nothing_is_critical(
    runtime_free_cwd: Path, tmp_path: Path, unbuilt_assets_dir: Path
) -> None:
    pipeline_file = _written_pipeline_file(tmp_path, NO_CRITICAL_PIPELINE_SOURCE)
    data_root = tmp_path / "bare-root"
    data_root.mkdir()
    payload = (
        _client_over(data_root, unbuilt_assets_dir, pipeline=str(pipeline_file))
        .get("/api/v1/pipeline/graph")
        .json()
    )
    gate = payload["quarantine_gate"]
    assert gate["critical_step_names"] == []
    assert "no critical checks" in gate["explanation"]


def test_graph_sync_engine_step_reports_a_transform_override(
    runtime_free_cwd: Path, tmp_path: Path, unbuilt_assets_dir: Path
) -> None:
    pipeline_file = _written_pipeline_file(
        tmp_path,
        """import hflow

app = hflow.App("override-demo")


@app.transform
def transform(source_path, destination_path, config):
    raise NotImplementedError


@app.derive("/derived/speed")
def speed(episode):
    raise NotImplementedError
""",
    )
    data_root = tmp_path / "bare-root"
    data_root.mkdir()
    payload = (
        _client_over(data_root, unbuilt_assets_dir, pipeline=str(pipeline_file))
        .get("/api/v1/pipeline/graph")
        .json()
    )
    sync_stage = next(stage for stage in payload["stages"] if stage["stage"] == "sync")
    summary = sync_stage["engine_steps"][0]["summary"]
    assert "transform override" in summary
    assert "1 registered derived channel" in summary


def test_graph_serves_the_stage_display_copy(
    runtime_free_cwd: Path, tmp_path: Path, unbuilt_assets_dir: Path
) -> None:
    data_root = tmp_path / "bare-root"
    data_root.mkdir()
    payload = _client_over(data_root, unbuilt_assets_dir).get("/api/v1/pipeline/graph").json()
    titles = {stage["stage"]: stage["title"] for stage in payload["stages"]}
    assert titles == {
        "sync": "Transform & sync",
        "meta": "Metadata",
        "labels": "Labels & artifacts",
        "media": "Media",
    }
    assert all(stage["description"] for stage in payload["stages"])
