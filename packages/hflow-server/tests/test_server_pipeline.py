"""GET /api/v1/pipeline plus the config capability behind it.

Every test imports a REAL tiny pipeline file written into tmp_path; the apps
inside are constructed WITHOUT data_root (environment resolution), so the
startup import is side-effect-free.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from hflow_server import ServerSettings, create_app
from ui_test_fixtures import PopulatedWorkspace

WORKING_PIPELINE_SOURCE = """import hflow

app = hflow.App("ui-demo")


@app.check(name="joint_check", critical=True)
def joint_check(episode):
    return hflow.CheckResult(measurements={"max_velocity": 1.0}, verdict=True)


@app.enrich(name="caption")
def caption(episode):
    return hflow.EnrichmentResult(labels={"caption": "hello"})
"""

RAISING_PIPELINE_SOURCE = 'raise RuntimeError("boom at import")\n'


def _client_over(data_root: Path, assets_dir: Path, *, pipeline: str | None) -> TestClient:
    settings = ServerSettings(data_root=str(data_root), assets_dir=assets_dir, pipeline=pipeline)
    return TestClient(create_app(settings))


def _written_pipeline_file(tmp_path: Path, source: str) -> Path:
    pipeline_file = tmp_path / "ui_pipeline.py"
    pipeline_file.write_text(source)
    return pipeline_file


def test_pipeline_page_reports_manifest_lanes_observed_and_stale(
    populated_workspace: PopulatedWorkspace, unbuilt_assets_dir: Path, tmp_path: Path
) -> None:
    pipeline_file = _written_pipeline_file(tmp_path, WORKING_PIPELINE_SOURCE)
    client = _client_over(
        populated_workspace.data_root, unbuilt_assets_dir, pipeline=str(pipeline_file)
    )
    response = client.get("/api/v1/pipeline")
    assert response.status_code == 200
    payload = response.json()

    manifest = payload["manifest"]
    assert manifest["pipeline_name"] == "ui-demo"
    assert [check["name"] for check in manifest["checks"]] == ["joint_check"]
    assert manifest["checks"][0]["critical"] is True
    assert manifest["checks"][0]["kind"] == "check"
    assert manifest["checks"][0]["version"]
    assert [enrichment["name"] for enrichment in manifest["enrichments"]] == ["caption"]

    # No stage lanes here: /pipeline/graph is the one owner of stage grouping,
    # so this page cannot disagree with it about which steps run where.
    assert "stages" not in payload

    observed_by_identity = {
        (row["check_name"], row["check_version"]): row for row in payload["observed"]
    }
    joint_check_observed = observed_by_identity[("joint_check", "v1")]
    assert joint_check_observed["run_count"] == 2  # the ok episode's two appends
    assert joint_check_observed["first_seen"] <= joint_check_observed["last_seen"]
    assert observed_by_identity[("camera_blackout", "v1")]["run_count"] == 1

    # The fixture stamps every episode with another pipeline_version, so all
    # four source recordings are stale against this App's current versions.
    assert payload["stale"] == {
        "pipeline_version": manifest["pipeline_version"],
        "count": 4,
    }


def test_pipeline_capability_and_empty_catalog(tmp_path: Path, unbuilt_assets_dir: Path) -> None:
    pipeline_file = _written_pipeline_file(tmp_path, WORKING_PIPELINE_SOURCE)
    data_root = tmp_path / "no-catalog-root"
    data_root.mkdir()
    client = _client_over(data_root, unbuilt_assets_dir, pipeline=str(pipeline_file))
    assert client.get("/api/v1/config").json()["capabilities"]["pipeline"] is True
    payload = client.get("/api/v1/pipeline").json()
    assert payload["manifest"]["pipeline_name"] == "ui-demo"
    # No catalog: nothing observed, staleness unknowable -- not an error.
    assert payload["observed"] == []
    assert payload["stale"] is None


def test_pipeline_spec_selects_a_named_app_variable(
    tmp_path: Path, unbuilt_assets_dir: Path
) -> None:
    pipeline_file = _written_pipeline_file(
        tmp_path, 'import hflow\n\nmy_app = hflow.App("named-app")\n'
    )
    data_root = tmp_path / "root"
    data_root.mkdir()
    client = _client_over(data_root, unbuilt_assets_dir, pipeline=f"{pipeline_file}:my_app")
    payload = client.get("/api/v1/pipeline").json()
    assert payload["manifest"]["pipeline_name"] == "named-app"


def test_pipeline_import_failure_is_remembered_not_fatal(
    tmp_path: Path, unbuilt_assets_dir: Path
) -> None:
    pipeline_file = _written_pipeline_file(tmp_path, RAISING_PIPELINE_SOURCE)
    data_root = tmp_path / "root"
    data_root.mkdir()
    client = _client_over(data_root, unbuilt_assets_dir, pipeline=str(pipeline_file))
    # The server still boots and answers; the capability reports the failure.
    assert client.get("/api/v1/config").json()["capabilities"]["pipeline"] is False
    response = client.get("/api/v1/pipeline")
    assert response.status_code == 409
    assert "boom at import" in response.json()["detail"]


def test_pipeline_file_without_the_app_variable_is_a_409(
    tmp_path: Path, unbuilt_assets_dir: Path
) -> None:
    pipeline_file = _written_pipeline_file(tmp_path, "x = 1\n")
    data_root = tmp_path / "root"
    data_root.mkdir()
    client = _client_over(data_root, unbuilt_assets_dir, pipeline=str(pipeline_file))
    response = client.get("/api/v1/pipeline")
    assert response.status_code == 409
    assert "no hflow.App named 'app'" in response.json()["detail"]


def test_pipeline_unconfigured_is_a_409_naming_the_flag(api: TestClient) -> None:
    response = api.get("/api/v1/pipeline")
    assert response.status_code == 409
    assert "--pipeline" in response.json()["detail"]


def test_failed_startup_import_warns_on_stderr_naming_path_and_reason(
    tmp_path: Path, unbuilt_assets_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The operator asked for a pipeline and did not get one: the launch says
    so on stderr, naming the path and the reason, while still serving."""
    pipeline_file = _written_pipeline_file(tmp_path, RAISING_PIPELINE_SOURCE)
    data_root = tmp_path / "root"
    data_root.mkdir()
    client = _client_over(data_root, unbuilt_assets_dir, pipeline=str(pipeline_file))
    assert client.get("/api/v1/config").json()["capabilities"]["pipeline"] is False
    stderr_text = capsys.readouterr().err
    assert str(pipeline_file) in stderr_text
    assert "boom at import" in stderr_text
    assert "--pipeline" in stderr_text


def test_no_import_warning_for_a_working_or_unconfigured_launch(
    populated_workspace: PopulatedWorkspace,
    unbuilt_assets_dir: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Only a configured-and-failed import warns: a working pipeline and the
    ordinary no---pipeline launch are both silent."""
    working_file = _written_pipeline_file(tmp_path, WORKING_PIPELINE_SOURCE)
    client = _client_over(
        populated_workspace.data_root, unbuilt_assets_dir, pipeline=str(working_file)
    )
    assert client.get("/api/v1/pipeline").status_code == 200
    assert "could not be imported" not in capsys.readouterr().err

    unconfigured = _client_over(populated_workspace.data_root, unbuilt_assets_dir, pipeline=None)
    assert unconfigured.get("/api/v1/pipeline").status_code == 409
    assert "could not be imported" not in capsys.readouterr().err
