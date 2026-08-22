"""GET /api/v1/config and /api/v1/health."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from hflow_ui import UiSettings, create_app
from ui_test_fixtures import PopulatedWorkspace

from hflow.runtime import RuntimeConfig, render_bundle
from hflow.workspace import Workspace


@pytest.fixture()
def no_ambient_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No ./runtime fallback in cwd and no remote environment exported."""
    working_directory = tmp_path / "config-cwd"
    working_directory.mkdir()
    monkeypatch.chdir(working_directory)
    monkeypatch.delenv("HFLOW_AIRFLOW_URL", raising=False)


def test_health_reports_ok(api: TestClient) -> None:
    response = api.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_config_reports_local_mode_and_capabilities(
    api: TestClient, populated_workspace: PopulatedWorkspace, no_ambient_runtime: None
) -> None:
    payload = api.get("/api/v1/config").json()
    assert payload["mode"] == "local"
    assert payload["read_only"] is False  # the default server accepts writes
    assert isinstance(payload["hflow_version"], str) and payload["hflow_version"]
    assert payload["hflow_ui_version"] == "0.1.0"
    assert payload["data_root"] == str(populated_workspace.data_root)
    assert payload["capabilities"] == {
        "catalog": True,
        "media": True,
        "runtime": False,  # no bundle rendered and no HFLOW_AIRFLOW_URL exported
        "pipeline": False,  # no --pipeline configured
    }
    # The trigger form's vocabularies come from the server, never hardcoded.
    assert "full" in payload["run_profiles"]
    assert payload["ingest_modes"] == ["batch", "online"]


def test_config_does_not_restate_the_airflow_deep_link_base(
    tmp_path: Path, unbuilt_assets_dir: Path, no_ambient_runtime: None
) -> None:
    # /runtime/status is the ONE owner of the runtime's addressing facts, the
    # web URL included; config only reports whether a runtime is addressed.
    data_root = tmp_path / "data"
    pipeline_file = tmp_path / "demo_pipeline.py"
    pipeline_file.write_text("import hflow\n\napp = hflow.App('demo', data_root='/tmp/x')\n")
    render_bundle(
        RuntimeConfig(pipeline_file=pipeline_file, data_root=data_root), data_root / "runtime"
    )
    settings = UiSettings(data_root=str(data_root), token=None, assets_dir=unbuilt_assets_dir)
    payload = TestClient(create_app(settings)).get("/api/v1/config").json()
    # Configured, not necessarily reachable: nothing is running here.
    assert payload["capabilities"]["runtime"] is True
    assert "airflow_web_url" not in payload


def test_config_runtime_capability_from_the_remote_environment(
    tmp_path: Path,
    unbuilt_assets_dir: Path,
    no_ambient_runtime: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HFLOW_AIRFLOW_URL", "https://workspace.example.com")
    data_root = tmp_path / "bare-root"
    data_root.mkdir()
    settings = UiSettings(data_root=str(data_root), token=None, assets_dir=unbuilt_assets_dir)
    payload = TestClient(create_app(settings)).get("/api/v1/config").json()
    assert payload["capabilities"]["runtime"] is True


def test_config_reports_the_read_only_setting(read_only_api: TestClient) -> None:
    assert read_only_api.get("/api/v1/config").json()["read_only"] is True


def test_config_never_mints_workspace_identity(
    api: TestClient, populated_workspace: PopulatedWorkspace
) -> None:
    assert api.get("/api/v1/config").json()["workspace_id"] is None
    assert not (populated_workspace.data_root / "workspace.json").exists()


def test_config_reports_missing_catalog(empty_workspace_api: TestClient) -> None:
    payload = empty_workspace_api.get("/api/v1/config").json()
    assert payload["capabilities"]["catalog"] is False
    assert payload["capabilities"]["media"] is True  # local root: serving is possible


def test_config_reports_a_minted_workspace_identity(tmp_path: Path) -> None:
    # The TEST mints the identity; the server itself never does.
    minted_identity = Workspace.parse(tmp_path).ensure_identity()
    client = TestClient(create_app(UiSettings(data_root=str(tmp_path), token=None)))
    payload = client.get("/api/v1/config").json()
    assert payload["workspace_id"] == minted_identity.workspace_id


def test_mutating_methods_are_rejected(api: TestClient) -> None:
    # Read-only surface: no route accepts writes.
    assert api.post("/api/v1/episodes").status_code == 405
    assert api.delete("/api/v1/config").status_code == 405
