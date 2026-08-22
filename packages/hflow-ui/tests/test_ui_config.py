"""GET /api/v1/config and /api/v1/health."""

from pathlib import Path

from fastapi.testclient import TestClient
from hflow_ui import UiSettings, create_app
from ui_test_fixtures import PopulatedWorkspace

from hflow.workspace import Workspace


def test_health_reports_ok(api: TestClient) -> None:
    response = api.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_config_reports_local_read_only_mode(
    api: TestClient, populated_workspace: PopulatedWorkspace
) -> None:
    payload = api.get("/api/v1/config").json()
    assert payload["mode"] == "local"
    assert payload["read_only"] is True
    assert isinstance(payload["hflow_version"], str) and payload["hflow_version"]
    assert payload["hflow_ui_version"] == "0.1.0"
    assert payload["data_root"] == str(populated_workspace.data_root)
    assert payload["capabilities"] == {"catalog": True, "media": True, "runtime": False}


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
