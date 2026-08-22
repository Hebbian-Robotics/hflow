"""The <data_root>/ui/state.json sidecar: atomic writes, loud boundary parsing."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from hflow_ui import UiSettings, create_app
from ui_test_fixtures import PopulatedWorkspace


@pytest.fixture()
def bare_root(tmp_path: Path) -> Path:
    """A data root with no catalog: the sidecar endpoints need none."""
    return tmp_path


@pytest.fixture()
def bare_api(bare_root: Path) -> TestClient:
    return TestClient(create_app(UiSettings(data_root=str(bare_root))))


def _write_state_file(bare_root: Path, payload: str) -> Path:
    """Plant a sidecar file and return its path. Named for the write: calling
    this in an assertion would overwrite the very bytes under test."""
    state_file = bare_root / "ui" / "state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(payload)
    return state_file


def test_a_missing_state_file_reads_as_empty_state(bare_api: TestClient) -> None:
    assert bare_api.get("/api/v1/queries").json() == {"queries": []}
    assert bare_api.get("/api/v1/manifests").json() == {"manifests": []}


def test_a_corrupt_state_file_is_a_loud_500_naming_the_file(
    bare_api: TestClient, bare_root: Path
) -> None:
    state_file = _write_state_file(bare_root, "this is { not json")
    response = bare_api.get("/api/v1/queries")
    assert response.status_code == 500
    detail = response.json()["detail"]
    assert str(state_file) in detail
    assert "corrupt" in detail


def test_a_wrong_state_version_is_refused_loudly(bare_api: TestClient, bare_root: Path) -> None:
    state_file = _write_state_file(
        bare_root, json.dumps({"state_version": 2, "saved_queries": [], "manifests": []})
    )
    response = bare_api.get("/api/v1/manifests")
    # 409, matching the catalog's format-version refusal: the state is intact,
    # this build just cannot read its version -- not a fault of the server.
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert str(state_file) in detail
    assert "state_version" in detail
    assert "2" in detail


def test_a_malformed_entry_is_refused_loudly(bare_api: TestClient, bare_root: Path) -> None:
    _write_state_file(
        bare_root,
        json.dumps({"state_version": 1, "saved_queries": [{"id": "only-an-id"}], "manifests": []}),
    )
    response = bare_api.get("/api/v1/queries")
    assert response.status_code == 500
    assert "name" in response.json()["detail"]


def test_a_corrupt_sidecar_blocks_writes_too(bare_api: TestClient, bare_root: Path) -> None:
    state_file = _write_state_file(bare_root, "garbage")
    response = bare_api.post("/api/v1/queries", json={"name": "x", "sql": "SELECT 1"})
    assert response.status_code == 500
    # The refusal must leave the operator's file byte-for-byte intact: a
    # rewrite here would silently destroy their saved queries and manifest
    # registry, which is precisely what the unreadable state is protecting.
    assert state_file.read_text() == "garbage"
    assert [file.name for file in state_file.parent.iterdir()] == ["state.json"]


def test_writes_land_atomically_in_the_documented_shape(
    writable_api: TestClient, writable_workspace: PopulatedWorkspace
) -> None:
    created = writable_api.post(
        "/api/v1/queries", json={"name": "shape check", "sql": "SELECT 1"}
    ).json()
    state_file = writable_workspace.data_root / "ui" / "state.json"
    stored = json.loads(state_file.read_text())
    assert stored == {"state_version": 1, "saved_queries": [created], "manifests": []}
    # No temp debris: the write moved into place, it did not copy.
    assert [file.name for file in state_file.parent.iterdir()] == ["state.json"]
