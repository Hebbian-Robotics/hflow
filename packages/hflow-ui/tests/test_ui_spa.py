"""SPA serving: built assets, client-route fallback, placeholder, traversal."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from hflow_ui import UiSettings, create_app
from ui_test_fixtures import PopulatedWorkspace


@pytest.fixture()
def built_assets_directory(tmp_path: Path) -> Path:
    assets_directory = tmp_path / "dist"
    assets_directory.mkdir()
    (assets_directory / "index.html").write_text("<html><body>SPA INDEX</body></html>")
    (assets_directory / "assets").mkdir()
    (assets_directory / "assets" / "app.js").write_text("console.log('hflow');")
    # A secret OUTSIDE the assets tree, for the traversal test.
    (tmp_path / "secret.txt").write_text("do not serve this")
    return assets_directory


@pytest.fixture()
def spa_api(populated_workspace: PopulatedWorkspace, built_assets_directory: Path) -> TestClient:
    settings = UiSettings(
        data_root=str(populated_workspace.data_root),
        token=None,
        assets_dir=built_assets_directory,
    )
    return TestClient(create_app(settings))


def test_placeholder_page_when_the_frontend_is_not_built(api: TestClient) -> None:
    response = api.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "not built" in response.text
    assert "ui/" in response.text  # points at the frontend dev instructions
    assert "/api/v1" in response.text  # the API stays discoverable


def test_client_routes_get_the_placeholder_too(api: TestClient) -> None:
    response = api.get("/episodes/some-episode-id")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_index_is_served_at_the_root(spa_api: TestClient) -> None:
    response = spa_api.get("/")
    assert response.status_code == 200
    assert "SPA INDEX" in response.text


def test_assets_are_served_by_path(spa_api: TestClient) -> None:
    response = spa_api.get("/assets/app.js")
    assert response.status_code == 200
    assert "console.log" in response.text


def test_extensionless_paths_fall_back_to_index(spa_api: TestClient) -> None:
    response = spa_api.get("/episodes/abc123")
    assert response.status_code == 200
    assert "SPA INDEX" in response.text


def test_missing_asset_files_are_404_not_index(spa_api: TestClient) -> None:
    assert spa_api.get("/missing.png").status_code == 404


def test_path_traversal_out_of_the_assets_tree_is_refused(spa_api: TestClient) -> None:
    response = spa_api.get("/%2e%2e/secret.txt")
    assert response.status_code == 404
    assert "do not serve this" not in response.text


def test_unknown_api_paths_are_json_404s(api: TestClient) -> None:
    response = api.get("/api/v1/definitely-not-a-route")
    assert response.status_code == 404
    assert "application/json" in response.headers["content-type"]
    assert response.json()["detail"]
