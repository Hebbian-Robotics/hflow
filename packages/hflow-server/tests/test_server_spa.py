"""SPA serving: built assets, client-route fallback, placeholder, traversal."""

from pathlib import Path

import hflow_server
import pytest
from fastapi.testclient import TestClient
from hflow_server import ServerSettings, create_app
from hflow_server.server import ASSETS_ENVIRONMENT_VARIABLE, _assets_directory
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
    settings = ServerSettings(
        data_root=str(populated_workspace.data_root),
        assets_dir=built_assets_directory,
    )
    return TestClient(create_app(settings))


def test_placeholder_page_when_no_frontend_is_installed(api: TestClient) -> None:
    """No bundle is not an error: the API is the product, a UI is a client."""
    response = api.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "No frontend bundle" in response.text
    assert "/api/v1" in response.text  # the API stays discoverable
    assert "/api/openapi.json" in response.text  # ...and so does how to build against it
    assert "HFLOW_UI_ASSETS" in response.text  # how to serve your own


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


def test_the_assets_environment_override_serves_a_local_build(
    populated_workspace: PopulatedWorkspace,
    built_assets_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The lane the placeholder page tells frontend developers to use: no
    # assets_dir pinned, HFLOW_UI_ASSETS pointing at a `pnpm build` output.
    monkeypatch.setenv(ASSETS_ENVIRONMENT_VARIABLE, str(built_assets_directory))
    client = TestClient(create_app(ServerSettings(data_root=str(populated_workspace.data_root))))
    assert "SPA INDEX" in client.get("/").text


def test_a_pinned_assets_directory_beats_the_environment_override(
    populated_workspace: PopulatedWorkspace,
    built_assets_directory: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Precedence matters because `hflow serve` never pins: an override left in a
    # developer's shell must not silently outrank an explicit setting.
    pinned_directory = tmp_path / "pinned"
    pinned_directory.mkdir()
    (pinned_directory / "index.html").write_text("<html><body>PINNED INDEX</body></html>")
    monkeypatch.setenv(ASSETS_ENVIRONMENT_VARIABLE, str(built_assets_directory))
    client = TestClient(
        create_app(
            ServerSettings(
                data_root=str(populated_workspace.data_root),
                assets_dir=pinned_directory,
            )
        )
    )
    assert "PINNED INDEX" in client.get("/").text


def test_packaged_assets_are_looked_up_inside_the_installed_package(
    populated_workspace: PopulatedWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The branch every installed-wheel user takes: nothing pinned, no override.

    hflow_server/static/ is a build artifact (gitignored, written by the frontend
    build), so this cannot assert a served page in either direction. What it
    can pin is the resolution: the importlib.resources anchor must land beside
    the installed hflow_server package, and the packaged branch must be taken
    exactly when that directory exists. A wrong anchor or a changed wheel
    layout degrades every real launch to the placeholder page, silently.
    """
    monkeypatch.delenv(ASSETS_ENVIRONMENT_VARIABLE, raising=False)
    settings = ServerSettings(data_root=str(populated_workspace.data_root))
    packaged_static = Path(str(hflow_server.__file__)).parent / "static"

    assert _assets_directory(settings) == (packaged_static if packaged_static.is_dir() else None)


def test_unknown_api_paths_are_json_404s(api: TestClient) -> None:
    response = api.get("/api/v1/definitely-not-a-route")
    assert response.status_code == 404
    assert "application/json" in response.headers["content-type"]
    assert response.json()["detail"]
