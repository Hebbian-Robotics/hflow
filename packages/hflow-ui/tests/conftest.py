"""Shared fixtures: real populated workspaces and TestClients over them."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from hflow_ui import UiSettings, create_app
from ui_test_fixtures import PopulatedWorkspace, build_populated_workspace

from hflow.catalog import Catalog


@pytest.fixture(scope="session")
def populated_workspace(tmp_path_factory: pytest.TempPathFactory) -> PopulatedWorkspace:
    return build_populated_workspace(tmp_path_factory)


@pytest.fixture(scope="session")
def unbuilt_assets_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An empty assets directory, for the clients that assert on served pages.

    The packaged default (hflow_ui/static/) holds the built SPA on a machine
    that has run the frontend build, and nothing on one that has not, so any
    test that asserts on a served PAGE pins assets_dir -- every client fixture
    below does, through this fixture. A client built inline inside an API test
    needs no pin: it only ever requests /api paths, which never consult the
    assets directory.
    """
    return tmp_path_factory.mktemp("ui-no-assets")


@pytest.fixture(scope="session")
def api(populated_workspace: PopulatedWorkspace, unbuilt_assets_dir: Path) -> TestClient:
    """A client over the populated root; the server authenticates nobody."""
    settings = UiSettings(
        data_root=str(populated_workspace.data_root), assets_dir=unbuilt_assets_dir
    )
    return TestClient(create_app(settings))


@pytest.fixture(scope="session")
def read_only_api(populated_workspace: PopulatedWorkspace, unbuilt_assets_dir: Path) -> TestClient:
    """A client whose server runs read-only: every write endpoint must 403.

    Session-scoped over the shared workspace on purpose -- a read-only server
    refuses before touching anything, so it cannot dirty the fixture.
    """
    settings = UiSettings(
        data_root=str(populated_workspace.data_root),
        assets_dir=unbuilt_assets_dir,
        read_only=True,
    )
    return TestClient(create_app(settings))


@pytest.fixture()
def writable_workspace(tmp_path_factory: pytest.TempPathFactory) -> PopulatedWorkspace:
    """A per-test workspace for tests that WRITE (pins, saved queries)."""
    return build_populated_workspace(tmp_path_factory)


@pytest.fixture()
def writable_api(writable_workspace: PopulatedWorkspace, unbuilt_assets_dir: Path) -> TestClient:
    settings = UiSettings(
        data_root=str(writable_workspace.data_root), assets_dir=unbuilt_assets_dir
    )
    return TestClient(create_app(settings))


@pytest.fixture(scope="session")
def empty_workspace_api(
    tmp_path_factory: pytest.TempPathFactory, unbuilt_assets_dir: Path
) -> TestClient:
    """A client over a data root that has no catalog at all."""
    empty_root = tmp_path_factory.mktemp("ui-empty-root")
    settings = UiSettings(data_root=str(empty_root), assets_dir=unbuilt_assets_dir)
    return TestClient(create_app(settings))


@pytest.fixture(scope="session")
def empty_catalog_api(
    tmp_path_factory: pytest.TempPathFactory, unbuilt_assets_dir: Path
) -> TestClient:
    """A client over a catalog that exists but holds zero episodes."""
    data_root = tmp_path_factory.mktemp("ui-empty-catalog-root")
    Catalog(data_root / "catalog")
    settings = UiSettings(data_root=str(data_root), assets_dir=unbuilt_assets_dir)
    return TestClient(create_app(settings))
