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
    """An empty assets directory, pinned explicitly.

    The packaged default (hflow_ui/static/) holds the built SPA on a machine
    that has run the frontend build, and nothing on one that has not -- so
    every fixture pins assets_dir to keep placeholder assertions independent
    of local build state.
    """
    return tmp_path_factory.mktemp("ui-no-assets")


@pytest.fixture(scope="session")
def api(populated_workspace: PopulatedWorkspace, unbuilt_assets_dir: Path) -> TestClient:
    """A tokenless client (auth disabled) over the populated root."""
    settings = UiSettings(
        data_root=str(populated_workspace.data_root), token=None, assets_dir=unbuilt_assets_dir
    )
    return TestClient(create_app(settings))


@pytest.fixture(scope="session")
def empty_workspace_api(
    tmp_path_factory: pytest.TempPathFactory, unbuilt_assets_dir: Path
) -> TestClient:
    """A client over a data root that has no catalog at all."""
    empty_root = tmp_path_factory.mktemp("ui-empty-root")
    settings = UiSettings(data_root=str(empty_root), token=None, assets_dir=unbuilt_assets_dir)
    return TestClient(create_app(settings))


@pytest.fixture(scope="session")
def empty_catalog_api(
    tmp_path_factory: pytest.TempPathFactory, unbuilt_assets_dir: Path
) -> TestClient:
    """A client over a catalog that exists but holds zero episodes."""
    data_root = tmp_path_factory.mktemp("ui-empty-catalog-root")
    Catalog(data_root / "catalog")
    settings = UiSettings(data_root=str(data_root), token=None, assets_dir=unbuilt_assets_dir)
    return TestClient(create_app(settings))
