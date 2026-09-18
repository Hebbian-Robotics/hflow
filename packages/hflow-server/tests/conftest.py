"""Shared fixtures: real populated workspaces and TestClients over them."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from hflow_server import ServerSettings, create_app
from ui_test_fixtures import PopulatedWorkspace, build_populated_workspace

from hflow.catalog import Catalog
from hflow.storage import BucketStorageRoot, StorageRoot
from hflow.storage import parse_storage_root as real_parse_storage_root


@pytest.fixture(scope="session")
def populated_workspace(tmp_path_factory: pytest.TempPathFactory) -> PopulatedWorkspace:
    return build_populated_workspace(tmp_path_factory)


@pytest.fixture(scope="session")
def unbuilt_assets_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An empty assets directory, for the clients that assert on served pages.

    The packaged default (hflow_server/static/) holds the built SPA on a machine
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
    settings = ServerSettings(
        data_root=str(populated_workspace.data_root), assets_dir=unbuilt_assets_dir
    )
    return TestClient(create_app(settings))


@pytest.fixture(scope="session")
def read_only_api(populated_workspace: PopulatedWorkspace, unbuilt_assets_dir: Path) -> TestClient:
    """A client whose server runs read-only: every write endpoint must 403.

    Session-scoped over the shared workspace on purpose -- a read-only server
    refuses before touching anything, so it cannot dirty the fixture.
    """
    settings = ServerSettings(
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
    settings = ServerSettings(
        data_root=str(writable_workspace.data_root), assets_dir=unbuilt_assets_dir
    )
    return TestClient(create_app(settings))


@pytest.fixture(scope="session")
def empty_workspace_api(
    tmp_path_factory: pytest.TempPathFactory, unbuilt_assets_dir: Path
) -> TestClient:
    """A client over a data root that has no catalog at all."""
    empty_root = tmp_path_factory.mktemp("ui-empty-root")
    settings = ServerSettings(data_root=str(empty_root), assets_dir=unbuilt_assets_dir)
    return TestClient(create_app(settings))


@pytest.fixture(scope="session")
def empty_catalog_api(
    tmp_path_factory: pytest.TempPathFactory, unbuilt_assets_dir: Path
) -> TestClient:
    """A client over a catalog that exists but holds zero episodes."""
    data_root = tmp_path_factory.mktemp("ui-empty-catalog-root")
    Catalog(data_root / "catalog")
    settings = ServerSettings(data_root=str(data_root), assets_dir=unbuilt_assets_dir)
    return TestClient(create_app(settings))


@pytest.fixture()
def bucket_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[str, BucketStorageRoot]:
    """Real bucket I/O over file://; only URL resolution is redirected.

    Public file:// settings resolve to LocalStorageRoot, so a sentinel URL
    selects a BucketStorageRoot without credentials or cloud requests. All
    catalog, sidecar and manifest operations run through the real backend.
    """
    pytest.importorskip("obstore", reason="bucket tests need the hflow[bucket] extra")
    remote_directory = tmp_path / "remote"
    remote_directory.mkdir()
    root = BucketStorageRoot(remote_directory.as_uri(), mirror=tmp_path / "mirror")
    sentinel = "gs://hflow-test-bucket/workspace"

    def parse_as_test_bucket(value: str | Path | StorageRoot) -> StorageRoot:
        return root if value == sentinel else real_parse_storage_root(value)

    for module in (
        "hflow.storage",
        "hflow.workspace",
        "hflow_server._settings",
        "hflow_server._sidecar",
    ):
        monkeypatch.setattr(f"{module}.parse_storage_root", parse_as_test_bucket)
    return sentinel, root
