"""Bucket capability promises exercised through HTTP and durable object bytes."""

import json
from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient
from hflow_server import ServerSettings, create_app
from ui_test_fixtures import PopulatedWorkspace

from hflow.format import CATALOG_FORMAT_VERSION
from hflow.storage import BucketStorageRoot, LocalStorageRoot

OK_CUT_SQL = "SELECT episode_id FROM episodes WHERE status = 'ok'"


def test_bucket_curation_persists_queries_and_pins_but_cannot_serve_bytes(
    bucket_workspace: tuple[str, BucketStorageRoot],
    populated_workspace: PopulatedWorkspace,
) -> None:
    sentinel, root = bucket_workspace
    for catalog_file in (populated_workspace.data_root / "catalog").rglob("*"):
        if catalog_file.is_file():
            root.write_bytes(
                catalog_file.relative_to(populated_workspace.data_root).as_posix(),
                catalog_file.read_bytes(),
            )
    settings = ServerSettings(data_root=sentinel)
    with TestClient(create_app(settings)) as client:
        for endpoint in ("preview", "report"):
            response = client.post(f"/api/v1/curation/{endpoint}", json={"sql": OK_CUT_SQL})
            assert response.status_code == 200, response.text
            assert response.json()["row_count"] == 3
        created = client.post("/api/v1/queries", json={"name": "ok cut", "sql": OK_CUT_SQL})
        assert created.status_code == 200, created.text
        query = created.json()
        pinned = client.post("/api/v1/curation/pin", json={"name": "ok cut", "sql": OK_CUT_SQL})
        assert pinned.status_code == 200, pinned.text
        entry = pinned.json()
        assert entry["row_count"] == 3
        assert entry["total_episodes"] == 4
        assert {item["check_name"] for item in entry["coverage"]} == {
            "joint_check",
            "media/contact_sheet",
            "camera_blackout",
        }

    # A fresh store and mirror cannot satisfy these reads from the writer's
    # cache. The real published Parquet must contain exactly the selected IDs.
    reader = BucketStorageRoot(root.url, mirror=root.mirror.parent / "fresh-reader")
    state = json.loads(reader.read_bytes("curation/state.json"))
    assert state["saved_queries"] == [query]
    assert state["manifests"] == [entry]
    with duckdb.connect() as connection:
        rows = connection.execute(
            "SELECT episode_id FROM read_parquet(?)", [str(reader.fetch(entry["manifest_path"]))]
        ).fetchall()
    assert {row[0] for row in rows} == {
        populated_workspace.ok_episode_id,
        populated_workspace.escaping_episode_id,
        populated_workspace.minimal_episode_id,
    }

    with TestClient(create_app(settings)) as restarted:
        assert restarted.get("/api/v1/queries").json()["queries"] == [query]
        assert restarted.get("/api/v1/manifests").json()["manifests"] == [entry]
        changed = restarted.put(f"/api/v1/queries/{query['id']}", json={"name": "renamed"})
        assert changed.status_code == 200
        assert json.loads(reader.read_bytes("curation/state.json"))["saved_queries"] == [
            changed.json()
        ]
        assert restarted.delete(f"/api/v1/queries/{query['id']}").status_code == 204
        assert json.loads(reader.read_bytes("curation/state.json")) == {
            "state_version": 1,
            "saved_queries": [],
            "manifests": [entry],
        }

        # Existing HTTP limitation: download fetches the manifest successfully,
        # then the shared media containment helper refuses a bucket data root.
        download = restarted.get(f"/api/v1/manifests/{entry['id']}/download")
        media = restarted.get(
            f"/api/v1/episodes/{populated_workspace.ok_episode_id}/media/wrist_cam"
        )
        for response in (download, media):
            assert response.status_code == 501, response.text
            assert "local data root" in response.json()["detail"]
        assert root.fetch(entry["manifest_path"]).read_bytes() == reader.read_bytes(
            entry["manifest_path"]
        )

        # Assert the changed flag after the actual writes, so mutation proof
        # establishes the original routes work even when config denies support.
        config = restarted.get("/api/v1/config")
        assert config.status_code == 200
        assert config.json()["capabilities"]["catalog"] is True
        assert config.json()["capabilities"]["media"] is False
        assert config.json()["capabilities"]["curation"] is True


@pytest.mark.parametrize("backend", ["local", "bucket"])
@pytest.mark.parametrize("catalog_state", ["missing", "empty", "incompatible"])
def test_curation_storage_support_is_separate_from_catalog_readiness(
    backend: str,
    catalog_state: str,
    bucket_workspace: tuple[str, BucketStorageRoot],
    tmp_path: Path,
) -> None:
    sentinel, bucket_root = bucket_workspace
    root = bucket_root if backend == "bucket" else LocalStorageRoot(tmp_path / "local")
    data_root = sentinel if backend == "bucket" else str(root)
    if catalog_state != "missing":
        version = CATALOG_FORMAT_VERSION if catalog_state == "empty" else "unsupported"
        root.write_bytes("catalog/format_version", (version + "\n").encode())
    with TestClient(create_app(ServerSettings(data_root=data_root))) as client:
        saved = client.post("/api/v1/queries", json={"name": "durable", "sql": OK_CUT_SQL})
        assert saved.status_code == 200, saved.text
        assert json.loads(root.read_bytes("curation/state.json"))["saved_queries"] == [saved.json()]
        for endpoint in ("preview", "report", "pin"):
            response = client.post(
                f"/api/v1/curation/{endpoint}", json={"name": "empty cut", "sql": OK_CUT_SQL}
            )
            expected_status = {"missing": 404, "empty": 200, "incompatible": 409}[catalog_state]
            assert response.status_code == expected_status, response.text
            if catalog_state == "empty":
                assert response.json()["row_count"] == 0
        manifests = client.get("/api/v1/manifests").json()["manifests"]
        assert len(manifests) == (1 if catalog_state == "empty" else 0)
        config = client.get("/api/v1/config").json()
        assert config["capabilities"]["catalog"] is (catalog_state == "empty")
        assert config["capabilities"]["media"] is (backend == "local")
        assert config["capabilities"]["curation"] is True


def test_bucket_read_only_policy_remains_separate_from_storage_support(
    bucket_workspace: tuple[str, BucketStorageRoot],
) -> None:
    sentinel, root = bucket_workspace
    with TestClient(create_app(ServerSettings(data_root=sentinel, read_only=True))) as client:
        responses = [
            client.post("/api/v1/queries", json={"name": "no write", "sql": "SELECT 1"}),
            client.put("/api/v1/queries/absent", json={"name": "no write"}),
            client.delete("/api/v1/queries/absent"),
            client.post("/api/v1/curation/pin", json={"name": "no pin", "sql": "SELECT 1"}),
        ]
        for response in responses:
            assert response.status_code == 403, response.text
            assert "read-only" in response.json()["detail"]
        assert not root.exists("curation/state.json")
        assert client.get("/api/v1/queries").json() == {"queries": []}
        config = client.get("/api/v1/config").json()
        assert config["read_only"] is True
        assert config["capabilities"]["curation"] is True
        assert config["capabilities"]["media"] is False
