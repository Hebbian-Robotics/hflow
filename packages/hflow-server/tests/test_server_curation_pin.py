"""POST /api/v1/curation/pin and /api/v1/manifests: immutable pinned cuts."""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import duckdb
import pytest
from fastapi.testclient import TestClient
from hflow_server import _curation
from ui_test_fixtures import PopulatedWorkspace

OK_CUT_SQL = "SELECT episode_id FROM episodes WHERE status = 'ok'"


def _pin(api: TestClient, name: str, sql: str = OK_CUT_SQL, description: str = "") -> dict:
    response = api.post(
        "/api/v1/curation/pin", json={"sql": sql, "name": name, "description": description}
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_pin_writes_the_manifest_and_a_registry_entry(
    writable_api: TestClient, writable_workspace: PopulatedWorkspace
) -> None:
    entry = _pin(writable_api, "Clean Fold Cut!", description="ok episodes only")
    assert entry["name"] == "Clean Fold Cut!"
    assert entry["description"] == "ok episodes only"
    assert entry["sql"] == OK_CUT_SQL
    assert entry["row_count"] == 3
    assert entry["total_episodes"] == 4
    assert len(entry["id"]) == 32  # uuid hex
    assert datetime.fromisoformat(entry["created_at"]).tzinfo is not None
    assert entry["manifest_path"].startswith("manifests/clean-fold-cut-")
    assert entry["manifest_path"].endswith(".parquet")
    assert {coverage["check_name"] for coverage in entry["coverage"]} == {
        "joint_check",
        "media/contact_sheet",
        "camera_blackout",
    }

    manifest_file = writable_workspace.data_root / entry["manifest_path"]
    assert manifest_file.is_file()
    # The pinned Parquet really is the cut: readable, with the cut's rows.
    (row_count,) = duckdb.connect().execute(
        "SELECT count(*) FROM read_parquet(?)", [str(manifest_file)]
    ).fetchone() or (0,)
    assert int(row_count) == 3

    listed = writable_api.get("/api/v1/manifests").json()["manifests"]
    assert [manifest["id"] for manifest in listed] == [entry["id"]]


def test_second_pin_with_the_same_name_gets_a_distinct_file(
    writable_api: TestClient, writable_workspace: PopulatedWorkspace
) -> None:
    first_entry = _pin(writable_api, "nightly cut")
    second_entry = _pin(writable_api, "nightly cut")
    assert first_entry["manifest_path"] != second_entry["manifest_path"]
    assert (writable_workspace.data_root / first_entry["manifest_path"]).is_file()
    assert (writable_workspace.data_root / second_entry["manifest_path"]).is_file()
    # The registry lists newest first.
    listed_ids = [
        manifest["id"] for manifest in writable_api.get("/api/v1/manifests").json()["manifests"]
    ]
    assert listed_ids == [second_entry["id"], first_entry["id"]]


def test_a_filename_collision_is_refused_never_overwritten(
    writable_api: TestClient,
    writable_workspace: PopulatedWorkspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_curation, "_manifest_timestamp", lambda: "20260821T000000000000Z")
    entry = _pin(writable_api, "pinned once")
    collision = writable_api.post(
        "/api/v1/curation/pin", json={"sql": OK_CUT_SQL, "name": "pinned once"}
    )
    assert collision.status_code == 409
    assert "never overwritten" in collision.json()["detail"]
    manifest_files = list((writable_workspace.data_root / "manifests").glob("*.parquet"))
    assert [file.name for file in manifest_files] == [entry["manifest_path"].split("/")[-1]]


def test_pin_requires_a_nonempty_name(writable_api: TestClient) -> None:
    assert (
        writable_api.post("/api/v1/curation/pin", json={"sql": OK_CUT_SQL, "name": ""}).status_code
        == 422
    )


def test_pin_names_without_ascii_alphanumerics_fall_back_to_a_slug(
    writable_api: TestClient,
) -> None:
    # Symbols-only and non-Latin-script names are valid: the full Unicode name
    # is stored, and the on-disk filename uses the fallback slug (the
    # timestamp suffix keeps it unique) rather than being refused.
    symbols_only = _pin(writable_api, "!!!")
    assert symbols_only["name"] == "!!!"
    assert symbols_only["manifest_path"].startswith("manifests/manifest-")

    non_latin = _pin(writable_api, "数据集")
    assert non_latin["name"] == "数据集"
    assert non_latin["manifest_path"].startswith("manifests/manifest-")


def test_concurrent_pins_all_land_in_the_registry(
    writable_api: TestClient, writable_workspace: PopulatedWorkspace
) -> None:
    # FastAPI runs these sync endpoints on a threadpool; without a lock around
    # the sidecar read-modify-write, overlapping pins would drop each other's
    # acknowledged registry entry and strand parquet files. Fire several at
    # once and assert every acknowledged pin is registered.
    names = [f"cut-{index}" for index in range(6)]
    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        responses = list(
            pool.map(
                lambda name: writable_api.post(
                    "/api/v1/curation/pin", json={"sql": OK_CUT_SQL, "name": name}
                ),
                names,
            )
        )
    acknowledged_ids = set()
    for response in responses:
        assert response.status_code == 200, response.text
        acknowledged_ids.add(response.json()["id"])
    assert len(acknowledged_ids) == len(names)
    registered_ids = {
        manifest["id"] for manifest in writable_api.get("/api/v1/manifests").json()["manifests"]
    }
    # No acknowledged pin was silently dropped.
    assert acknowledged_ids <= registered_ids
    # And no parquet is stranded (every file on disk has a registry entry).
    manifest_files = list((writable_workspace.data_root / "manifests").glob("*.parquet"))
    assert len(manifest_files) == len(names)


def test_pin_with_bad_sql_is_400_and_registers_nothing(
    writable_api: TestClient, writable_workspace: PopulatedWorkspace
) -> None:
    response = writable_api.post(
        "/api/v1/curation/pin", json={"sql": "SELECT * FROM nope", "name": "broken"}
    )
    assert response.status_code == 400
    manifests_directory = writable_workspace.data_root / "manifests"
    if manifests_directory.exists():
        assert list(manifests_directory.glob("*.parquet")) == []
    assert writable_api.get("/api/v1/manifests").json()["manifests"] == []


def test_pin_is_403_when_read_only(read_only_api: TestClient) -> None:
    response = read_only_api.post(
        "/api/v1/curation/pin", json={"sql": OK_CUT_SQL, "name": "should not land"}
    )
    assert response.status_code == 403
    assert "read-only" in response.json()["detail"]


def test_manifest_download_streams_the_parquet_as_an_attachment(
    writable_api: TestClient, writable_workspace: PopulatedWorkspace
) -> None:
    entry = _pin(writable_api, "download me")
    response = writable_api.get(f"/api/v1/manifests/{entry['id']}/download")
    assert response.status_code == 200
    manifest_file = writable_workspace.data_root / entry["manifest_path"]
    assert response.content == manifest_file.read_bytes()
    content_disposition = response.headers["content-disposition"]
    assert "attachment" in content_disposition
    assert manifest_file.name in content_disposition


def test_manifest_download_of_an_unknown_id_is_404(writable_api: TestClient) -> None:
    assert writable_api.get("/api/v1/manifests/no-such-id/download").status_code == 404


def test_manifest_download_refuses_a_registry_path_outside_the_root(
    writable_api: TestClient, writable_workspace: PopulatedWorkspace
) -> None:
    # A hand-edited registry pointing outside the data root is contained
    # exactly like hostile media URIs -- refused, path never echoed.
    outside_file = writable_workspace.outside_media_file
    state_file = writable_workspace.data_root / "curation" / "state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    escaping_relative_path = f"../{outside_file.parent.name}/{outside_file.name}"
    state_file.write_text(
        json.dumps(
            {
                "state_version": 1,
                "saved_queries": [],
                "manifests": [
                    {
                        "id": "0" * 32,
                        "name": "escape",
                        "description": "",
                        "sql": "SELECT 1",
                        "manifest_path": escaping_relative_path,
                        "row_count": 1,
                        "total_episodes": 1,
                        "coverage": [],
                        "created_at": "2026-08-21T00:00:00+00:00",
                    }
                ],
            }
        )
    )
    response = writable_api.get(f"/api/v1/manifests/{'0' * 32}/download")
    assert response.status_code == 403
    assert str(outside_file) not in response.text


def test_manifest_download_whose_file_vanished_is_404(
    writable_api: TestClient, writable_workspace: PopulatedWorkspace
) -> None:
    entry = _pin(writable_api, "soon gone")
    (writable_workspace.data_root / entry["manifest_path"]).unlink()
    assert writable_api.get(f"/api/v1/manifests/{entry['id']}/download").status_code == 404
