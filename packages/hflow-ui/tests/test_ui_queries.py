"""/api/v1/queries: saved-query CRUD over the sidecar state."""

from fastapi.testclient import TestClient
from hflow_ui import UiSettings, create_app
from ui_test_fixtures import PopulatedWorkspace


def _created_query(api: TestClient, name: str, sql: str) -> dict:
    response = api.post("/api/v1/queries", json={"name": name, "sql": sql})
    assert response.status_code == 200, response.text
    return response.json()


def test_queries_start_empty(writable_api: TestClient) -> None:
    assert writable_api.get("/api/v1/queries").json() == {"queries": []}


def test_create_list_update_delete_roundtrip(writable_api: TestClient) -> None:
    created = _created_query(writable_api, "ok cut", "SELECT * FROM episodes")
    assert set(created) == {"id", "name", "sql", "updated_at"}
    assert created["name"] == "ok cut"

    listed = writable_api.get("/api/v1/queries").json()["queries"]
    assert listed == [created]

    renamed = writable_api.put(f"/api/v1/queries/{created['id']}", json={"name": "great cut"})
    assert renamed.status_code == 200
    assert renamed.json()["name"] == "great cut"
    assert renamed.json()["sql"] == created["sql"]  # untouched fields survive

    new_sql = "SELECT episode_id FROM episodes WHERE status = 'ok'"
    edited = writable_api.put(f"/api/v1/queries/{created['id']}", json={"sql": new_sql})
    assert edited.status_code == 200
    assert edited.json()["sql"] == new_sql
    assert edited.json()["name"] == "great cut"
    assert edited.json()["updated_at"] >= created["updated_at"]

    deleted = writable_api.delete(f"/api/v1/queries/{created['id']}")
    assert deleted.status_code == 204
    assert writable_api.get("/api/v1/queries").json() == {"queries": []}


def test_unknown_query_ids_are_404(writable_api: TestClient) -> None:
    assert writable_api.put("/api/v1/queries/missing", json={"name": "x"}).status_code == 404
    assert writable_api.delete("/api/v1/queries/missing").status_code == 404


def test_create_requires_nonempty_name_and_sql(writable_api: TestClient) -> None:
    assert (
        writable_api.post("/api/v1/queries", json={"name": "", "sql": "SELECT 1"}).status_code
        == 422
    )
    assert (
        writable_api.post("/api/v1/queries", json={"name": "   ", "sql": "SELECT 1"}).status_code
        == 400
    )
    assert writable_api.post("/api/v1/queries", json={"name": "q", "sql": " ; "}).status_code == 400


def test_update_to_a_blank_name_or_sql_is_refused(writable_api: TestClient) -> None:
    created = _created_query(writable_api, "keep me", "SELECT 1")
    assert (
        writable_api.put(f"/api/v1/queries/{created['id']}", json={"name": " "}).status_code == 400
    )
    assert (
        writable_api.put(f"/api/v1/queries/{created['id']}", json={"sql": ";"}).status_code == 400
    )


def test_create_refuses_oversized_name_and_sql(writable_api: TestClient) -> None:
    over_long_name = writable_api.post(
        "/api/v1/queries", json={"name": "n" * 201, "sql": "SELECT 1"}
    )
    assert over_long_name.status_code == 422
    over_long_sql = writable_api.post(
        "/api/v1/queries", json={"name": "ok", "sql": "SELECT 1 -- " + "A" * 100_001}
    )
    assert over_long_sql.status_code == 422


def test_oversized_request_body_is_refused_at_the_boundary(writable_api: TestClient) -> None:
    # A body far larger than any per-field cap is refused (413) before it is
    # parsed or persisted.
    huge_body = {"name": "ok", "sql": "SELECT 1 -- " + "A" * (5 * 1024 * 1024)}
    response = writable_api.post("/api/v1/queries", json=huge_body)
    assert response.status_code == 413


def test_query_writes_are_403_when_read_only(read_only_api: TestClient) -> None:
    assert read_only_api.get("/api/v1/queries").status_code == 200  # reading stays open
    refusals = [
        read_only_api.post("/api/v1/queries", json={"name": "x", "sql": "SELECT 1"}),
        read_only_api.put("/api/v1/queries/some-id", json={"name": "x"}),
        read_only_api.delete("/api/v1/queries/some-id"),
    ]
    for refusal in refusals:
        assert refusal.status_code == 403
        assert "read-only" in refusal.json()["detail"]


def test_saved_queries_persist_across_server_restarts(
    writable_api: TestClient, writable_workspace: PopulatedWorkspace
) -> None:
    created = _created_query(writable_api, "durable", "SELECT 1")
    restarted_api = TestClient(
        create_app(UiSettings(data_root=str(writable_workspace.data_root), token=None))
    )
    assert restarted_api.get("/api/v1/queries").json()["queries"] == [created]
