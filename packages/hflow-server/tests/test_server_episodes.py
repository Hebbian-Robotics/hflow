"""GET /api/v1/episodes: filters, ordering, pagination, facets, SQL safety."""

from datetime import datetime

from fastapi.testclient import TestClient
from ui_test_fixtures import PIPELINE_VERSION, PopulatedWorkspace

import hflow


def _episode_rows(api: TestClient, **query_params: str | int | list[str]) -> dict:
    response = api.get("/api/v1/episodes", params=query_params)
    assert response.status_code == 200, response.text
    return response.json()


def test_default_listing_returns_every_episode(api: TestClient) -> None:
    payload = _episode_rows(api)
    assert payload["total"] == 4
    assert len(payload["rows"]) == 4
    column_names = {column["name"] for column in payload["columns"]}
    assert {"episode_id", "task", "status", "recorded_at"} <= column_names
    assert all(set(column) == {"name", "type"} for column in payload["columns"])


def test_measurement_keys_become_columns(api: TestClient) -> None:
    payload = _episode_rows(api)
    column_names = {column["name"] for column in payload["columns"]}
    assert "max_velocity" in column_names  # pivoted from the measurements table


def test_timestamps_are_iso_8601_strings(api: TestClient) -> None:
    for row in _episode_rows(api)["rows"]:
        assert isinstance(row["recorded_at"], str)
        assert datetime.fromisoformat(row["recorded_at"]).tzinfo is not None


def test_timestamps_use_a_full_colon_utc_offset(api: TestClient) -> None:
    # DuckDB's strftime %z renders the offset as "+00" (which JS Date.parse
    # rejects and the frontend's offset regex misses); every timestamp must
    # carry the full "+00:00" the rest of the stack assumes.
    for row in _episode_rows(api)["rows"]:
        assert row["recorded_at"].endswith("+00:00"), row["recorded_at"]
        assert not row["recorded_at"].endswith("+00")  # sanity: not the bare form


def test_pagination_over_a_tied_sort_key_never_overlaps_or_drops(api: TestClient) -> None:
    # pipeline_version is identical across all four episodes, so without a
    # deterministic tiebreaker successive pages could overlap or drop rows.
    total = _episode_rows(api, order_by="pipeline_version")["total"]
    assert total == 4
    seen_ids: list[str] = []
    for offset in range(0, total, 2):
        page = _episode_rows(api, order_by="pipeline_version", order="asc", limit=2, offset=offset)
        seen_ids.extend(row["episode_id"] for row in page["rows"])
    # Every episode appears exactly once across the walked pages.
    assert len(seen_ids) == total
    assert len(set(seen_ids)) == total
    # And the walk is repeatable: the same offsets return the same rows.
    repeat_first_page = _episode_rows(
        api, order_by="pipeline_version", order="asc", limit=2, offset=0
    )
    assert [row["episode_id"] for row in repeat_first_page["rows"]] == seen_ids[:2]


def test_non_finite_doubles_become_null(
    api: TestClient, populated_workspace: PopulatedWorkspace
) -> None:
    rows_by_id = {row["episode_id"]: row for row in _episode_rows(api)["rows"]}
    ok_row = rows_by_id[populated_workspace.ok_episode_id]
    assert ok_row["nan_metric"] is None
    assert ok_row["inf_metric"] is None
    assert ok_row["max_velocity"] == 2.0  # the latest run's value


def test_task_filter_matches_exactly(api: TestClient) -> None:
    payload = _episode_rows(api, task="fold_napkin")
    assert payload["total"] == 2
    assert all(row["task"] == "fold_napkin" for row in payload["rows"])


def test_repeated_filter_values_are_or_combined(api: TestClient) -> None:
    payload = _episode_rows(api, task=["fold_napkin", "pour_water"])
    assert payload["total"] == 3


def test_different_filters_are_and_combined(api: TestClient) -> None:
    payload = _episode_rows(api, task="fold_napkin", operator="alice")
    assert payload["total"] == 1
    assert payload["rows"][0]["operator"] == "alice"


def test_status_filter(api: TestClient, populated_workspace: PopulatedWorkspace) -> None:
    payload = _episode_rows(api, status="quarantined")
    assert payload["total"] == 1
    assert payload["rows"][0]["episode_id"] == populated_workspace.quarantined_episode_id
    assert _episode_rows(api, status="ok")["total"] == 3


def test_invalid_status_value_is_a_422(api: TestClient) -> None:
    assert api.get("/api/v1/episodes", params={"status": "banana"}).status_code == 422


def test_success_filter(api: TestClient) -> None:
    assert _episode_rows(api, success="true")["total"] == 1
    assert _episode_rows(api, success="false")["total"] == 1


def test_search_is_case_insensitive_substring(
    api: TestClient, populated_workspace: PopulatedWorkspace
) -> None:
    assert _episode_rows(api, search="FOLD")["total"] == 2  # task match
    assert _episode_rows(api, search="ALICE")["total"] == 1  # operator match
    by_id = _episode_rows(api, search=populated_workspace.ok_episode_id)
    assert by_id["total"] == 1  # episode_id match
    assert by_id["rows"][0]["episode_id"] == populated_workspace.ok_episode_id


def test_search_wildcards_are_literal(api: TestClient) -> None:
    # Un-escaped, "%" would match every row and "_" any character.
    assert _episode_rows(api, search="%")["total"] == 0
    assert _episode_rows(api, search="fold_napkin")["total"] == 2
    assert _episode_rows(api, search="foldXnapkin")["total"] == 0


def test_filter_values_cannot_inject_sql(api: TestClient) -> None:
    hostile_value = "x' OR '1'='1"
    assert _episode_rows(api, task=hostile_value)["total"] == 0
    assert _episode_rows(api, search=hostile_value)["total"] == 0
    assert _episode_rows(api, search="'; DROP TABLE episodes; --")["total"] == 0


def test_order_by_direction_flips_the_listing(api: TestClient) -> None:
    ascending_ids = [
        row["episode_id"] for row in _episode_rows(api, order_by="recorded_at", order="asc")["rows"]
    ]
    descending_ids = [
        row["episode_id"]
        for row in _episode_rows(api, order_by="recorded_at", order="desc")["rows"]
    ]
    assert ascending_ids == list(reversed(descending_ids))


def test_order_by_any_view_column_including_measurements(api: TestClient) -> None:
    payload = _episode_rows(api, order_by="task", order="asc")
    listed_tasks = [row["task"] for row in payload["rows"]]
    assert listed_tasks == sorted(listed_tasks)
    assert _episode_rows(api, order_by="max_velocity")["total"] == 4


def test_unknown_order_by_column_is_a_400_not_sql(api: TestClient) -> None:
    response = api.get("/api/v1/episodes", params={"order_by": "no_such_column"})
    assert response.status_code == 400
    assert "no_such_column" in response.json()["detail"]
    hostile = api.get("/api/v1/episodes", params={"order_by": "task; DROP TABLE episodes"})
    assert hostile.status_code == 400


def test_pagination_pages_share_one_total(api: TestClient) -> None:
    first_page = _episode_rows(api, limit=2, offset=0, order_by="episode_id", order="asc")
    second_page = _episode_rows(api, limit=2, offset=2, order_by="episode_id", order="asc")
    beyond_page = _episode_rows(api, limit=2, offset=4, order_by="episode_id", order="asc")
    assert first_page["total"] == second_page["total"] == beyond_page["total"] == 4
    assert len(first_page["rows"]) == 2
    assert len(second_page["rows"]) == 2
    assert beyond_page["rows"] == []
    first_ids = {row["episode_id"] for row in first_page["rows"]}
    second_ids = {row["episode_id"] for row in second_page["rows"]}
    assert first_ids.isdisjoint(second_ids)


def test_pagination_bounds_are_enforced(api: TestClient) -> None:
    assert api.get("/api/v1/episodes", params={"limit": 501}).status_code == 422
    assert api.get("/api/v1/episodes", params={"limit": 0}).status_code == 422
    assert api.get("/api/v1/episodes", params={"offset": -1}).status_code == 422


def test_compiled_sql_is_returned_and_runnable(
    api: TestClient, populated_workspace: PopulatedWorkspace
) -> None:
    payload = _episode_rows(api, task="fold_napkin", limit=2)
    compiled_sql = payload["sql"]
    assert "\"task\" IN ('fold_napkin')" in compiled_sql
    assert "LIMIT 2" in compiled_sql
    assert "?" not in compiled_sql  # fully rendered, copy-pastable
    # The affordance is real: the displayed SQL runs against the same catalog.
    # (Counted in SQL: materializing a TIMESTAMPTZ into Python needs pytz.)
    connection = hflow.open_catalog_connection(populated_workspace.data_root / "catalog")
    try:
        replayed_count_row = connection.execute(f"SELECT count(*) FROM ({compiled_sql})").fetchone()
    finally:
        connection.close()
    assert replayed_count_row is not None
    assert int(replayed_count_row[0]) == len(payload["rows"])


def test_listing_without_a_catalog_is_a_404(empty_workspace_api: TestClient) -> None:
    response = empty_workspace_api.get("/api/v1/episodes")
    assert response.status_code == 404
    assert "catalog" in response.json()["detail"]


def test_listing_an_empty_catalog_returns_zero_rows(empty_catalog_api: TestClient) -> None:
    payload = empty_catalog_api.get("/api/v1/episodes").json()
    assert payload["rows"] == []
    assert payload["total"] == 0
    assert {column["name"] for column in payload["columns"]} >= {"episode_id", "status"}


def test_facets_counts_skip_null_buckets(api: TestClient) -> None:
    response = api.get("/api/v1/episodes/facets")
    assert response.status_code == 200
    facets = response.json()
    assert set(facets) == {"task", "operator", "embodiment", "status", "pipeline_version"}
    task_counts = {entry["value"]: entry["count"] for entry in facets["task"]}
    assert task_counts == {"fold_napkin": 2, "pour_water": 1, "stack_blocks": 1}
    # The no-operator episode contributes no null bucket.
    assert {entry["value"] for entry in facets["operator"]} == {"alice", "bob", "carol"}
    status_counts = {entry["value"]: entry["count"] for entry in facets["status"]}
    assert status_counts == {"ok": 3, "quarantined": 1}
    pipeline_counts = {entry["value"]: entry["count"] for entry in facets["pipeline_version"]}
    assert pipeline_counts == {PIPELINE_VERSION: 4}


def test_facets_of_an_empty_catalog_are_empty_lists(empty_catalog_api: TestClient) -> None:
    facets = empty_catalog_api.get("/api/v1/episodes/facets").json()
    assert all(entries == [] for entries in facets.values())
