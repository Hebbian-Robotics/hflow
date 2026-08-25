"""/api/v1/catalog/tables: the browsable schema tree and per-table summaries."""

from fastapi.testclient import TestClient

EXPECTED_TABLE_NAMES = [
    "episodes",
    "episodes_latest",
    "episodes_raw",
    "check_runs",
    "measurements",
    "measurements_latest",
    "tags",
    "intervals",
    # The complement of `episodes`: sources that produced no row there.
    "ingest_failures",
]


def test_tables_lists_every_registered_view_with_columns(api: TestClient) -> None:
    response = api.get("/api/v1/catalog/tables")
    assert response.status_code == 200
    tables = response.json()["tables"]
    assert [table["name"] for table in tables] == EXPECTED_TABLE_NAMES
    assert all(table["kind"] in ("view", "table") for table in tables)
    columns_by_table = {table["name"]: table["columns"] for table in tables}
    episode_column_names = {column["name"] for column in columns_by_table["episodes"]}
    # The wide view: episode columns, the status column, pivoted measurements.
    assert {"episode_id", "task", "status", "max_velocity"} <= episode_column_names
    check_run_column_names = {column["name"] for column in columns_by_table["check_runs"]}
    assert {"check_name", "status", "duration_s"} <= check_run_column_names
    for table in tables:
        assert all(set(column) == {"name", "type"} for column in table["columns"])


def test_table_summary_profiles_row_count_and_columns(api: TestClient) -> None:
    response = api.get("/api/v1/catalog/tables/episodes/summary")
    assert response.status_code == 200
    payload = response.json()
    assert payload["row_count"] == 4
    profiled_names = {entry["column_name"] for entry in payload["columns"]}
    assert {"episode_id", "task", "status", "max_velocity"} <= profiled_names
    velocity_profile = next(
        entry for entry in payload["columns"] if entry["column_name"] == "max_velocity"
    )
    assert velocity_profile["max"] == "2.0"  # SUMMARIZE renders extremes as text


def test_unknown_or_hostile_table_names_are_404(api: TestClient) -> None:
    assert api.get("/api/v1/catalog/tables/no_such_table/summary").status_code == 404
    hostile = api.get("/api/v1/catalog/tables/episodes; DROP TABLE episodes_raw/summary")
    assert hostile.status_code == 404


def test_summary_of_an_empty_catalog_reports_zero_rows(empty_catalog_api: TestClient) -> None:
    response = empty_catalog_api.get("/api/v1/catalog/tables/episodes/summary")
    assert response.status_code == 200
    assert response.json()["row_count"] == 0


def test_tables_without_a_catalog_is_404(empty_workspace_api: TestClient) -> None:
    assert empty_workspace_api.get("/api/v1/catalog/tables").status_code == 404
    assert empty_workspace_api.get("/api/v1/catalog/tables/episodes/summary").status_code == 404
