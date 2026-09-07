"""POST /api/v1/curation/preview: wrapped user SQL on a constrained connection."""

import os
import time
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient


def _preview(api: TestClient, **body: object) -> dict:
    response = api.post("/api/v1/curation/preview", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def test_preview_returns_rows_columns_count_and_sql(api: TestClient) -> None:
    payload = _preview(api, sql="SELECT episode_id, task, status FROM episodes")
    assert payload["row_count"] == 4
    assert len(payload["rows"]) == 4
    assert payload["truncated"] is False
    assert payload["column_stats"] is None
    assert [column["name"] for column in payload["columns"]] == ["episode_id", "task", "status"]
    assert all(set(column) == {"name", "type"} for column in payload["columns"])
    assert "SELECT episode_id, task, status FROM episodes" in payload["sql"]
    assert "LIMIT" in payload["sql"]


def test_preview_truncation_flag_reflects_the_full_count(api: TestClient) -> None:
    payload = _preview(api, sql="SELECT episode_id FROM episodes", limit=2)
    assert len(payload["rows"]) == 2
    assert payload["row_count"] == 4
    assert payload["truncated"] is True


def test_preview_timestamps_are_iso_8601_utc_text(api: TestClient) -> None:
    payload = _preview(api, sql="SELECT episode_id, recorded_at FROM episodes")
    for row in payload["rows"]:
        assert isinstance(row["recorded_at"], str)
        assert row["recorded_at"].endswith("+00:00")
        parsed = datetime.fromisoformat(row["recorded_at"])
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == datetime.now(UTC).utcoffset()


def test_preview_null_timestamps_stay_null(api: TestClient) -> None:
    payload = _preview(api, sql="SELECT NULL::TIMESTAMPTZ AS ts")
    assert payload["rows"] == [{"ts": None}]


@pytest.fixture()
def _non_utc_host_timezone() -> object:
    """Run the body with the process pinned to a non-UTC timezone.

    The constrained connection cannot ``SET TimeZone`` (locked at open), so it
    inherits the host's -- this fixture makes that host non-UTC so a
    timezone-leaking render is observable.
    """
    previous_tz = os.environ.get("TZ")
    os.environ["TZ"] = "America/New_York"
    time.tzset()
    try:
        yield
    finally:
        if previous_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous_tz
        time.tzset()


def test_preview_stats_render_timestamps_in_utc_on_a_non_utc_host(
    api: TestClient, _non_utc_host_timezone: object
) -> None:
    payload = _preview(api, sql="SELECT recorded_at FROM episodes", stats=True)
    # Rows are UTC ISO text...
    for row in payload["rows"]:
        assert row["recorded_at"].endswith("+00:00")
    # ...and the column stats agree, so the stats panel never shows a different
    # offset or calendar day than the rows above it.
    stats_by_column = {entry["column_name"]: entry for entry in payload["column_stats"]}
    recorded_at_stats = stats_by_column["recorded_at"]
    row_timestamps = sorted(datetime.fromisoformat(row["recorded_at"]) for row in payload["rows"])
    expected_bounds = {"min": row_timestamps[0], "max": row_timestamps[-1]}
    for bound_key in ("min", "max"):
        bound_value = recorded_at_stats[bound_key]
        assert bound_value.endswith("+00:00"), bound_value
        parsed_bound = datetime.fromisoformat(bound_value)
        assert parsed_bound.utcoffset() == timedelta(0), bound_value
        assert parsed_bound == expected_bounds[bound_key], bound_value
    # The fixture stamps recorded_at as it appends, so a correct render lands
    # near now. Rows and stats pass through the same projection and would shift
    # together, so agreement alone cannot catch a clock moved into the host zone
    # and then stamped +00:00; the host offset here is whole hours away from UTC.
    assert abs(datetime.now(UTC) - expected_bounds["max"]) < timedelta(hours=1)


def test_preview_stats_returns_summarize_rows(api: TestClient) -> None:
    payload = _preview(api, sql="SELECT task, max_velocity FROM episodes", stats=True)
    assert isinstance(payload["column_stats"], list)
    profiled_names = {entry["column_name"] for entry in payload["column_stats"]}
    assert profiled_names == {"task", "max_velocity"}
    for entry in payload["column_stats"]:
        assert "column_type" in entry
        assert "null_percentage" in entry


def test_preview_handles_json_hostile_result_types(api: TestClient) -> None:
    # DECIMAL literals and nested lists come back JSON-legal, never a 500.
    payload = _preview(api, sql="SELECT 1.5 AS a_decimal, [1, 2] AS a_list")
    assert payload["rows"] == [{"a_decimal": 1.5, "a_list": [1, 2]}]


def test_preview_bad_sql_is_400_with_the_duckdb_message(api: TestClient) -> None:
    response = api.post("/api/v1/curation/preview", json={"sql": "SELEC 1"})
    assert response.status_code == 400
    assert "Parser Error" in response.json()["detail"]
    unknown_table = api.post(
        "/api/v1/curation/preview", json={"sql": "SELECT * FROM no_such_table"}
    )
    assert unknown_table.status_code == 400
    assert "no_such_table" in unknown_table.json()["detail"]


def test_preview_refuses_a_paren_closing_smuggle_as_400_not_500(api: TestClient) -> None:
    # This shape closes the subquery wrapper's paren and smuggles a second
    # statement; it used to 500 (an unhandled IndexError), and its CREATE
    # would run. It must be a clean 400 with nothing executed.
    response = api.post(
        "/api/v1/curation/preview",
        json={
            "sql": "SELECT 1 AS a); CREATE TABLE pwned AS SELECT 1; "
            "SELECT count(*) FROM (SELECT 1 AS a"
        },
    )
    assert response.status_code == 400
    # A DROP smuggle in the same shape is likewise refused, not run.
    drop_smuggle = api.post(
        "/api/v1/curation/preview",
        json={"sql": "SELECT 1 AS a); DROP VIEW episodes; SELECT 1 AS a FROM (SELECT 1 AS a"},
    )
    assert drop_smuggle.status_code == 400
    # ...and episodes is still queryable afterward.
    assert (
        api.post("/api/v1/curation/preview", json={"sql": "SELECT * FROM episodes"}).status_code
        == 200
    )


def test_preview_refuses_an_oversized_sql_body(api: TestClient) -> None:
    # The per-field max_length rejects a multi-megabyte SQL body (422) before
    # it can be executed or persisted.
    huge_sql = "SELECT 1 -- " + "A" * 2_000_000
    response = api.post("/api/v1/curation/preview", json={"sql": huge_sql})
    assert response.status_code == 422


def test_preview_cannot_touch_the_filesystem(api: TestClient) -> None:
    # The constrained connection refuses file functions: a 400, never a leak.
    response = api.post(
        "/api/v1/curation/preview", json={"sql": "SELECT * FROM read_csv('/etc/passwd')"}
    )
    assert response.status_code == 400


def test_preview_limit_bounds_are_enforced(api: TestClient) -> None:
    assert (
        api.post("/api/v1/curation/preview", json={"sql": "SELECT 1", "limit": 0}).status_code
        == 422
    )
    assert (
        api.post("/api/v1/curation/preview", json={"sql": "SELECT 1", "limit": 1001}).status_code
        == 422
    )


def test_preview_blank_sql_is_400(api: TestClient) -> None:
    response = api.post("/api/v1/curation/preview", json={"sql": "  ;  "})
    assert response.status_code == 400
    assert "non-empty" in response.json()["detail"]


def test_preview_tolerates_a_trailing_semicolon(api: TestClient) -> None:
    payload = _preview(api, sql="SELECT episode_id FROM episodes;")
    assert payload["row_count"] == 4


def test_preview_still_works_when_read_only(read_only_api: TestClient) -> None:
    response = read_only_api.post(
        "/api/v1/curation/preview", json={"sql": "SELECT count(*) AS n FROM episodes"}
    )
    assert response.status_code == 200
    assert response.json()["rows"] == [{"n": 4}]


def test_preview_without_a_catalog_is_404(empty_workspace_api: TestClient) -> None:
    response = empty_workspace_api.post("/api/v1/curation/preview", json={"sql": "SELECT 1"})
    assert response.status_code == 404
    assert "catalog" in response.json()["detail"]


def test_preview_over_an_empty_catalog_returns_zero_rows(empty_catalog_api: TestClient) -> None:
    payload = _preview(empty_catalog_api, sql="SELECT * FROM episodes")
    assert payload["rows"] == []
    assert payload["row_count"] == 0
    assert payload["truncated"] is False


def test_preview_rejects_pragma_and_describe_consistently(api: TestClient) -> None:
    # DuckDB labels PRAGMA/DESCRIBE/SHOW/SUMMARIZE as SELECT, but the
    # endpoints advertise exactly one read-only SELECT. Preview previously
    # interpolated the SQL as ``SELECT * FROM (<sql>)`` where those forms are
    # a syntax error, leaking the wrapper (``DESCRIBE SELECT * FROM (PRAGMA
    # database_list)``) as the caller's 400. The shared gate now requires
    # SELECT/WITH text, so preview and pin agree (issue #450).
    for sql in (
        "PRAGMA database_list",
        "PRAGMA show_tables",
        "PRAGMA version",
        "DESCRIBE SELECT 1",
        "SHOW TABLES",
        "SUMMARIZE SELECT 1",
    ):
        response = api.post("/api/v1/curation/preview", json={"sql": sql})
        assert response.status_code == 400, response.text
        assert response.json()["detail"] == "sql must be exactly one read-only SELECT statement"
        # No wrapper leak: the detail is the fixed gate message, not DuckDB's
        # parser diagnostic for the rewritten query.
        assert "DESCRIBE SELECT * FROM" not in response.json()["detail"]
        assert "syntax error at or near" not in response.json()["detail"]


def test_preview_accepts_select_with_leading_comments_and_cte(api: TestClient) -> None:
    for sql in (
        "-- comment\nSELECT episode_id FROM episodes",
        "/* block */ SELECT episode_id FROM episodes",
        "WITH c AS (SELECT 1 AS one) SELECT * FROM c",
    ):
        assert api.post("/api/v1/curation/preview", json={"sql": sql}).status_code == 200


def test_preview_accepts_from_first_and_parenthesized_selects(api: TestClient) -> None:
    # Review of #453: FROM-first queries, parenthesized SELECTs, and VALUES
    # clauses are legal read-only DuckDB statements (StatementType.SELECT)
    # that the preview endpoint must run end-to-end, not refuse at the gate.
    for sql in (
        "FROM episodes WHERE status = 'ok'",
        "(SELECT episode_id FROM episodes)",
        "VALUES (1, 'a'), (2, 'b')",
    ):
        response = api.post("/api/v1/curation/preview", json={"sql": sql})
        assert response.status_code == 200, response.text
