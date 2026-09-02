"""POST /api/v1/curation/report: row count + coverage denominators, no writes."""

from fastapi.testclient import TestClient


def test_report_counts_rows_and_coverage_denominators(api: TestClient) -> None:
    response = api.post(
        "/api/v1/curation/report",
        json={"sql": "SELECT episode_id FROM episodes WHERE status = 'ok'"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["row_count"] == 3
    assert payload["total_episodes"] == 4
    coverage_by_check = {entry["check_name"]: entry for entry in payload["coverage"]}
    # Coverage is over the WHOLE catalog (denominators), not the cut.
    assert set(coverage_by_check) == {"joint_check", "media/contact_sheet", "camera_blackout"}
    contact_sheet = coverage_by_check["media/contact_sheet"]
    assert contact_sheet == {
        "check_name": "media/contact_sheet",
        "episodes_ran": 2,
        "total_episodes": 4,
        "fraction": 0.5,
    }
    assert coverage_by_check["joint_check"]["episodes_ran"] == 1
    assert coverage_by_check["camera_blackout"]["episodes_ran"] == 1


def test_report_bad_sql_is_400_with_the_duckdb_message(api: TestClient) -> None:
    response = api.post("/api/v1/curation/report", json={"sql": "SELECT * FROM nope"})
    assert response.status_code == 400
    assert "nope" in response.json()["detail"]


def test_report_refuses_smuggled_second_statement_and_reports_honestly(api: TestClient) -> None:
    # A paren-closing smuggle used to slip past the subquery wrapper: the
    # extra CREATE ran and the reported row_count came from a trailing SELECT.
    smuggle = api.post(
        "/api/v1/curation/report",
        json={
            "sql": "SELECT episode_id FROM episodes); CREATE TABLE pwn AS SELECT 1; SELECT 999 --"
        },
    )
    assert smuggle.status_code == 400
    # The honest single-statement query still reports the real row count.
    honest = api.post("/api/v1/curation/report", json={"sql": "SELECT episode_id FROM episodes"})
    assert honest.status_code == 200
    assert honest.json()["row_count"] == 4


def test_report_over_an_empty_catalog(empty_catalog_api: TestClient) -> None:
    response = empty_catalog_api.post(
        "/api/v1/curation/report", json={"sql": "SELECT * FROM episodes"}
    )
    assert response.status_code == 200
    assert response.json() == {"row_count": 0, "total_episodes": 0, "coverage": []}


def test_report_still_works_when_read_only(read_only_api: TestClient) -> None:
    response = read_only_api.post("/api/v1/curation/report", json={"sql": "SELECT * FROM episodes"})
    assert response.status_code == 200
    assert response.json()["row_count"] == 4


def test_report_without_a_catalog_is_404(empty_workspace_api: TestClient) -> None:
    response = empty_workspace_api.post("/api/v1/curation/report", json={"sql": "SELECT 1"})
    assert response.status_code == 404
