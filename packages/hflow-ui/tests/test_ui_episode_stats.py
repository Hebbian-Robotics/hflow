"""GET /api/v1/episodes/stats: per-column mini-distributions under filters."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from hflow_ui import UiSettings, create_app
from ui_test_fixtures import STAMPS, PopulatedWorkspace

import hflow
from hflow.catalog import Catalog, CheckRunRow

# (task, operator, max_velocity) for six episodes: two tasks, three
# operators, six distinct velocities -- plus constant columns (success,
# pipeline_version, ...) and all-unique ones (episode_id, uri) that the
# degenerate-skip rules must drop.
EPISODE_SPECIFICATIONS = (
    ("fold", "alice", 1.0),
    ("fold", "alice", 2.0),
    ("fold", "bob", 3.0),
    ("fold", "bob", 4.0),
    ("pour", "carol", 10.0),
    ("pour", "carol", 20.0),
)


def _build_stats_workspace(tmp_path: Path) -> Path:
    data_root = tmp_path / "stats-root"
    episodes_directory = data_root / "episodes"
    episodes_directory.mkdir(parents=True)
    catalog = Catalog(data_root / "catalog")
    for index, (task, operator, velocity) in enumerate(EPISODE_SPECIFICATIONS):
        canonical_file = episodes_directory / f"episode_{index}.canonical.mcap"
        canonical_file.write_bytes(f"canonical body {index}".encode())
        appended = catalog.append_episode(
            canonical_path=canonical_file,
            stamps=STAMPS,
            episode_metadata={"task": task, "operator": operator, "success": "true"},
            check_rows=[
                CheckRunRow(
                    check_name="velocity_check",
                    check_version="v1",
                    critical=False,
                    status=hflow.CheckStatus.MEASURED,
                    duration_s=0.01,
                    measurements={"max_velocity": velocity},
                )
            ],
        )
        assert appended.written
    return data_root


@pytest.fixture()
def stats_api(tmp_path: Path, unbuilt_assets_dir: Path) -> TestClient:
    data_root = _build_stats_workspace(tmp_path)
    settings = UiSettings(data_root=str(data_root), assets_dir=unbuilt_assets_dir)
    return TestClient(create_app(settings))


def _columns_by_name(payload: dict) -> dict[str, dict]:
    return {column["name"]: column for column in payload["columns"]}


def test_stats_shapes_numeric_histograms_and_categorical_top_values(
    stats_api: TestClient,
) -> None:
    response = stats_api.get("/api/v1/episodes/stats")
    assert response.status_code == 200
    columns = _columns_by_name(response.json())

    velocity = columns["max_velocity"]
    assert velocity["kind"] == "numeric"
    assert len(velocity["buckets"]) == 12
    assert sum(bucket["count"] for bucket in velocity["buckets"]) == 6
    assert velocity["buckets"][0]["lo"] == 1.0
    assert velocity["buckets"][-1]["hi"] == 20.0
    # The maximum value lands in the LAST bucket, never off the end.
    assert velocity["buckets"][-1]["count"] >= 1

    task = columns["task"]
    assert task["kind"] == "categorical"
    assert task["values"] == [
        {"value": "fold", "count": 4},
        {"value": "pour", "count": 2},
    ]
    assert task["other_count"] == 0

    operator = columns["operator"]
    assert {entry["value"]: entry["count"] for entry in operator["values"]} == {
        "alice": 2,
        "bob": 2,
        "carol": 2,
    }


def test_stats_skips_degenerate_columns(stats_api: TestClient) -> None:
    column_names = set(_columns_by_name(stats_api.get("/api/v1/episodes/stats").json()))
    # All-unique (id-like) columns are not distributions.
    assert "episode_id" not in column_names
    assert "uri" not in column_names
    # Single-valued columns carry no information under these episodes.
    assert "success" not in column_names
    assert "pipeline_version" not in column_names
    assert "schema_version" not in column_names
    assert "status" not in column_names


def test_stats_respects_the_active_filters(stats_api: TestClient) -> None:
    columns = _columns_by_name(
        stats_api.get("/api/v1/episodes/stats", params={"task": "fold"}).json()
    )
    velocity = columns["max_velocity"]
    assert sum(bucket["count"] for bucket in velocity["buckets"]) == 4
    assert velocity["buckets"][-1]["hi"] == 4.0  # pour's 10.0/20.0 filtered away
    assert {entry["value"] for entry in columns["operator"]["values"]} == {"alice", "bob"}
    # Under the filter every row is task=fold: now degenerate, so dropped.
    assert "task" not in columns


def test_stats_filter_values_are_bound_not_interpolated(stats_api: TestClient) -> None:
    response = stats_api.get("/api/v1/episodes/stats", params={"task": "x' OR '1'='1"})
    assert response.status_code == 200
    assert response.json() == {"columns": []}  # matches nothing, injects nothing


def test_stats_over_the_populated_workspace_keeps_the_status_split(
    api: TestClient, populated_workspace: PopulatedWorkspace
) -> None:
    columns = _columns_by_name(api.get("/api/v1/episodes/stats").json())
    status = columns["status"]
    assert status["kind"] == "categorical"
    assert {entry["value"]: entry["count"] for entry in status["values"]} == {
        "ok": 3,
        "quarantined": 1,
    }
    # NaN/inf-poisoned measurement columns are degenerate, never a crash.
    assert "nan_metric" not in columns
    assert "inf_metric" not in columns


def test_stats_without_a_catalog_is_a_404(empty_workspace_api: TestClient) -> None:
    assert empty_workspace_api.get("/api/v1/episodes/stats").status_code == 404
