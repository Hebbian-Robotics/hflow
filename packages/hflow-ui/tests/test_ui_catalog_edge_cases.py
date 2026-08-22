"""Catalog edge cases over purpose-built workspaces: measurement-key columns
whose names contain '?' (sortable header the endpoint advertises) and
numeric columns whose value span overflows to infinity."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from hflow_ui import UiSettings, create_app
from ui_test_fixtures import STAMPS

import hflow
from hflow.catalog import Catalog, CheckRunRow
from hflow.steps import MeasurementValue


def _client_over(data_root: Path) -> TestClient:
    return TestClient(create_app(UiSettings(data_root=str(data_root))))


def _append_with_measurements(
    catalog: Catalog,
    episodes_dir: Path,
    stem: str,
    task: str,
    measurements: dict[str, MeasurementValue],
) -> str:
    canonical = episodes_dir / f"{stem}.canonical.mcap"
    canonical.write_bytes(b"canonical " + stem.encode())
    result = catalog.append_episode(
        canonical_path=canonical,
        stamps=STAMPS,
        episode_metadata={"task": task, "operator": "alice", "embodiment": "arm-1"},
        check_rows=[
            CheckRunRow(
                check_name="metrics_check",
                check_version="v1",
                critical=False,
                status=hflow.CheckStatus.MEASURED,
                duration_s=0.01,
                measurements=measurements,
            )
        ],
    )
    return result.episode_id


@pytest.fixture()
def question_mark_column_api(tmp_path: Path) -> TestClient:
    """A workspace whose measurement key contains '?', so it pivots into an
    episodes column named e.g. ``gripper_ok?``."""
    data_root = tmp_path / "data"
    episodes_dir = data_root / "episodes"
    episodes_dir.mkdir(parents=True)
    catalog = Catalog(data_root / "catalog")
    _append_with_measurements(catalog, episodes_dir, "a", "fold", {"gripper_ok?": 1.0})
    _append_with_measurements(catalog, episodes_dir, "b", "pour", {"gripper_ok?": 0.0})
    return _client_over(data_root)


def test_order_by_a_column_whose_name_contains_a_question_mark(
    question_mark_column_api: TestClient,
) -> None:
    listing = question_mark_column_api.get("/api/v1/episodes")
    assert listing.status_code == 200
    # The endpoint advertises the '?'-named column as sortable.
    assert "gripper_ok?" in {column["name"] for column in listing.json()["columns"]}
    # Sorting by it must not 500 on the display-SQL renderer.
    ordered = question_mark_column_api.get(
        "/api/v1/episodes", params={"order_by": "gripper_ok?", "order": "asc"}
    )
    assert ordered.status_code == 200, ordered.text
    values = [row["gripper_ok?"] for row in ordered.json()["rows"]]
    assert values == sorted(values)
    # The rendered display SQL still round-trips (its '?' is inside a quoted
    # identifier, never miscounted as a bind placeholder).
    assert 'ORDER BY "gripper_ok?"' in ordered.json()["sql"]


def test_order_by_a_question_mark_column_with_a_filter(
    question_mark_column_api: TestClient,
) -> None:
    # A filter adds real bind placeholders; the '?' in the order_by identifier
    # must still not be counted among them.
    filtered = question_mark_column_api.get(
        "/api/v1/episodes", params={"order_by": "gripper_ok?", "task": "fold"}
    )
    assert filtered.status_code == 200, filtered.text
    assert filtered.json()["total"] == 1


@pytest.fixture()
def infinite_span_stats_api(tmp_path: Path) -> TestClient:
    """A workspace with a numeric column whose max-min overflows to inf."""
    data_root = tmp_path / "data"
    episodes_dir = data_root / "episodes"
    episodes_dir.mkdir(parents=True)
    catalog = Catalog(data_root / "catalog")
    _append_with_measurements(catalog, episodes_dir, "lo", "fold", {"huge": -1.7e308, "ok": 1.0})
    _append_with_measurements(catalog, episodes_dir, "hi", "fold", {"huge": 1.7e308, "ok": 2.0})
    _append_with_measurements(catalog, episodes_dir, "mid", "fold", {"huge": 0.0, "ok": 3.0})
    return _client_over(data_root)


def test_stats_do_not_500_when_a_columns_span_overflows_to_infinity(
    infinite_span_stats_api: TestClient,
) -> None:
    response = infinite_span_stats_api.get("/api/v1/episodes/stats")
    assert response.status_code == 200, response.text
    profiled = {column["name"] for column in response.json()["columns"]}
    # The overflowing column is skipped as degenerate; the well-behaved
    # numeric column beside it still earns a histogram.
    assert "huge" not in profiled
    assert "ok" in profiled
