"""GET /api/v1/episodes/{id}/timeline: the episode's time axis, server-side.

The span derivation is the point: intervals give the axis when they exist, a
duration-naming measurement gives (or extends) it otherwise, and an episode
that offers neither returns nulls so the UI can say "unknown" instead of
drawing a fabricated axis.
"""

import pytest
from fastapi.testclient import TestClient
from hflow_server import ServerSettings, create_app
from ui_test_fixtures import STAMPS, PopulatedWorkspace

import hflow
from hflow.catalog import Catalog, CheckRunRow

NANOSECONDS_PER_SECOND = 1_000_000_000


@pytest.fixture(scope="module")
def timeline_workspace(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """A catalog whose episodes exercise every branch of the span derivation."""
    data_root = tmp_path_factory.mktemp("ui-timeline-root")
    episodes_directory = data_root / "episodes"
    episodes_directory.mkdir()
    catalog = Catalog(data_root / "catalog")

    def appended(name: str, check_rows: list[CheckRunRow]) -> str:
        canonical_file = episodes_directory / f"{name}.canonical.mcap"
        canonical_file.write_bytes(f"canonical {name}".encode())
        return catalog.append_episode(
            canonical_path=canonical_file,
            stamps=STAMPS,
            episode_metadata={"task": name},
            check_rows=check_rows,
        ).episode_id

    intervals_and_duration = appended(
        "intervals_and_duration",
        [
            CheckRunRow(
                check_name="gap_check",
                check_version="v1",
                critical=False,
                status=hflow.CheckStatus.MEASURED,
                duration_s=0.01,
                measurements={
                    "episode_duration_s": 12.5,
                    "max_gap_ms": 220.0,
                    "black_pct": 3.5,
                    "max_velocity": 1.25,
                },
                intervals=[
                    hflow.Interval(
                        start_ns=1 * NANOSECONDS_PER_SECOND,
                        end_ns=2 * NANOSECONDS_PER_SECOND,
                        label="gap:/imu",
                    ),
                    hflow.Interval(
                        start_ns=3 * NANOSECONDS_PER_SECOND,
                        end_ns=4 * NANOSECONDS_PER_SECOND,
                        label="joint_discontinuity:/joint_states",
                    ),
                    hflow.Interval(
                        start_ns=5 * NANOSECONDS_PER_SECOND,
                        end_ns=6 * NANOSECONDS_PER_SECOND,
                        label="",
                    ),
                ],
            )
        ],
    )
    duration_only = appended(
        "duration_only",
        [
            CheckRunRow(
                check_name="length_check",
                check_version="v1",
                critical=False,
                status=hflow.CheckStatus.MEASURED,
                duration_s=0.01,
                measurements={"episode_duration": 30.0},
            )
        ],
    )
    duration_named_percentage = appended(
        "duration_named_percentage",
        [
            CheckRunRow(
                check_name="duty_cycle_check",
                check_version="v1",
                critical=False,
                status=hflow.CheckStatus.MEASURED,
                duration_s=0.01,
                # Says "duration", measures a percentage: the suffix names a
                # dimension that is not a time.
                measurements={"duty_cycle_duration_pct": 45.0},
            )
        ],
    )
    no_span = appended(
        "no_span",
        [
            CheckRunRow(
                check_name="counting_check",
                check_version="v1",
                critical=False,
                status=hflow.CheckStatus.MEASURED,
                duration_s=0.01,
                measurements={"frame_count": 900.0, "note": "text is not a bar"},
            )
        ],
    )
    return {
        "data_root": str(data_root),
        "intervals_and_duration": intervals_and_duration,
        "duration_only": duration_only,
        "duration_named_percentage": duration_named_percentage,
        "no_span": no_span,
    }


@pytest.fixture(scope="module")
def timeline_api(
    timeline_workspace: dict[str, str], tmp_path_factory: pytest.TempPathFactory
) -> TestClient:
    assets_directory = tmp_path_factory.mktemp("ui-timeline-assets")
    settings = ServerSettings(
        data_root=timeline_workspace["data_root"], assets_dir=assets_directory
    )
    return TestClient(create_app(settings))


def test_timeline_spans_the_intervals_and_the_duration_measurement(
    timeline_api: TestClient, timeline_workspace: dict[str, str]
) -> None:
    payload = timeline_api.get(
        f"/api/v1/episodes/{timeline_workspace['intervals_and_duration']}/timeline"
    ).json()
    # The axis starts at the first interval; the 12.5s duration measurement
    # claims a longer episode than the intervals do, so the span extends.
    assert payload["start_ns"] == 1 * NANOSECONDS_PER_SECOND
    assert payload["end_ns"] == 1 * NANOSECONDS_PER_SECOND + int(12.5 * NANOSECONDS_PER_SECOND)
    assert payload["duration_s"] == pytest.approx(12.5)

    intervals = payload["intervals"]
    assert [interval["kind"] for interval in intervals] == [
        "gap",
        "joint_discontinuity",
        # An empty label groups under the check that produced it.
        "gap_check",
    ]
    assert intervals[0]["label"] == "gap:/imu"
    assert intervals[0]["check_name"] == "gap_check"
    # Seconds are RELATIVE to the span start, computed server-side.
    assert intervals[0]["start_s"] == pytest.approx(0.0)
    assert intervals[0]["end_s"] == pytest.approx(1.0)
    assert intervals[1]["start_s"] == pytest.approx(2.0)
    assert intervals[1]["end_s"] == pytest.approx(3.0)


def test_timeline_measurements_are_numeric_bars_with_inferred_units(
    timeline_api: TestClient, timeline_workspace: dict[str, str]
) -> None:
    payload = timeline_api.get(
        f"/api/v1/episodes/{timeline_workspace['intervals_and_duration']}/timeline"
    ).json()
    measurements_by_key = {entry["key"]: entry for entry in payload["measurements"]}
    assert measurements_by_key["max_gap_ms"] == {
        "key": "max_gap_ms",
        "value": 220.0,
        "unit": "ms",
    }
    assert measurements_by_key["black_pct"]["unit"] == "%"
    assert measurements_by_key["episode_duration_s"]["unit"] == "s"
    # A key with no recognized unit suffix gets no invented dimension.
    assert measurements_by_key["max_velocity"]["unit"] is None
    assert [entry["key"] for entry in payload["measurements"]] == sorted(measurements_by_key)


def test_timeline_from_a_duration_measurement_alone_is_zero_based(
    timeline_api: TestClient, timeline_workspace: dict[str, str]
) -> None:
    payload = timeline_api.get(
        f"/api/v1/episodes/{timeline_workspace['duration_only']}/timeline"
    ).json()
    assert payload["intervals"] == []
    # No intervals to anchor the axis: an unsuffixed duration key reads as
    # seconds and the axis starts at zero.
    assert payload["start_ns"] == 0
    assert payload["end_ns"] == 30 * NANOSECONDS_PER_SECOND
    assert payload["duration_s"] == pytest.approx(30.0)


def test_a_duration_key_measuring_a_percentage_does_not_become_the_axis(
    timeline_api: TestClient, timeline_workspace: dict[str, str]
) -> None:
    """The bar's unit and the axis must read the same key the same way.

    ``duty_cycle_duration_pct`` is labelled "45 %"; reading its suffix as a
    time as well would claim a 45-second episode -- a fabricated axis 1e9
    times the number's own dimension.
    """
    payload = timeline_api.get(
        f"/api/v1/episodes/{timeline_workspace['duration_named_percentage']}/timeline"
    ).json()
    assert payload["measurements"] == [
        {"key": "duty_cycle_duration_pct", "value": 45.0, "unit": "%"}
    ]
    assert payload["start_ns"] is None
    assert payload["end_ns"] is None
    assert payload["duration_s"] is None


def test_timeline_without_any_span_returns_nulls_not_a_guess(
    timeline_api: TestClient, timeline_workspace: dict[str, str]
) -> None:
    payload = timeline_api.get(f"/api/v1/episodes/{timeline_workspace['no_span']}/timeline").json()
    assert payload["start_ns"] is None
    assert payload["end_ns"] is None
    assert payload["duration_s"] is None
    assert payload["intervals"] == []
    # The bars still work without an axis -- and text measurements are not bars.
    assert payload["measurements"] == [{"key": "frame_count", "value": 900.0, "unit": "count"}]


def test_timeline_over_the_shared_fixture_drops_illegal_doubles(
    api: TestClient, populated_workspace: PopulatedWorkspace
) -> None:
    payload = api.get(f"/api/v1/episodes/{populated_workspace.ok_episode_id}/timeline").json()
    assert payload["start_ns"] == 0
    assert payload["end_ns"] == 100
    assert payload["duration_s"] == pytest.approx(100 / NANOSECONDS_PER_SECOND)
    assert [interval["kind"] for interval in payload["intervals"]] == ["span"]
    # NaN/inf are illegal in JSON and meaningless as bars: both are dropped,
    # and the artifact URI measurement is text, not a bar.
    assert [entry["key"] for entry in payload["measurements"]] == ["max_velocity"]
    assert payload["measurements"][0]["value"] == pytest.approx(2.0)


def test_timeline_of_an_episode_without_evidence_is_all_nulls(
    api: TestClient, populated_workspace: PopulatedWorkspace
) -> None:
    payload = api.get(f"/api/v1/episodes/{populated_workspace.minimal_episode_id}/timeline").json()
    assert payload == {
        "start_ns": None,
        "end_ns": None,
        "duration_s": None,
        "intervals": [],
        "measurements": [],
    }


def test_timeline_of_an_unknown_episode_is_a_404(api: TestClient) -> None:
    response = api.get("/api/v1/episodes/does-not-exist/timeline")
    assert response.status_code == 404
    assert "does-not-exist" in response.json()["detail"]


def test_timeline_without_a_catalog_is_a_404_not_a_500(
    empty_workspace_api: TestClient,
) -> None:
    response = empty_workspace_api.get("/api/v1/episodes/anything/timeline")
    assert response.status_code == 404
    assert "Traceback" not in response.text
