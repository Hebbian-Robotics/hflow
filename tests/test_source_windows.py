"""Source plans cover the whole recording without an undersized tail."""

from itertools import pairwise

import pytest

from hflow import SourceWindow, plan_source_windows


@pytest.mark.parametrize(
    ("duration_millis", "expected_bounds"),
    [
        (1, [(0, 1)]),
        (120_000, [(0, 120_000)]),
        (120_001, [(0, 60_001), (60_001, 120_001)]),
        (240_002, [(0, 80_001), (80_001, 160_002), (160_002, 240_002)]),
    ],
)
def test_planning_balances_the_tail_instead_of_discarding_it(
    duration_millis: int, expected_bounds: list[tuple[int, int]]
) -> None:
    windows = plan_source_windows(duration_millis, maximum_window_millis=120_000)
    assert [(window.start_millis, window.end_millis) for window in windows] == expected_bounds


@pytest.mark.parametrize("maximum_window_millis", [1, 7, 100, 120_000])
def test_plans_cover_sources_once_with_minimal_balanced_windows(maximum_window_millis: int) -> None:
    for duration_millis in range(1, 401):
        windows = plan_source_windows(duration_millis, maximum_window_millis=maximum_window_millis)
        durations = [window.duration_millis for window in windows]
        assert windows[0].start_millis == 0
        assert windows[-1].end_millis == duration_millis
        assert all(earlier.end_millis == later.start_millis for earlier, later in pairwise(windows))
        assert sum(durations) == duration_millis
        assert 1 <= min(durations) <= max(durations) <= maximum_window_millis
        assert max(durations) - min(durations) <= 1
        assert (
            len(windows) == (duration_millis + maximum_window_millis - 1) // maximum_window_millis
        )


def test_plans_refuse_partial_coverage_when_limits_cannot_be_satisfied() -> None:
    with pytest.raises(ValueError, match="minimum_window_millis"):
        plan_source_windows(101, maximum_window_millis=100, minimum_window_millis=60)
    with pytest.raises(ValueError, match="maximum_windows"):
        plan_source_windows(10**100, maximum_window_millis=100, maximum_windows=3)
    assert len(plan_source_windows(300, maximum_window_millis=100, maximum_windows=3)) == 3


@pytest.mark.parametrize("duration_millis", [True, 1.5, "100", 0, -1])
def test_invalid_durations_are_not_silently_coerced(duration_millis: object) -> None:
    with pytest.raises(ValueError):
        plan_source_windows(duration_millis, maximum_window_millis=100)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("bounds", [(-1, 1), (0, 0), (2, 1), (False, 1)])
def test_source_windows_require_a_nonempty_forward_interval(bounds: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        SourceWindow(*bounds)
