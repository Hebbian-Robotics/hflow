"""Weighted measurements retain coverage, duration, and cross-batch semantics."""

import itertools
import math

import pytest

from hflow import (
    WeightedHistogramBin,
    WeightedPercentile,
    WeightedValue,
    summarize_weighted_distribution,
)


def test_summary_uses_weights_for_mean_histogram_and_nearest_rank() -> None:
    distribution = summarize_weighted_distribution(
        [WeightedValue(0, 1), WeightedValue(50, 2), WeightedValue(100, 7)],
        bin_edges=(0, 50, 100),
    )
    assert distribution is not None
    assert distribution.sample_count == 3
    assert distribution.total_weight == 10
    assert distribution.mean == pytest.approx(80)
    assert distribution.percentiles == (
        WeightedPercentile(0, 0),
        WeightedPercentile(25, 50),
        WeightedPercentile(50, 100),
        WeightedPercentile(75, 100),
        WeightedPercentile(100, 100),
    )
    assert distribution.histogram == (
        WeightedHistogramBin(0, 50, 1, 1),
        WeightedHistogramBin(50, 100, 9, 2),
    )


def test_histogram_compares_boundaries_exactly_and_includes_final_edge() -> None:
    distribution = summarize_weighted_distribution(
        [WeightedValue(value, 1) for value in (0, math.nextafter(10, 0), 10, 100)],
        bin_edges=(0, 10, 20, 100),
    )
    assert distribution is not None
    assert distribution.histogram == (
        WeightedHistogramBin(0, 10, 2, 2),
        WeightedHistogramBin(10, 20, 1, 1),
        WeightedHistogramBin(20, 100, 1, 1),
    )


def test_zero_weights_do_not_change_coverage_or_statistics() -> None:
    assessed_values = [WeightedValue(40, 3), WeightedValue(60, 1)]
    assert summarize_weighted_distribution(
        [WeightedValue(-999, 0), *assessed_values, WeightedValue(999, 0)],
        bin_edges=(0, 50, 100),
    ) == summarize_weighted_distribution(assessed_values, bin_edges=(0, 50, 100))
    assert summarize_weighted_distribution([], bin_edges=(0, 100)) is None
    assert summarize_weighted_distribution([WeightedValue(50, 0)], bin_edges=(0, 100)) is None


def test_generic_range_and_requested_percentile_order_are_preserved() -> None:
    distribution = summarize_weighted_distribution(
        [WeightedValue(-273, 0.5), WeightedValue(10, 1.5), WeightedValue(150, 2)],
        bin_edges=(-300, 0, 20, 200),
        percentiles=(100, 12.5, 50, 0, 50),
    )
    assert distribution is not None
    assert distribution.mean == pytest.approx(44.625)
    assert distribution.percentiles == (
        WeightedPercentile(100, 150),
        WeightedPercentile(12.5, -273),
        WeightedPercentile(50, 10),
        WeightedPercentile(0, -273),
        WeightedPercentile(50, 10),
    )


def test_permutations_and_concatenated_batches_produce_the_same_distribution() -> None:
    first_batch = [WeightedValue(10, 0.1), WeightedValue(80, 0.3)]
    second_batch = [WeightedValue(10, 0.2), WeightedValue(90, 0.4)]
    observations = [*first_batch, *second_batch]
    expected = summarize_weighted_distribution(observations, bin_edges=(0, 20, 100))
    assert expected is not None
    assert expected.percentiles[2] == WeightedPercentile(50, 80)
    for permutation in itertools.permutations(observations):
        assert summarize_weighted_distribution(permutation, bin_edges=(0, 20, 100)) == expected
    assert (
        summarize_weighted_distribution(
            itertools.chain(first_batch, second_batch), bin_edges=(0, 20, 100)
        )
        == expected
    )


def test_nearest_rank_preserves_exact_cumulative_weight_boundaries() -> None:
    distribution = summarize_weighted_distribution(
        [WeightedValue(10, 0.1), WeightedValue(20, 0.2), WeightedValue(30, 0.7)],
        bin_edges=(0, 100),
        percentiles=(10, 30, 30.01, 100),
    )
    assert distribution is not None
    assert distribution.percentiles == (
        WeightedPercentile(10, 10),
        WeightedPercentile(30, 20),
        WeightedPercentile(30.01, 30),
        WeightedPercentile(100, 30),
    )
    rounded_boundary = summarize_weighted_distribution(
        [WeightedValue(10, 0.3), WeightedValue(20, 0.6), WeightedValue(30, 0.1)],
        bin_edges=(0, 100),
        percentiles=(90,),
    )
    assert rounded_boundary is not None
    assert rounded_boundary.percentiles == (WeightedPercentile(90, 20),)


def test_mean_handles_large_finite_values_without_overflow() -> None:
    distribution = summarize_weighted_distribution(
        [WeightedValue(-1e308, 1e100), WeightedValue(1e308, 1e100)],
        bin_edges=(-1e308, 0, 1e308),
    )
    assert distribution is not None
    assert distribution.mean == 0
    assert distribution.total_weight == 2e100


@pytest.mark.parametrize(
    ("observations", "expected_mean"),
    [
        ([WeightedValue(0, 1e308), WeightedValue(1e308, 1e-308)], 1e-308),
        ([WeightedValue(0, 1e-308), WeightedValue(1e-308, 1e-308)], 5e-309),
        ([WeightedValue(0, 1e308), WeightedValue(-1e308, 1e-308)], -1e-308),
        ([WeightedValue(1e308, 1e308)], 1e308),
        (
            [WeightedValue(-1e308, 1), WeightedValue(1e-308, 1), WeightedValue(1e308, 1)],
            1e-308 / 3,
        ),
    ],
)
def test_mean_preserves_representable_results_across_large_numeric_ranges(
    observations: list[WeightedValue], expected_mean: float
) -> None:
    distribution = summarize_weighted_distribution(observations, bin_edges=(-1e308, 1e308))
    assert distribution is not None
    assert distribution.mean == pytest.approx(expected_mean, rel=1e-14, abs=0)


def test_mean_stays_finite_and_bounded_near_largest_representable_value() -> None:
    maximum_float = float.fromhex("0x1.fffffffffffffp+1023")
    distribution = summarize_weighted_distribution(
        [WeightedValue(maximum_float, weight) for weight in (0.1, 0.2, 0.3, 0.4)],
        bin_edges=(0, maximum_float),
    )
    assert distribution is not None
    assert distribution.mean == maximum_float


def test_mean_combines_tiny_contributions_before_rounding_to_smallest_float() -> None:
    smallest_float = math.ulp(0.0)
    distribution = summarize_weighted_distribution(
        [WeightedValue(0, 1), *(WeightedValue(smallest_float, 1) for _ in range(3))],
        bin_edges=(0, 1),
    )
    assert distribution is not None
    assert distribution.mean == smallest_float


@pytest.mark.parametrize("invalid_value", [True, "1", None, math.inf, -math.inf, math.nan])
def test_measurements_reject_nonfinite_or_nonnumeric_values(invalid_value: object) -> None:
    with pytest.raises(ValueError, match="value must be a finite number"):
        WeightedValue(invalid_value, 1)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("invalid_weight", [True, "1", None, math.inf, -math.inf, math.nan, -1])
def test_measurements_reject_invalid_weights(invalid_weight: object) -> None:
    with pytest.raises(ValueError, match="weight must be"):
        WeightedValue(1, invalid_weight)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("bin_edges", [(), (0,), (0, 0), (10, 0), (0, math.nan), (0, math.inf)])
def test_invalid_histogram_configuration_is_rejected_even_without_data(
    bin_edges: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError, match="bin"):
        summarize_weighted_distribution([], bin_edges=bin_edges)


@pytest.mark.parametrize("percentile", [-1, 101, math.nan, math.inf, True])
def test_invalid_percentiles_are_rejected_even_without_data(percentile: float) -> None:
    with pytest.raises(ValueError, match="percentile"):
        summarize_weighted_distribution([], bin_edges=(0, 100), percentiles=(percentile,))


@pytest.mark.parametrize("value", [-1, 101])
def test_positive_weight_values_outside_histogram_range_are_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="within the bin_edges range"):
        summarize_weighted_distribution([WeightedValue(value, 1)], bin_edges=(0, 100))


def test_total_weight_must_be_representable_as_a_finite_number() -> None:
    with pytest.raises(ValueError, match="total weight must be finite"):
        summarize_weighted_distribution(
            [WeightedValue(10, 1e308), WeightedValue(20, 1e308)], bin_edges=(0, 100)
        )
