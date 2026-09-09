"""Weighted summaries of numeric measurements without acceptance thresholds."""

import math
from bisect import bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise


def _finite_number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        parsed_value = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(parsed_value):
        raise ValueError(f"{name} must be a finite number")
    return parsed_value


@dataclass(frozen=True)
class WeightedValue:
    """A finite measurement with an explicit, finite, nonnegative weight."""

    value: float
    weight: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _finite_number(self.value, "value"))
        parsed_weight = _finite_number(self.weight, "weight")
        if parsed_weight < 0:
            raise ValueError("weight must be nonnegative")
        object.__setattr__(self, "weight", parsed_weight)


@dataclass(frozen=True)
class WeightedPercentile:
    """A requested percentile in [0, 100] and its weighted nearest-rank value."""

    percentile: float
    value: float


@dataclass(frozen=True)
class WeightedHistogramBin:
    """Weight and positive-weight sample count within one histogram interval."""

    lower_bound: float
    upper_bound: float
    weight: float
    sample_count: int


@dataclass(frozen=True)
class WeightedDistribution:
    """A summary of assessed measurements, retaining the caller's weight units."""

    sample_count: int
    total_weight: float
    mean: float
    percentiles: tuple[WeightedPercentile, ...]
    histogram: tuple[WeightedHistogramBin, ...]


def _weighted_mean(ordered_values: Sequence[WeightedValue], total_weight: float) -> float:
    total_mantissa, total_exponent = math.frexp(total_weight)
    contributions: list[float] = []
    subnormal_units: list[float] = []
    for observation in ordered_values:
        value_mantissa, value_exponent = math.frexp(observation.value)
        weight_mantissa, weight_exponent = math.frexp(observation.weight)
        # Neither value * weight nor weight / total must be representable on
        # its own: combine exponents before rounding the final contribution.
        contribution_mantissa = value_mantissa * weight_mantissa / total_mantissa
        contribution_exponent = value_exponent + weight_exponent - total_exponent
        if contribution_exponent < -1022:
            # Sum tiny contributions in units of the smallest float before
            # rounding them; individually they may all round down to zero.
            subnormal_units.append(math.ldexp(contribution_mantissa, contribution_exponent + 1074))
            continue
        try:
            contribution = math.ldexp(contribution_mantissa, contribution_exponent)
        except OverflowError:
            # A weight never exceeds the total; only rounding at the largest
            # finite float can make its contribution overflow.
            contribution = observation.value
        contributions.append(contribution)
    contributions.append(math.ldexp(math.fsum(subnormal_units), -1074))

    minimum_value = ordered_values[0].value
    maximum_value = ordered_values[-1].value
    try:
        mean = math.fsum(contributions)
    except OverflowError:
        # The mathematical mean is bounded by the observations. Scale only
        # this near-overflow case, retaining tiny contributions otherwise.
        value_scale = max(abs(minimum_value), abs(maximum_value))
        scaled_mean = math.fsum(contribution / value_scale for contribution in contributions)
        mean = value_scale * min(1.0, max(-1.0, scaled_mean))
    return min(maximum_value, max(minimum_value, mean))


def _weighted_percentiles(
    ordered_values: Sequence[WeightedValue],
    total_weight: float,
    percentiles: Sequence[float],
) -> tuple[WeightedPercentile, ...]:
    cumulative_weights: list[float] = []
    cumulative_weight = 0.0
    rounding_correction = 0.0
    for observation in ordered_values:
        # Compensate small increments so many small weights are not discarded.
        corrected_weight = observation.weight - rounding_correction
        updated_weight = cumulative_weight + corrected_weight
        rounding_correction = (updated_weight - cumulative_weight) - corrected_weight
        cumulative_weight = updated_weight
        cumulative_weights.append(cumulative_weight)

    results: list[WeightedPercentile] = []
    for percentile in percentiles:
        if percentile == 0:
            percentile_value = ordered_values[0].value
        elif percentile == 100:
            percentile_value = ordered_values[-1].value
        else:
            target_weight = total_weight * (percentile / 100)
            percentile_value = ordered_values[-1].value
            for observation, assessed_weight in zip(
                ordered_values, cumulative_weights, strict=True
            ):
                if math.nextafter(assessed_weight, math.inf) >= target_weight:
                    percentile_value = observation.value
                    break
        results.append(WeightedPercentile(percentile, percentile_value))
    return tuple(results)


def summarize_weighted_distribution(
    values: Iterable[WeightedValue],
    *,
    bin_edges: Sequence[float],
    percentiles: Sequence[float] = (0, 25, 50, 75, 100),
) -> WeightedDistribution | None:
    """Summarize measurements with explicit weights, bins, and percentiles.

    Weights may represent duration, frame counts, or another common unit.
    Omit missing measurements: they are not zero-valued observations. Zero
    weights contribute neither counts nor statistics; no positive weights
    returns ``None``. Configuration is validated even for empty input.

    Bin edges must be finite and strictly increasing. Every positive-weight
    value must be within their inclusive outer range. Bins include their lower
    edge and exclude their upper edge, except the final bin includes both.

    Percentiles use weighted nearest rank: the smallest value whose cumulative
    weight reaches the requested percentage of total weight. Zero and 100
    return the minimum and maximum; requested order and duplicates are retained.
    A one-ULP allowance at cumulative-weight boundaries absorbs floating-point
    rounding. Combine original observations across batches, not their percentiles.
    """
    validated_edges = tuple(_finite_number(edge, "bin edge") for edge in bin_edges)
    if len(validated_edges) < 2 or any(
        lower >= upper for lower, upper in pairwise(validated_edges)
    ):
        raise ValueError("bin_edges must contain at least two strictly increasing edges")
    validated_percentiles = tuple(
        _finite_number(percentile, "percentile") for percentile in percentiles
    )
    if any(percentile < 0 or percentile > 100 for percentile in validated_percentiles):
        raise ValueError("percentiles must be between 0 and 100")

    ordered_values = sorted(
        (observation for observation in values if observation.weight > 0),
        key=lambda observation: (observation.value, observation.weight),
    )
    if not ordered_values:
        return None
    if (
        ordered_values[0].value < validated_edges[0]
        or ordered_values[-1].value > validated_edges[-1]
    ):
        raise ValueError("positive-weight values must lie within the bin_edges range")
    try:
        total_weight = math.fsum(observation.weight for observation in ordered_values)
    except OverflowError as error:
        raise ValueError("total weight must be finite") from error

    histogram_weights: list[list[float]] = [[] for _ in validated_edges[1:]]
    for observation in ordered_values:
        bin_index = min(
            bisect_right(validated_edges, observation.value) - 1, len(histogram_weights) - 1
        )
        histogram_weights[bin_index].append(observation.weight)
    histogram = tuple(
        WeightedHistogramBin(
            lower_bound=validated_edges[bin_index],
            upper_bound=validated_edges[bin_index + 1],
            weight=math.fsum(weights),
            sample_count=len(weights),
        )
        for bin_index, weights in enumerate(histogram_weights)
    )

    return WeightedDistribution(
        sample_count=len(ordered_values),
        total_weight=total_weight,
        mean=_weighted_mean(ordered_values, total_weight),
        percentiles=_weighted_percentiles(ordered_values, total_weight, validated_percentiles),
        histogram=histogram,
    )
