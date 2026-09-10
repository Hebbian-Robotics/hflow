# Summarize weighted measurements

Use this guide to compare the spread of numeric measurements across episodes
or windows without choosing a pass/fail threshold. It works with measurements
from ordinary Python workers, a catalog query, or another scheduler.

The [runnable example](../../examples/measurement_distribution.py) needs only
the normal HFlow development environment. It reads no recordings, writes no
files, and makes no network or model calls:

```bash
uv run python examples/measurement_distribution.py
```

The example prints a weighted mean of 18, a median of 10, and a 95th percentile
of 90. The two observations have weights of 90 and 10, so treating them as
equally important would incorrectly produce a mean of 50.

## Choose the observation and its weight

`WeightedValue` carries a finite numeric value and a nonnegative numeric
weight. HFlow does not guess either from a measurement's name:

- Use assessed duration for a duration-weighted distribution of window scores.
- Use assessed sample counts when combining rates measured over samples.
- Use a weight of one when each observation should contribute equally.

Only combine values with the same meaning, scale, and observation unit. A
distribution of window scores is not a distribution of individual frames or
source recordings. Group windows by recording first if recordings are the
unit you want to compare.

```python
from hflow import WeightedValue, summarize_weighted_distribution

summary = summarize_weighted_distribution(
    [WeightedValue(value=10, weight=90), WeightedValue(value=90, weight=10)],
    bin_edges=(0, 20, 40, 60, 80, 100),
    percentiles=(0, 50, 95, 100),
)
```

The range need not be a percentage: temperatures, durations, and other numeric
measurements work with caller-supplied bin edges in their own units. The helper
does not change `CheckResult`, run checks, or record anything in the catalog.

## Read the result

A nonempty `WeightedDistribution` contains:

- `sample_count`: the number of positive-weight observations;
- `total_weight` and `mean`;
- `percentiles`: each requested rank and its observed value;
- `histogram`: every bin's bounds, total weight, and unweighted sample count,
  including empty bins.

Percentiles use weighted nearest rank: the smallest observed value whose
cumulative weight reaches the requested fraction of the total. There is no
interpolation between values. The endpoint ranks return the minimum and
maximum positive-weight values.

Bins include their lower bound and exclude their upper bound, except the final
bin includes the maximum edge. Bounds are exact; the helper does not round or
silently clamp measurements. Round only when formatting the consumer's report.

To display a bin's fraction of assessed footage, divide its `weight` by
`total_weight`. Its `sample_count` answers a different question: how many
windows contributed, regardless of their durations.

## Preserve missing evidence and worker boundaries

Omit unassessed measurements. Do not replace them with zero: an assessed zero
is valid evidence and belongs in the distribution. Zero-weight observations
contribute to neither counts nor statistics. Empty or all-zero-weight input
returns `None`, not a fabricated zero-valued summary. Invalid numeric values,
weights, percentile ranks, or histogram bounds raise an error.

Combine workers' underlying weighted observations before calculating an exact
dataset distribution. Averaging worker medians or percentiles does not produce
the dataset's median or percentiles. This helper materializes and sorts its
positive-weight observations; it is an exact in-memory reducer, not a
bounded-memory streaming approximation or persistence layer.
