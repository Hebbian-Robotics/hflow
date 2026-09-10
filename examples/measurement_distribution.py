"""Summarize numeric observations without files, services, or quality verdicts.

Run from the repository root:
    uv run python examples/measurement_distribution.py
"""

import json
from dataclasses import asdict

from hflow import WeightedValue, summarize_weighted_distribution


def main() -> None:
    observations = (
        WeightedValue(value=10.0, weight=90.0),
        WeightedValue(value=90.0, weight=10.0),
    )
    distribution = summarize_weighted_distribution(
        observations,
        bin_edges=(0, 20, 40, 60, 80, 100),
        percentiles=(0, 25, 50, 75, 90, 95, 100),
    )
    assert distribution is not None
    print(json.dumps(asdict(distribution), indent=2))


if __name__ == "__main__":
    main()
