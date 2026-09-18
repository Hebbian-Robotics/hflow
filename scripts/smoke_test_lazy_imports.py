"""Verify utility consumers can run without initializing pipeline dependencies."""

import sys

import hflow


def smoke_test_lazy_imports() -> None:
    """Exercise utility results and the import boundary in a fresh interpreter."""
    # Access modules first: later symbol imports also populate package attributes.
    batching_module = hflow.batching
    statistics_module = hflow.statistics
    from hflow import (
        WeightedValue,
        plan_batches,
        plan_source_windows,
        summarize_weighted_distribution,
    )

    assert plan_batches is batching_module.plan_batches
    assert WeightedValue is statistics_module.WeightedValue
    batches = plan_batches(
        {"first": 30, "second": 20, "third": 10},
        target_batch_bytes=100,
        maximum_items_per_batch=2,
    )
    assert [batch.items for batch in batches] == [("first", "second"), ("third",)]
    distribution = summarize_weighted_distribution(
        [WeightedValue(20, 1), WeightedValue(80, 3)], bin_edges=(0, 50, 100)
    )
    assert distribution is not None
    assert distribution.mean == 65
    assert distribution.total_weight == 4
    windows = plan_source_windows(120_001, maximum_window_millis=120_000)
    assert [window.duration_millis for window in windows] == [60_001, 60_000]
    assert "App" in dir(hflow)
    assert not hasattr(hflow, "missing_public_export")
    allowed_modules = {
        "hflow",
        "hflow._version",
        "hflow._field_guards",
        "hflow.batching",
        "hflow.statistics",
        "hflow.source_windows",
    }
    loaded_modules = {
        module_name
        for module_name in sys.modules
        if module_name == "hflow" or module_name.startswith("hflow.")
    }
    assert loaded_modules <= allowed_modules, loaded_modules - allowed_modules


if __name__ == "__main__":
    smoke_test_lazy_imports()
