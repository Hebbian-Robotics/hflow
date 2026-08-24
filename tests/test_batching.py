"""Byte-balanced bin-packing and staggered batch starts."""

from pathlib import Path

import pytest

from hflow.batching import plan_batches, plan_batches_from_files


def test_fixed_count_batches_are_near_equal_bytes() -> None:
    # Widely varying sizes (the scenario in Dyna's article): a naive
    # round-robin would unbalance workers badly; least-loaded keeps spread
    # within one item.
    item_sizes = {
        f"ep{index:02d}": size
        for index, size in enumerate([900, 850, 500, 400, 300, 250, 200, 150, 100, 50])
    }
    batches = plan_batches(item_sizes, batch_count=3)
    assert len(batches) == 3
    totals = [batch.total_bytes for batch in batches]
    assert sum(totals) == sum(item_sizes.values())
    assert max(totals) - min(totals) <= max(item_sizes.values())
    assert {item for batch in batches for item in batch.items} == set(item_sizes)


def test_plan_is_deterministic() -> None:
    item_sizes = {f"ep{index}": 100 for index in range(9)}
    assert plan_batches(item_sizes, batch_count=3) == plan_batches(item_sizes, batch_count=3)


def test_capacity_mode_opens_new_batches_and_handles_oversize_items() -> None:
    item_sizes = {"big": 1000, "a": 400, "b": 350, "c": 300}
    batches = plan_batches(item_sizes, target_batch_bytes=700)
    assert batches[0].items == ("big",)
    for batch in batches[1:]:
        assert batch.total_bytes <= 700
    assert {item for batch in batches for item in batch.items} == set(item_sizes)


def test_stagger_interval_spaces_start_delays() -> None:
    item_sizes = {"a": 1, "b": 2, "c": 3, "d": 4}
    batches = plan_batches(item_sizes, batch_count=4, stagger_interval_s=2.5)
    assert [batch.start_delay_s for batch in batches] == [0.0, 2.5, 5.0, 7.5]


def test_more_batches_than_items_collapses() -> None:
    batches = plan_batches({"only": 10}, batch_count=5)
    assert len(batches) == 1


def test_argument_validation() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        plan_batches({"a": 1})
    with pytest.raises(ValueError, match="exactly one"):
        plan_batches({"a": 1}, batch_count=1, target_batch_bytes=1)
    with pytest.raises(ValueError, match="negative"):
        plan_batches({"a": -1}, batch_count=1)
    with pytest.raises(ValueError, match="batch_count must be >= 1, got 0"):
        plan_batches({"a": 1}, batch_count=0)
    with pytest.raises(ValueError, match="target_batch_bytes must be >= 1, got 0"):
        plan_batches({"a": 1}, target_batch_bytes=0)
    assert plan_batches({}, batch_count=3) == []


def test_plan_from_real_files(tmp_path: Path) -> None:
    small = tmp_path / "small.mcap"
    small.write_bytes(b"x" * 100)
    large = tmp_path / "large.mcap"
    large.write_bytes(b"x" * 10_000)
    batches = plan_batches_from_files([small, large], batch_count=2)
    assert batches[0].total_bytes == 10_000
    assert batches[0].items == (str(large),)
