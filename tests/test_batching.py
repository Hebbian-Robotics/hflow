"""Byte-balanced bin-packing and staggered batch starts."""

from pathlib import Path
from typing import Any

import pytest

from hflow.batching import PlannedBatch, plan_batches, plan_batches_from_files


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


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"batch_count": True}, "batch_count must be an int, got bool"),
        ({"batch_count": 1.0}, "batch_count must be an int, got float"),
        ({"target_batch_bytes": False}, "target_batch_bytes must be an int, got bool"),
        ({"target_batch_bytes": "100"}, "target_batch_bytes must be an int, got str"),
    ],
)
def test_batch_parameters_require_real_integers(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        plan_batches({"a": 1}, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("size", "message"),
    [
        (True, "item 'a' has type bool, expected int bytes"),
        (1.0, "item 'a' has type float, expected int bytes"),
    ],
)
def test_item_sizes_require_real_integers(size: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        plan_batches({"a": size}, batch_count=1)  # type: ignore[dict-item]


@pytest.mark.parametrize(
    ("stagger", "message"),
    [
        (True, "stagger_interval_s must be a number, got bool"),
        ("1.0", "stagger_interval_s must be a number, got str"),
        (-1.0, "stagger_interval_s must be nonnegative, got -1.0"),
        (float("nan"), "stagger_interval_s must be finite, got nan"),
        (float("inf"), "stagger_interval_s must be finite, got inf"),
    ],
)
def test_stagger_interval_requires_a_finite_nonnegative_number(stagger: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        plan_batches({"a": 1}, batch_count=1, stagger_interval_s=stagger)  # type: ignore[arg-type]


def test_zero_size_items_and_zero_stagger_remain_valid() -> None:
    batches = plan_batches({"empty": 0}, batch_count=1, stagger_interval_s=0)
    assert batches == [PlannedBatch(items=("empty",), total_bytes=0, start_delay_s=0)]


def test_plan_from_files_forwards_stagger_validation(tmp_path: Path) -> None:
    item = tmp_path / "item.mcap"
    item.write_bytes(b"x")
    with pytest.raises(ValueError, match="stagger_interval_s must be finite"):
        plan_batches_from_files([item], batch_count=1, stagger_interval_s=float("nan"))


def test_plan_from_real_files(tmp_path: Path) -> None:
    small = tmp_path / "small.mcap"
    small.write_bytes(b"x" * 100)
    large = tmp_path / "large.mcap"
    large.write_bytes(b"x" * 10_000)
    batches = plan_batches_from_files([small, large], batch_count=2)
    assert batches[0].total_bytes == 10_000
    assert batches[0].items == (str(large),)
