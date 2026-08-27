"""Byte-based batch planning with staggered starts.

Two scheduler-protecting measures, applied at plan time: inputs are
bin-packed into batches of near-equal *bytes* (file sizes vary widely, so
count-based batches unbalance workers), and batch starts are staggered to
smooth scheduler write bursts. Dyna's article reports both as the fixes
when the scheduler database became the choke point at scale. These are the
simple v1 versions of both; the joint optimizer tuning worker counts
against network/IO/DB constraints stays on the scale path.

Two planning modes, chosen by which parameter you pass:

- ``batch_count``: partition into exactly N batches with near-equal bytes
  (longest-first, least-loaded assignment -- the fixed-bin-count variant of
  first-fit-decreasing).
- ``target_batch_bytes``: classic capacity-based first-fit-decreasing; a new
  batch opens when nothing fits, and an item larger than the capacity gets a
  batch of its own.

Staggering is deterministic and even (``stagger_interval_s`` seconds between
consecutive batch starts): even spacing smooths scheduler write bursts at
least as well as random jitter and reproduces exactly.
"""

import heapq
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path


@dataclass(frozen=True)
class PlannedBatch:
    """One near-equal-bytes group of inputs plus its staggered start delay."""

    items: tuple[str, ...]
    total_bytes: int
    start_delay_s: float


def plan_batches(
    item_sizes: Mapping[str, int],
    *,
    batch_count: int | None = None,
    target_batch_bytes: int | None = None,
    stagger_interval_s: float = 0.0,
) -> list[PlannedBatch]:
    """Pack ``item_sizes`` (uri -> size in bytes) into near-equal-byte batches.

    Pass exactly one of ``batch_count`` or ``target_batch_bytes``. Batches
    come back largest-first; ``start_delay_s`` is ``index *
    stagger_interval_s``. Deterministic for a given input.
    """
    if (batch_count is None) == (target_batch_bytes is None):
        raise ValueError("pass exactly one of batch_count or target_batch_bytes")
    for uri, size_bytes in item_sizes.items():
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool):
            raise ValueError(
                f"item {uri!r} has type {type(size_bytes).__name__}, expected int bytes"
            )
        if size_bytes < 0:
            raise ValueError(f"item {uri!r} has negative size {size_bytes}")

    if not isinstance(stagger_interval_s, Real) or isinstance(stagger_interval_s, bool):
        raise ValueError(
            f"stagger_interval_s must be a number, got {type(stagger_interval_s).__name__}"
        )
    if not math.isfinite(stagger_interval_s):
        raise ValueError(f"stagger_interval_s must be finite, got {stagger_interval_s}")
    if stagger_interval_s < 0:
        raise ValueError(f"stagger_interval_s must be nonnegative, got {stagger_interval_s}")

    if batch_count is not None:
        if not isinstance(batch_count, int) or isinstance(batch_count, bool):
            raise ValueError(f"batch_count must be an int, got {type(batch_count).__name__}")
        if batch_count < 1:
            raise ValueError(f"batch_count must be >= 1, got {batch_count}")
    else:
        if not isinstance(target_batch_bytes, int) or isinstance(target_batch_bytes, bool):
            raise ValueError(
                f"target_batch_bytes must be an int, got {type(target_batch_bytes).__name__}"
            )
        if target_batch_bytes < 1:
            raise ValueError(f"target_batch_bytes must be >= 1, got {target_batch_bytes}")

    if not item_sizes:
        return []

    # Largest first; uri tiebreak keeps the plan reproducible.
    descending_items = sorted(item_sizes.items(), key=lambda entry: (-entry[1], entry[0]))

    batches: list[tuple[list[str], int]]
    if batch_count is not None:
        # Least-loaded assignment: push each item onto the currently-lightest
        # batch. Heap entries are (total_bytes, batch_index) so byte ties
        # break on index, deterministically.
        batches = [([], 0) for _ in range(min(batch_count, len(descending_items)))]
        heap = [(0, index) for index in range(len(batches))]
        heapq.heapify(heap)
        for uri, size_bytes in descending_items:
            total, index = heapq.heappop(heap)
            batches[index][0].append(uri)
            batches[index] = (batches[index][0], total + size_bytes)
            heapq.heappush(heap, (total + size_bytes, index))
    else:
        assert target_batch_bytes is not None
        # First-fit-decreasing: first open batch with room, else a new one.
        # An item exceeding the capacity still gets (its own) batch.
        batches = []
        for uri, size_bytes in descending_items:
            for index, (uris, total) in enumerate(batches):
                if total + size_bytes <= target_batch_bytes:
                    uris.append(uri)
                    batches[index] = (uris, total + size_bytes)
                    break
            else:
                batches.append(([uri], size_bytes))

    ordered = sorted(
        (batch for batch in batches if batch[0]),
        key=lambda batch: (-batch[1], batch[0]),
    )
    return [
        PlannedBatch(
            items=tuple(uris),
            total_bytes=total,
            start_delay_s=index * stagger_interval_s,
        )
        for index, (uris, total) in enumerate(ordered)
    ]


def plan_batches_from_files(
    paths: Iterable[Path | str],
    *,
    batch_count: int | None = None,
    target_batch_bytes: int | None = None,
    stagger_interval_s: float = 0.0,
) -> list[PlannedBatch]:
    """:func:`plan_batches` over real files, sized with ``stat()``."""
    item_sizes = {str(path): Path(path).stat().st_size for path in paths}
    return plan_batches(
        item_sizes,
        batch_count=batch_count,
        target_batch_bytes=target_batch_bytes,
        stagger_interval_s=stagger_interval_s,
    )
