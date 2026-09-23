"""Bounded prefetch overlaps preparation with consumption without leaking scratch space."""

import asyncio
import threading
import time
from pathlib import Path
from typing import Any, cast

import pytest

from hflow.asyncio_utils import PrefetchedItem, prefetch


def _write_marker(item: int, directory: Path) -> Path:
    marker = directory / f"item-{item}.txt"
    marker.write_text(str(item))
    return marker


def test_items_arrive_in_order_and_only_the_held_item_and_lookahead_occupy_disk(
    tmp_path: Path,
) -> None:
    held_directory_counts: list[int] = []

    async def scenario() -> list[PrefetchedItem[int, Path]]:
        received: list[PrefetchedItem[int, Path]] = []
        async with prefetch(
            range(5), _write_marker, working_directory=tmp_path, lookahead=2
        ) as prepared_items:
            async for prepared in prepared_items:
                assert prepared.value.read_text() == str(prepared.item)
                held_directory_counts.append(len(tuple(tmp_path.iterdir())))
                received.append(prepared)
        return received

    received = asyncio.run(scenario())
    assert [prepared.item for prepared in received] == [0, 1, 2, 3, 4]
    assert max(held_directory_counts) <= 3
    assert all(not prepared.directory.exists() for prepared in received)
    assert not tuple(tmp_path.iterdir())


def test_next_item_prepares_while_the_caller_waits_on_the_current_one(tmp_path: Path) -> None:
    second_item_prepared = threading.Event()

    def prepare(item: int, directory: Path) -> int:
        if item == 1:
            second_item_prepared.set()
        return item

    async def scenario() -> None:
        async with prefetch(
            range(3), prepare, working_directory=tmp_path, lookahead=1
        ) as prepared_items:
            first = await anext(prepared_items)
            assert first.item == 0
            # Simulated model request for item 0: item 1 must prepare meanwhile.
            prepared_during_request = await asyncio.to_thread(second_item_prepared.wait, 10)
            assert prepared_during_request

    asyncio.run(scenario())


def test_zero_lookahead_prepares_only_on_request(tmp_path: Path) -> None:
    prepared_items_log: list[int] = []

    def prepare(item: int, directory: Path) -> int:
        prepared_items_log.append(item)
        return item

    async def scenario() -> None:
        async with prefetch(
            range(3), prepare, working_directory=tmp_path, lookahead=0
        ) as prepared_items:
            await anext(prepared_items)
            await asyncio.sleep(0.05)
            assert prepared_items_log == [0]

    asyncio.run(scenario())


def test_leaving_early_runs_the_hook_and_removes_every_directory(tmp_path: Path) -> None:
    hook_calls: list[str] = []

    async def stop_reader() -> None:
        hook_calls.append("stopped")

    async def scenario() -> None:
        async with prefetch(
            range(10),
            _write_marker,
            working_directory=tmp_path,
            lookahead=2,
            cancel_hook=stop_reader,
        ) as prepared_items:
            async for prepared in prepared_items:
                if prepared.item == 1:
                    break

    asyncio.run(scenario())
    assert hook_calls == ["stopped"]
    assert not tuple(tmp_path.iterdir())


def test_exhausting_the_items_releases_everything_without_the_hook(tmp_path: Path) -> None:
    hook_calls: list[str] = []

    async def stop_reader() -> None:
        hook_calls.append("stopped")

    async def scenario() -> None:
        async with prefetch(
            range(3), _write_marker, working_directory=tmp_path, cancel_hook=stop_reader
        ) as prepared_items:
            async for _prepared in prepared_items:
                pass

    asyncio.run(scenario())
    assert hook_calls == []
    assert not tuple(tmp_path.iterdir())


def test_cancelling_the_caller_stops_the_reader_before_draining_preparation(
    tmp_path: Path,
) -> None:
    # Item 1's preparation waits on an external reader that only the hook stops.
    reader_stopped = threading.Event()
    lifecycle: list[str] = []

    def prepare(item: int, directory: Path) -> int:
        if item == 1:
            if not reader_stopped.wait(timeout=10):
                raise TimeoutError("cancel hook never stopped the reader")
            (directory / "late-write.txt").write_text("written before cleanup")
            lifecycle.append("preparation drained")
        return item

    async def stop_reader() -> None:
        lifecycle.append("hook ran")
        reader_stopped.set()

    async def consume() -> None:
        async with prefetch(
            range(3),
            prepare,
            working_directory=tmp_path,
            lookahead=1,
            cancel_hook=stop_reader,
        ) as prepared_items:
            await anext(prepared_items)
            await asyncio.Event().wait()  # a model request that never returns

    async def scenario() -> None:
        consumer = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(consumer, timeout=10)

    asyncio.run(scenario())
    assert lifecycle == ["hook ran", "preparation drained"]
    assert not tuple(tmp_path.iterdir())


def test_preparation_failure_surfaces_at_its_item_after_earlier_items(tmp_path: Path) -> None:
    def prepare(item: int, directory: Path) -> int:
        if item == 2:
            raise ValueError("window 2 is unreadable")
        return item

    async def scenario() -> list[int]:
        received: list[int] = []
        with pytest.raises(ValueError, match="window 2 is unreadable"):
            async with prefetch(
                range(5), prepare, working_directory=tmp_path, lookahead=2
            ) as prepared_items:
                async for prepared in prepared_items:
                    received.append(prepared.item)
        return received

    assert asyncio.run(scenario()) == [0, 1]
    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize("lookahead", [-1, True, 1.5])
def test_invalid_lookahead_is_rejected(tmp_path: Path, lookahead: object) -> None:
    with pytest.raises(ValueError, match="lookahead"):
        prefetch(
            range(1), _write_marker, working_directory=tmp_path, lookahead=cast(Any, lookahead)
        )


@pytest.mark.parametrize("lookahead", [0, 1, 3])
def test_running_preparations_never_exceed_the_lookahead(tmp_path: Path, lookahead: int) -> None:
    running_lock = threading.Lock()
    running_count = 0
    peak_running_count = 0

    def slow_prepare(item: int, directory: Path) -> int:
        nonlocal running_count, peak_running_count
        with running_lock:
            running_count += 1
            peak_running_count = max(peak_running_count, running_count)
        time.sleep(0.03)
        with running_lock:
            running_count -= 1
        return item

    async def scenario() -> None:
        async with prefetch(
            range(8), slow_prepare, working_directory=tmp_path, lookahead=lookahead
        ) as prepared_items:
            async for _prepared in prepared_items:
                await asyncio.sleep(0.01)

    asyncio.run(scenario())
    # Reaching the bound shows the overlap; never exceeding it bounds CPU use.
    assert peak_running_count == max(lookahead, 1)
