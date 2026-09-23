"""Explicit blocking work inside async pipelines, with lifetime-safe cancellation."""

import asyncio
import shutil
import tempfile
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator
from contextlib import AbstractContextManager, asynccontextmanager, suppress
from contextvars import copy_context
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Generic, ParamSpec, TypeVar, cast

_Parameters = ParamSpec("_Parameters")
_Result = TypeVar("_Result")
_Item = TypeVar("_Item")
_Prepared = TypeVar("_Prepared")
_NO_MORE_ITEMS = object()


async def run_blocking(
    operation: Callable[_Parameters, _Result],
    *arguments: _Parameters.args,
    **keyword_arguments: _Parameters.kwargs,
) -> _Result:
    """Offload blocking work and drain it before cancellation releases its files.

    Threads cannot be forcibly stopped. HTTP and other cancellable I/O should
    use native async APIs instead; this adapter is for blocking media/file work.
    """
    return await run_blocking_with_cancel_hook(None, operation, *arguments, **keyword_arguments)


async def run_blocking_with_cancel_hook(
    cancel_hook: Callable[[], Awaitable[object]] | None,
    operation: Callable[_Parameters, _Result],
    *arguments: _Parameters.args,
    **keyword_arguments: _Parameters.kwargs,
) -> _Result:
    """Like :func:`run_blocking`, but run ``cancel_hook`` before draining on cancellation.

    Use the hook to stop something outside this thread that the blocking work
    waits on, such as a local model server reading the same files. Otherwise the
    drain could wait on that reader indefinitely. The hook runs to completion
    even if cancellation repeats; its failure never skips the drain or replaces
    the propagating cancellation.
    """
    # An executor Future is not part of asyncio.run's cancel-all-Tasks sweep.
    # A to_thread Task can become cancelled while its underlying thread still
    # owns episode files, falsely appearing drained during loop shutdown.
    operation_future = asyncio.get_running_loop().run_in_executor(
        None, copy_context().run, partial(operation, *arguments, **keyword_arguments)
    )
    try:
        return await asyncio.shield(operation_future)
    except asyncio.CancelledError:
        if cancel_hook is not None:
            await _run_cancel_hook_to_completion(cancel_hook)
        await _await_ignoring_cancellation(operation_future)
        raise


async def _run_cancel_hook_to_completion(
    cancel_hook: Callable[[], Awaitable[object]],
) -> None:
    async def invoke_cancel_hook() -> None:
        # Calling the hook inside the task keeps a synchronous failure before
        # its awaitable exists from skipping the caller's drain.
        await cancel_hook()

    await _await_ignoring_cancellation(asyncio.ensure_future(invoke_cancel_hook()))


async def _await_ignoring_cancellation(future: asyncio.Future[Any]) -> None:
    """Wait until ``future`` settles, absorbing its failure and repeated cancellation."""
    while not future.done():
        with suppress(asyncio.CancelledError, Exception):
            await asyncio.shield(future)
    with suppress(asyncio.CancelledError, Exception):
        future.result()


@dataclass(frozen=True)
class PrefetchedItem(Generic[_Item, _Prepared]):
    """One prepared item; ``directory`` exists until the consumer advances or closes."""

    item: _Item
    value: _Prepared
    directory: Path


@dataclass(frozen=True)
class _PendingPreparation(Generic[_Item, _Prepared]):
    item: _Item
    directory: Path
    task: asyncio.Task[_Prepared]


class PrefetchedItems(Generic[_Item, _Prepared]):
    """Prepared items in input order, with bounded lookahead; see :func:`prefetch`.

    A class rather than an async generator: awaiting inside a closing async
    generator is unreliable under repeated cancellation once compiled.
    """

    def __init__(
        self,
        items: Iterable[_Item],
        prepare: Callable[[_Item, Path], _Prepared],
        *,
        working_directory: Path,
        lookahead: int,
        cancel_hook: Callable[[], Awaitable[object]] | None,
    ) -> None:
        if isinstance(lookahead, bool) or not isinstance(lookahead, int) or lookahead < 0:
            raise ValueError("lookahead must be a non-negative integer")
        self._remaining_items: Iterator[_Item] = iter(items)
        self._prepare = prepare
        self._working_directory = Path(working_directory)
        self._lookahead = lookahead
        self._cancel_hook = cancel_hook
        self._pending: deque[_PendingPreparation[_Item, _Prepared]] = deque()
        self._current: PrefetchedItem[_Item, _Prepared] | None = None
        self._consumer_waiting = False
        self._release_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> "PrefetchedItems[_Item, _Prepared]":
        return self

    async def __aexit__(self, *exception_info: object) -> None:
        await self.aclose()

    def __aiter__(self) -> "PrefetchedItems[_Item, _Prepared]":
        return self

    async def __anext__(self) -> PrefetchedItem[_Item, _Prepared]:
        if self._release_task is not None:
            raise StopAsyncIteration
        if self._consumer_waiting:
            raise RuntimeError("prefetched items must be consumed one at a time")
        self._consumer_waiting = True
        try:
            if self._current is not None:
                released_directory = self._current.directory
                self._current = None
                await run_blocking(_remove_directory, released_directory)
            if self._release_task is not None:
                raise StopAsyncIteration
            # Start the requested item only if lookahead has not already; this
            # keeps at most max(lookahead, 1) preparations running at once.
            self._start_preparations_until(max(self._lookahead, 1))
            if not self._pending:
                raise StopAsyncIteration
            pending = self._pending[0]
            # Shielded: cancelling the caller must not cancel a preparation
            # whose thread may still write into its directory. aclose drains it.
            try:
                prepared_value = await asyncio.shield(pending.task)
            except BaseException:
                if self._release_task is not None:
                    raise StopAsyncIteration from None
                if pending.task.done() and not pending.task.cancelled():
                    self._pending.popleft()
                    await run_blocking(_remove_directory, pending.directory)
                raise
            if self._release_task is not None:
                raise StopAsyncIteration
            self._pending.popleft()
            self._current = PrefetchedItem(pending.item, prepared_value, pending.directory)
            # Background preparation continues while the caller uses this item.
            self._start_preparations_until(self._lookahead)
            return self._current
        finally:
            self._consumer_waiting = False

    def _start_preparations_until(self, pending_limit: int) -> None:
        if self._release_task is not None:
            return
        while len(self._pending) < pending_limit:
            next_item = next(self._remaining_items, _NO_MORE_ITEMS)
            if next_item is _NO_MORE_ITEMS:
                return
            self._pending.append(self._start_preparation(cast(_Item, next_item)))

    def _start_preparation(self, item: _Item) -> _PendingPreparation[_Item, _Prepared]:
        directory = Path(tempfile.mkdtemp(dir=self._working_directory, prefix="prefetch-"))
        task = asyncio.ensure_future(run_blocking(self._prepare, item, directory))
        return _PendingPreparation(item, directory, task)

    async def aclose(self) -> None:
        """Release every held item; run the cancel hook first if any were held.

        Repeated cancellation cannot interrupt the release. A cancellation
        received while closing is re-raised once every directory is removed.
        """
        if self._release_task is None:
            self._release_task = asyncio.ensure_future(self._release_held_items())
        cancelled_while_closing = False
        while not self._release_task.done():
            try:
                await asyncio.shield(self._release_task)
            except asyncio.CancelledError:
                cancelled_while_closing = True
        self._release_task.result()
        if cancelled_while_closing:
            raise asyncio.CancelledError

    async def _release_held_items(self) -> None:
        held_preparations = tuple(self._pending)
        held_directories = (
            (self._current.directory,) if self._current is not None else ()
        ) + tuple(preparation.directory for preparation in held_preparations)
        self._pending.clear()
        self._current = None
        if not held_directories:
            return
        try:
            # Stop outside readers (such as a model server still reading the
            # current item) before waiting on threads that may depend on them.
            if self._cancel_hook is not None:
                await _run_cancel_hook_to_completion(self._cancel_hook)
            for preparation in held_preparations:
                await _await_ignoring_cancellation(preparation.task)
        finally:
            removal_errors: list[OSError] = []
            for directory in held_directories:
                try:
                    await run_blocking(_remove_directory, directory)
                except OSError as error:
                    removal_errors.append(error)
            if removal_errors:
                raise removal_errors[0]


def prefetch(
    items: Iterable[_Item],
    prepare: Callable[[_Item, Path], _Prepared],
    *,
    working_directory: Path | str,
    lookahead: int = 1,
    cancel_hook: Callable[[], Awaitable[object]] | None = None,
) -> PrefetchedItems[_Item, _Prepared]:
    """Prepare items ahead of their consumer, each in its own scratch directory.

    ``prepare(item, directory)`` is blocking work, such as sampling frames or
    re-encoding a window, and runs through :func:`run_blocking` in a new empty
    directory under ``working_directory``. Items are yielded in input order.
    While the caller uses one item, up to ``lookahead`` later items are prepared
    or kept ready. At most ``max(lookahead, 1)`` preparations run at once and at
    most ``lookahead + 1`` directories exist. ``lookahead=0`` prepares each item
    only when requested.

    An item's directory is removed when the caller advances to the next item or
    closes the iterator. A preparation failure is raised when the caller
    reaches that item. Close the iterator with ``async with``; closing it while
    it holds any item first runs ``cancel_hook`` (to stop a reader such as a
    local model server), then waits for started preparations, then removes the
    directories.

    Use it when a sequential loop alternates heavy local preparation with
    waiting on a model endpoint. ``App.process_many(max_workers=...)`` already
    overlaps work across episodes; this overlaps it within one.
    """
    return PrefetchedItems(
        items,
        prepare,
        working_directory=Path(working_directory),
        lookahead=lookahead,
        cancel_hook=cancel_hook,
    )


def _remove_directory(directory: Path) -> None:
    with suppress(FileNotFoundError):
        shutil.rmtree(directory)


@asynccontextmanager
async def blocking_context(manager: AbstractContextManager[_Result]) -> AsyncIterator[_Result]:
    """Enter and close a blocking resource without releasing it during acquisition."""
    entered = False

    def enter() -> _Result:
        nonlocal entered
        resource = manager.__enter__()
        entered = True
        return resource

    try:
        yield await run_blocking(enter)
    except BaseException as error:
        if entered and await run_blocking(
            manager.__exit__, type(error), error, error.__traceback__
        ):
            return
        raise
    else:
        await run_blocking(manager.__exit__, None, None, None)
