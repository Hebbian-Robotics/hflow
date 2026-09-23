"""Explicit blocking work inside async pipelines, with lifetime-safe cancellation."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractContextManager, asynccontextmanager, suppress
from contextvars import copy_context
from functools import partial
from typing import Any, ParamSpec, TypeVar

_Parameters = ParamSpec("_Parameters")
_Result = TypeVar("_Result")


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

            async def run_cancel_hook() -> None:
                # Calling the hook inside the task keeps a synchronous failure
                # before its awaitable exists from skipping the drain below.
                await cancel_hook()

            await _await_ignoring_cancellation(asyncio.ensure_future(run_cancel_hook()))
        await _await_ignoring_cancellation(operation_future)
        raise


async def _await_ignoring_cancellation(future: asyncio.Future[Any]) -> None:
    """Wait until ``future`` settles, absorbing its failure and repeated cancellation."""
    while not future.done():
        with suppress(asyncio.CancelledError, Exception):
            await asyncio.shield(future)
    with suppress(asyncio.CancelledError, Exception):
        future.result()


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
