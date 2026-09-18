"""Explicit blocking work inside async pipelines, with lifetime-safe cancellation."""

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractContextManager, asynccontextmanager, suppress
from contextvars import copy_context
from functools import partial
from typing import ParamSpec, TypeVar

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
    # An executor Future is not part of asyncio.run's cancel-all-Tasks sweep.
    # A to_thread Task can become cancelled while its underlying thread still
    # owns episode files, falsely appearing drained during loop shutdown.
    operation_future = asyncio.get_running_loop().run_in_executor(
        None, copy_context().run, partial(operation, *arguments, **keyword_arguments)
    )
    try:
        return await asyncio.shield(operation_future)
    except asyncio.CancelledError:
        while not operation_future.done():
            with suppress(asyncio.CancelledError, Exception):
                await asyncio.shield(operation_future)
        with suppress(asyncio.CancelledError, Exception):
            operation_future.result()
        raise


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
