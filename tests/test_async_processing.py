"""Async checks share the caller loop and keep episode lifetimes safe on cancellation."""

import asyncio
import threading
from collections.abc import Awaitable
from pathlib import Path

import pytest

import hflow
from hflow.asyncio_utils import run_blocking, run_blocking_with_cancel_hook
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode


def test_cancelling_batch_cancels_admitted_checks_without_starting_queued_episodes(
    tmp_path: Path,
) -> None:
    source_paths = tuple(
        synthesize_episode(
            tmp_path / f"source-{index}.mcap",
            SyntheticEpisodeSpec(duration_s=0.1, cameras=(), task=str(index)),
        )
        for index in range(3)
    )
    application = hflow.App("async-cancellation", data_root=tmp_path / "data", default_checks=())

    async def scenario() -> None:
        caller_loop = asyncio.get_running_loop()
        admitted: set[str] = set()
        cancelled: set[str] = set()
        both_admitted = asyncio.Event()

        @application.check(version="1")
        async def remote_measurement(episode: hflow.Episode) -> hflow.CheckResult:
            assert asyncio.get_running_loop() is caller_loop
            source_name = str(episode.metadata["task"])
            admitted.add(source_name)
            if len(admitted) == 2:
                both_admitted.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.add(source_name)
            return hflow.CheckResult()

        async with asyncio.timeout(10):
            batch_task = asyncio.create_task(
                application.process_many(
                    source_paths,
                    max_workers=2,
                    record=False,
                    stages=(hflow.Stage.SYNC, hflow.Stage.META),
                )
            )
            await both_admitted.wait()
            batch_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await batch_task
        assert admitted == cancelled == {"0", "1"}
        assert not tuple(application.workspace.catalog_root.workspace.glob("**/*.parquet"))

    asyncio.run(scenario())


def test_cancellation_drains_blocking_media_before_closing_the_episode(tmp_path: Path) -> None:
    source_path = synthesize_episode(
        tmp_path / "source.mcap", SyntheticEpisodeSpec(duration_s=0.1, cameras=())
    )
    application = hflow.App("blocking-cancellation", data_root=tmp_path / "data", default_checks=())
    release_media = threading.Event()
    media_finished = threading.Event()
    check_finished = threading.Event()

    async def scenario() -> None:
        caller_loop = asyncio.get_running_loop()
        media_started = asyncio.Event()

        def read_media(episode: hflow.Episode) -> hflow.CheckResult:
            caller_loop.call_soon_threadsafe(media_started.set)
            if not release_media.wait(timeout=10):
                raise TimeoutError("test did not release media work")
            # Read the live episode after cancellation was requested, before
            # the pipeline is permitted to release its reader and workspace.
            measurement_count = len(episode.channel("/joint_states"))
            media_finished.set()
            return hflow.CheckResult(measurements={"messages": measurement_count})

        @application.check(version="1")
        async def media_measurement(episode: hflow.Episode) -> hflow.CheckResult:
            try:
                return await run_blocking(read_media, episode)
            finally:
                assert media_finished.is_set()
                check_finished.set()

        async with asyncio.timeout(10):
            processing_task = asyncio.create_task(
                application.process(
                    source_path, record=False, stages=(hflow.Stage.SYNC, hflow.Stage.META)
                )
            )
            try:
                await media_started.wait()
                processing_task.cancel()
                await asyncio.sleep(0)
                assert not processing_task.done()
                assert not check_finished.is_set()
            finally:
                release_media.set()
            with pytest.raises(asyncio.CancelledError):
                await processing_task
        assert media_finished.is_set() and check_finished.is_set()

    asyncio.run(scenario())


def test_loop_shutdown_finishes_media_before_episode_cleanup(tmp_path: Path) -> None:
    source_path = synthesize_episode(
        tmp_path / "source.mcap", SyntheticEpisodeSpec(duration_s=0.1, cameras=())
    )
    application = hflow.App("shutdown", data_root=tmp_path / "data", default_checks=())
    release_media = threading.Event()
    lifecycle: list[str] = []
    release_timer: threading.Timer | None = None

    async def scenario() -> None:
        nonlocal release_timer
        caller_loop = asyncio.get_running_loop()
        media_started = asyncio.Event()

        def read_media(episode: hflow.Episode) -> hflow.CheckResult:
            caller_loop.call_soon_threadsafe(media_started.set)
            if not release_media.wait(timeout=10):
                raise TimeoutError("test did not release media")
            message_count = len(episode.channel("/joint_states"))
            lifecycle.append("media finished")
            return hflow.CheckResult(measurements={"messages": message_count})

        @application.check(version="1")
        async def media_measurement(episode: hflow.Episode) -> hflow.CheckResult:
            try:
                return await run_blocking(read_media, episode)
            finally:
                lifecycle.append("check finished")

        processing_task = asyncio.create_task(
            application.process(
                source_path, record=False, stages=(hflow.Stage.SYNC, hflow.Stage.META)
            )
        )
        await asyncio.wait_for(media_started.wait(), timeout=10)
        assert not processing_task.done()
        release_timer = threading.Timer(0.1, release_media.set)
        release_timer.start()
        # Returning cancels every task during asyncio.run's shutdown. The
        # blocking reader must finish before the check's files are released.

    try:
        asyncio.run(scenario())
    finally:
        release_media.set()
        if release_timer is not None:
            release_timer.join()
    assert lifecycle == ["media finished", "check finished"]


def test_cancel_hook_stops_the_external_reader_before_draining_blocked_work() -> None:
    # The blocking work waits on an external reader (think: a model server
    # holding its input files). Only the hook can release it, so the drain must
    # not start waiting until the hook has run.
    reader_stopped = threading.Event()
    lifecycle: list[str] = []

    def wait_for_reader_shutdown() -> str:
        if not reader_stopped.wait(timeout=10):
            raise TimeoutError("cancel hook never stopped the reader")
        lifecycle.append("blocking work finished")
        return "unused"

    async def stop_reader() -> None:
        lifecycle.append("hook ran")
        reader_stopped.set()

    async def scenario() -> None:
        work_task = asyncio.create_task(
            run_blocking_with_cancel_hook(stop_reader, wait_for_reader_shutdown)
        )
        await asyncio.sleep(0.05)
        work_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(work_task, timeout=10)
        lifecycle.append("cancellation propagated")

    asyncio.run(scenario())
    assert lifecycle == ["hook ran", "blocking work finished", "cancellation propagated"]


def test_failing_and_recancelled_hook_still_drains_and_propagates_cancellation() -> None:
    release_work = threading.Event()
    work_finished = threading.Event()

    def blocked_work() -> None:
        if not release_work.wait(timeout=10):
            raise TimeoutError("test did not release blocking work")
        work_finished.set()

    async def scenario() -> None:
        hook_started = asyncio.Event()
        hook_may_fail = asyncio.Event()

        async def failing_hook() -> None:
            hook_started.set()
            await hook_may_fail.wait()
            release_work.set()
            raise RuntimeError("reader shutdown failed")

        work_task = asyncio.create_task(run_blocking_with_cancel_hook(failing_hook, blocked_work))
        await asyncio.sleep(0.05)
        work_task.cancel()
        await asyncio.wait_for(hook_started.wait(), timeout=10)
        # A second cancellation must neither interrupt the hook nor skip the drain.
        work_task.cancel()
        await asyncio.sleep(0)
        assert not work_task.done()
        hook_may_fail.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(work_task, timeout=10)
        assert work_finished.is_set()

    try:
        asyncio.run(scenario())
    finally:
        release_work.set()


def test_hook_failing_before_returning_an_awaitable_still_drains() -> None:
    release_work = threading.Event()
    work_finished = threading.Event()

    def blocked_work() -> None:
        if not release_work.wait(timeout=10):
            raise TimeoutError("test did not release blocking work")
        work_finished.set()

    def hook_failing_during_setup() -> Awaitable[None]:
        release_work.set()
        raise RuntimeError("hook setup failed before returning an awaitable")

    async def scenario() -> None:
        work_task = asyncio.create_task(
            run_blocking_with_cancel_hook(hook_failing_during_setup, blocked_work)
        )
        await asyncio.sleep(0.05)
        work_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(work_task, timeout=10)
        assert work_finished.is_set()

    try:
        asyncio.run(scenario())
    finally:
        release_work.set()
