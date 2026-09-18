"""Async checks share the caller loop and keep episode lifetimes safe on cancellation."""

import asyncio
import threading
from pathlib import Path

import pytest

import hflow
from hflow.asyncio_utils import run_blocking
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
