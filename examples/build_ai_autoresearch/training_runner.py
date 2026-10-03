"""Snapshot editable code and enforce startup/training deadlines out of process."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

from examples.build_ai_autoresearch.contracts import (
    TrialBudget,
    WorkerStarted,
    file_sha256,
    read_record,
)


def snapshot_training_source(source: Path, trial_directory: Path) -> str:
    if source.is_symlink() or not source.is_file():
        raise ValueError("editable train.py must be a regular file")
    encoded = source.read_bytes()
    compile(encoded, str(source), "exec")
    with (trial_directory / "train.py").open("xb") as snapshot:
        snapshot.write(encoded)
    return file_sha256(trial_directory / "train.py")


def run_worker(command: Sequence[str], trial_directory: Path, budget: TrialBudget) -> None:
    with (trial_directory / "training.log").open("xb") as log:
        process_started = time.monotonic()
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        startup_deadline = time.monotonic() + budget.startup_seconds
        try:
            started_path = trial_directory / "started.json"
            while not started_path.exists():
                if process.poll() is not None:
                    raise RuntimeError(f"training worker exited before startup; see {log.name}")
                if time.monotonic() >= startup_deadline:
                    raise TimeoutError("training worker exceeded its fixed startup allowance")
                time.sleep(0.05)
            started = read_record(started_path, WorkerStarted)
            observed_time = time.monotonic()
            if not process_started <= started.started_monotonic <= observed_time:
                raise ValueError("worker startup receipt has an invalid monotonic timestamp")
            remaining = started.started_monotonic + budget.training_seconds - observed_time
            if remaining <= 0:
                raise TimeoutError("training worker exceeded its frozen compute budget")
            try:
                return_code = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired as error:
                raise TimeoutError("training worker exceeded its frozen compute budget") from error
            if return_code != 0:
                raise RuntimeError(
                    f"training worker failed with exit code {return_code}; see {log.name}"
                )
        finally:
            # Kill the process group even after a worker exits, so spawned children cannot outlive a trial.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
