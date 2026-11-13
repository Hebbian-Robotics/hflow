"""Bounded subprocess execution for the standalone video-measurements package."""

import os
import selectors
import subprocess
import time
from dataclasses import dataclass


class BoundedProcessError(RuntimeError):
    """A subprocess failed, timed out, or exceeded its output limit."""


@dataclass(frozen=True)
class BoundedProcessResult:
    returncode: int
    stdout: bytes


def run_bounded_process(
    arguments: list[str], *, timeout_seconds: float, maximum_output_bytes: int
) -> BoundedProcessResult:
    """Run a command with bounded runtime and captured stdout, reaping it on every exit."""
    deadline = time.monotonic() + timeout_seconds
    try:
        with subprocess.Popen(
            arguments,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ) as process:
            try:
                assert process.stdout is not None
                output = bytearray()
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    while selector.get_map():
                        remaining_seconds = deadline - time.monotonic()
                        if remaining_seconds <= 0:
                            raise BoundedProcessError("media command timed out")
                        for selector_key, _events in selector.select(remaining_seconds):
                            chunk = os.read(selector_key.fd, 65536)
                            if not chunk:
                                selector.unregister(selector_key.fd)
                            elif len(output) + len(chunk) > maximum_output_bytes:
                                raise BoundedProcessError("media command output exceeded its limit")
                            else:
                                output.extend(chunk)
                returncode = process.wait(timeout=max(0.001, deadline - time.monotonic()))
                if time.monotonic() > deadline:
                    raise BoundedProcessError("media command timed out")
                return BoundedProcessResult(returncode, bytes(output))
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()
    except (OSError, subprocess.TimeoutExpired):
        raise BoundedProcessError("media command failed or timed out") from None
