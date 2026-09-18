"""Bounded local media-tool execution without diagnostic disclosure."""

import os
import selectors
import subprocess
import time
from dataclasses import dataclass, field


class MediaToolError(RuntimeError):
    """An operational media-tool failure; safe to expose without raw diagnostics."""


@dataclass(frozen=True)
class MediaCommandResult:
    returncode: int
    stdout: bytes
    stderr: bytes = field(default=b"", repr=False)


def run_media_command(
    arguments: list[str], *, timeout_seconds: float, maximum_output_bytes: int = 65536
) -> MediaCommandResult:
    """Drain bounded output, retain private diagnostics, and reap the process on every exit."""
    deadline = time.monotonic() + timeout_seconds
    try:
        with subprocess.Popen(
            arguments, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ) as process:
            try:
                assert process.stdout is not None
                assert process.stderr is not None
                output = bytearray()
                diagnostics = bytearray()
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ, output)
                    selector.register(process.stderr, selectors.EVENT_READ, diagnostics)
                    while selector.get_map():
                        remaining_seconds = deadline - time.monotonic()
                        if remaining_seconds <= 0:
                            raise MediaToolError("media command timed out")
                        for selector_key, _events in selector.select(remaining_seconds):
                            chunk = os.read(selector_key.fd, 65536)
                            if not chunk:
                                selector.unregister(selector_key.fd)
                            elif len(output) + len(diagnostics) + len(chunk) > maximum_output_bytes:
                                raise MediaToolError("media command output exceeded its limit")
                            else:
                                selector_key.data.extend(chunk)
                returncode = process.wait(timeout=max(0.001, deadline - time.monotonic()))
                if time.monotonic() > deadline:
                    raise MediaToolError("media command timed out")
                if returncode < 0:
                    raise MediaToolError("media command was terminated")
                return MediaCommandResult(returncode, bytes(output), bytes(diagnostics))
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()
    except (OSError, subprocess.TimeoutExpired):
        raise MediaToolError("media command failed or timed out") from None


def media_input_was_rejected(result: MediaCommandResult) -> bool:
    """Recognize input decode failures; ambiguous command failures stay operational.

    Diagnostics stay private and bounded. Output/resource failures take precedence
    even if a command also reported a damaged input packet.
    """
    if result.returncode == 0:
        return False
    diagnostics = result.stderr.lower()
    operational_markers = (
        b"permission denied",
        b"no space left",
        b"cannot allocate memory",
        b"out of memory",
        b"unknown encoder",
        b"encoder not found",
        b"unrecognized option",
        b"option not found",
        b"error opening output",
        b"error initializing output",
        b"error writing",
        b"i/o error",
    )
    media_markers = (
        b"moov atom not found",
        b"invalid data found when processing input",
        b"end of file",
        b"error while decoding",
        b"error submitting packet to decoder",
        b"corrupt input packet",
        b"could not find codec parameters",
        b"invalid nal unit",
        b"error parsing nal unit",
    )
    if not any(marker in diagnostics for marker in operational_markers) and any(
        marker in diagnostics for marker in media_markers
    ):
        return True
    raise MediaToolError("media command failed")
