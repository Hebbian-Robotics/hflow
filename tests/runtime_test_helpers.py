"""Shared fakes for the Airflow runtime tests: a clock, a closed port, a .env reader."""

from __future__ import annotations

import socket
from pathlib import Path


class InstantlyAdvancingClock:
    """A ``time`` stand-in whose ``sleep`` advances ``monotonic`` without waiting."""

    def __init__(self) -> None:
        self.current_time_s = 0.0

    def monotonic(self) -> float:
        return self.current_time_s

    def sleep(self, duration_s: float) -> None:
        self.current_time_s += duration_s


def unused_local_port() -> int:
    """A loopback port that had no listener when probed."""
    with socket.socket() as port_probe:
        port_probe.bind(("127.0.0.1", 0))
        return port_probe.getsockname()[1]


def read_env_file_values(env_file: Path) -> dict[str, str]:
    """``KEY=value`` pairs from a rendered ``.env``, skipping blanks and comments."""
    return dict(
        line.split("=", 1)
        for line in env_file.read_text().splitlines()
        if line and not line.startswith("#")
    )
