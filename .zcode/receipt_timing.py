"""Timing: the receipt cost — content_episode_id + stat per episode.

Measures the hashing added by #379 on a real canonical episode file,
at fixture scale and extrapolated at delivery scale.

Run: uv run --locked --all-extras python .zcode/receipt_timing.py <episode.mcap>
"""

import statistics
import sys
import time
from pathlib import Path

from hflow.catalog import content_episode_id


def main() -> None:
    path = Path(sys.argv[1])
    size = __import__("pathlib").Path(path).stat().st_size
    timings = []
    for _ in range(20):
        started = time.perf_counter()
        content_episode_id(path)
        timings.append(time.perf_counter() - started)
    median_us = statistics.median(timings) * 1_000_000
    print(f"episode file: {path} ({size} bytes)")
    print(f"content_episode_id median of 20 runs: {median_us:.1f} us")
    # sha256 throughput ~500 MB/s in CPython: a 100 MB episode costs ~0.2 s,
    # one linear read of a file that is already on local disk pre-publish.
    projected_100mb_s = 100 / 500
    print(f"projected at 100 MB episode (~500 MB/s sha256): ~{projected_100mb_s:.2f} s")


if __name__ == "__main__":
    main()
