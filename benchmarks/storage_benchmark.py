"""Benchmark (issue #26): per-frame JPEG storage vs canonical MCAP by GOP preset.

Measures the per-frame-JPEG vs canonical-MCAP storage effect at honest
small scale. The sum of the recording's JPEG payload bytes is the
per-frame-JPEG baseline (the storage floor of an H5-of-JPEGs layout),
compared against the canonical MCAP file size under each GOP preset.

Run: ``uv run python benchmarks/storage_benchmark.py`` on the synthetic
fixture (``--quick`` for a shorter one), or ``--input <recording.mcap>`` to
measure a real recording (see ``benchmarks/_real_input.py`` for the reference
nuScenes sample and its provenance). Deterministic content; results print as
a markdown table.
"""

import argparse
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from _real_input import discover_topics

from hflow.episode import Episode
from hflow.format import GopPreset
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode
from hflow.transform import TransformConfig, write_canonical_episode

FULL_DURATION_S = 30.0
QUICK_DURATION_S = 8.0


@dataclass(frozen=True)
class StorageRow:
    label: str
    file_bytes: int
    video_payload_bytes: int
    # Video PAYLOAD vs the JPEG payload baseline isolates the codec effect.
    # File sizes are shown alongside
    # but are not the reduction metric: a real recording can carry hundreds of
    # MB of non-camera channels (lidar/radar/diagnostics) that pass through
    # into the canonical file and would swamp a file-level comparison.
    video_reduction_vs_jpeg_pct: float | None  # None for the baseline row itself
    transform_seconds: float | None = None  # None for non-transform rows


def _sum_camera_payload_bytes(path: Path) -> int:
    """Sum of decoded image/video payload bytes across all camera topics."""
    total_bytes = 0
    with Episode(path) as episode:
        for camera_topic in episode.cameras:
            channel = episode.channel(camera_topic)
            total_bytes += sum(len(bytes(message.data)) for message in channel.messages)
    return total_bytes


def _format_size(size_bytes: int) -> str:
    return f"{size_bytes / 1_000_000:.2f} MB"


def _measure_source(source_path: Path, working_dir: Path) -> list[StorageRow]:
    jpeg_payload_bytes = _sum_camera_payload_bytes(source_path)
    rows = [
        StorageRow(
            label="per-frame JPEG payloads (baseline)",
            file_bytes=jpeg_payload_bytes,
            video_payload_bytes=jpeg_payload_bytes,
            video_reduction_vs_jpeg_pct=None,
        ),
        StorageRow(
            label="source MCAP (JPEG in-band, as recorded)",
            file_bytes=source_path.stat().st_size,
            video_payload_bytes=jpeg_payload_bytes,
            video_reduction_vs_jpeg_pct=None,
        ),
    ]
    for preset in (GopPreset.VLA, GopPreset.WORLD_MODEL):
        canonical_path = working_dir / f"canonical-{preset}.mcap"
        started = time.perf_counter()
        write_canonical_episode(source_path, canonical_path, TransformConfig(gop_preset=preset))
        transform_seconds = time.perf_counter() - started
        video_payload_bytes = _sum_camera_payload_bytes(canonical_path)
        rows.append(
            StorageRow(
                label=f"canonical MCAP, gop_preset={preset} (zstd chunks)",
                file_bytes=canonical_path.stat().st_size,
                video_payload_bytes=video_payload_bytes,
                video_reduction_vs_jpeg_pct=100.0
                * (1.0 - video_payload_bytes / jpeg_payload_bytes),
                transform_seconds=transform_seconds,
            )
        )
    return rows


def run_storage_benchmark(duration_s: float) -> list[StorageRow]:
    # Defect injections off: black frames compress to almost nothing and
    # would flatter every codec; the comparison wants steady content.
    spec = SyntheticEpisodeSpec(
        duration_s=duration_s,
        black_segment=None,
        joint_jump_at_s=None,
        timestamp_offset_segment=None,
    )
    working_dir = Path(tempfile.mkdtemp(prefix="storage-benchmark-"))
    try:
        source_path = synthesize_episode(working_dir / "source.mcap", spec)
        return _measure_source(source_path, working_dir)
    finally:
        shutil.rmtree(working_dir, ignore_errors=True)


def run_storage_benchmark_on_input(input_path: Path) -> list[StorageRow]:
    working_dir = Path(tempfile.mkdtemp(prefix="storage-benchmark-"))
    try:
        return _measure_source(input_path, working_dir)
    finally:
        shutil.rmtree(working_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="short fixture, fast run")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="measure a real recording instead of the synthetic fixture",
    )
    arguments = parser.parse_args()

    if arguments.input is not None:
        topics = discover_topics(arguments.input)
        rows = run_storage_benchmark_on_input(arguments.input)
        print(
            f"storage benchmark: {arguments.input.name}, {topics.duration_s:.1f}s real "
            f"recording, {len(topics.camera_topics)} cameras\n"
        )
    else:
        duration_s = QUICK_DURATION_S if arguments.quick else FULL_DURATION_S
        rows = run_storage_benchmark(duration_s)
        print(f"storage benchmark: {duration_s:g}s synthetic episode, 2 cameras @ 15 Hz, 320x240\n")
    print("| layout | file size | video payload | video reduction vs JPEG | transform time |")
    print("|---|---|---|---|---|")
    for row in rows:
        reduction = (
            f"{row.video_reduction_vs_jpeg_pct:.1f}%"
            if row.video_reduction_vs_jpeg_pct is not None
            else "-"
        )
        transform_time = f"{row.transform_seconds:.1f} s" if row.transform_seconds else "-"
        print(
            f"| {row.label} | {_format_size(row.file_bytes)} "
            f"| {_format_size(row.video_payload_bytes)} | {reduction} | {transform_time} |"
        )
    print(
        "\nreduction compares video PAYLOAD bytes (the codec effect); file sizes also"
        "\ncarry every non-camera channel passed through byte-for-byte."
        "\nDyna's at-scale measurement for comparison: ~68% reduction (real footage, tuned GOPs)."
    )


if __name__ == "__main__":
    main()
