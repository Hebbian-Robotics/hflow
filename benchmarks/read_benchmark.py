"""Benchmark (issue #27): chunk layout vs read cost, three layouts, two patterns.

Measures the chunk-layout effect at honest small scale, alongside the
results Dyna's article reports (its Figure 3, panels 2-3). Default MCAP
writing gives each topic its own chunks, so one training sample costs a
read per topic; grouping topics by read pattern (cameras time-major in one
chunk stream, proprioception+actions in another) makes samples cost one
read per *group* -- Dyna's article reports ~3.4x fewer fetches and ~2.9x
faster reads at their scale on object storage.

Layouts compared (identical message content):

- **per-topic**: every topic its own chunk stream (Dyna's baseline).
- **interleaved**: the stock Python writer's single chunk builder (all
  topics share every chunk) -- included because it is what most tooling
  writes today.
- **topic-group** (ours): cameras in one chunk stream, state in another.

Access patterns measured for each layout:

- **training sample**: a random 1-second window reading ALL cameras plus
  ``/joint_states`` (a multi-view sample).
- **state-only scan**: the whole ``/joint_states`` stream and nothing else
  (a curation/QC pass).

Metrics: fetches (distinct chunks touched -- on object storage each is a
round trip), compressed bytes fetched (what those chunks cost to pull), and
local wall-clock read time with the same stock reader. Local disk turns a
"fetch" into a decompression instead of a network round trip, so wall-clock
here understates the benefit that matters on S3; the fetch and byte counts
are layout facts and transfer directly.

Run: ``uv run python benchmarks/read_benchmark.py`` (``--quick`` for fewer
samples on a shorter fixture), or ``--input <recording.mcap>`` to measure a
real recording (see ``benchmarks/_real_input.py`` for the reference nuScenes
sample). Sampling is seeded; timings are wall-clock.
"""

import argparse
import random
import shutil
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from _real_input import discover_topics, plan_topic_groups
from mcap.reader import make_reader
from mcap.summary import Summary
from mcap.writer import Writer as StockWriter

from hflow.format import DEFAULT_CHUNK_SIZE_BYTES, GopPreset
from hflow.mcap_writer import CanonicalMcapWriter
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode
from hflow.transform import TransformConfig, write_canonical_episode

FULL_DURATION_S = 60.0
QUICK_DURATION_S = 15.0
FULL_SAMPLE_COUNT = 200
QUICK_SAMPLE_COUNT = 60
SAMPLE_WINDOW_S = 1.0
CAMERA_NAMES = ("wrist_cam", "overhead_cam", "left_cam", "right_cam")
RANDOM_SEED = 20260818


@dataclass(frozen=True)
class SampleWindow:
    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class PatternStats:
    """One (layout, access pattern) measurement."""

    layout: str
    fetches_per_sample: float
    compressed_bytes_per_sample: float
    read_seconds: float
    message_bytes_read: int


def _rewrite(
    source: Path,
    output: Path,
    library_label: str,
    group_for_topic: Callable[[str], str] | None,
    chunk_size_bytes: int = DEFAULT_CHUNK_SIZE_BYTES,
) -> None:
    """Rewrite every message of ``source`` into ``output`` in a new layout.

    ``group_for_topic=None`` uses the stock writer (single interleaved chunk
    builder); otherwise the canonical writer with the given group assignment.
    """
    with source.open("rb") as in_stream, output.open("wb") as out_stream:
        reader = make_reader(in_stream)
        schema_id_map: dict[int, int] = {}
        channel_id_map: dict[int, int] = {}
        if group_for_topic is None:
            stock_writer = StockWriter(out_stream, chunk_size=chunk_size_bytes)
            stock_writer.start(profile="", library=library_label)
            for schema, channel, message in reader.iter_messages(log_time_order=True):
                if schema is not None and schema.id not in schema_id_map:
                    schema_id_map[schema.id] = stock_writer.register_schema(
                        schema.name, schema.encoding, schema.data
                    )
                if channel.id not in channel_id_map:
                    channel_id_map[channel.id] = stock_writer.register_channel(
                        topic=channel.topic,
                        message_encoding=channel.message_encoding,
                        schema_id=schema_id_map.get(channel.schema_id, 0),
                    )
                stock_writer.add_message(
                    channel_id_map[channel.id],
                    log_time=message.log_time,
                    data=message.data,
                    publish_time=message.publish_time,
                    sequence=message.sequence,
                )
            stock_writer.finish()
            return
        with CanonicalMcapWriter(
            out_stream, chunk_size=chunk_size_bytes, library=library_label
        ) as grouped_writer:
            for schema, channel, message in reader.iter_messages(log_time_order=True):
                if schema is not None and schema.id not in schema_id_map:
                    schema_id_map[schema.id] = grouped_writer.register_schema(
                        schema.name, schema.encoding, schema.data
                    )
                if channel.id not in channel_id_map:
                    channel_id_map[channel.id] = grouped_writer.register_channel(
                        channel.topic,
                        message_encoding=channel.message_encoding,
                        schema_id=schema_id_map.get(channel.schema_id, 0),
                        group=group_for_topic(channel.topic),
                    )
                grouped_writer.write_message(
                    channel_id_map[channel.id],
                    log_time=message.log_time,
                    data=message.data,
                    publish_time=message.publish_time,
                    sequence=message.sequence,
                )


def _read_summary(stream: IO[bytes], path: Path) -> Summary:
    summary = make_reader(stream).get_summary()
    if summary is None:
        raise ValueError(f"{path} has no summary section")
    return summary


def _chunk_cost(
    summary: Summary,
    topics: Sequence[str],
    start_ns: int,
    end_ns: int,
) -> tuple[int, int]:
    """(distinct chunks touched, their compressed bytes) for one logical read."""
    channel_ids_by_topic = {channel.topic: channel.id for channel in summary.channels.values()}
    needed_channel_ids = {channel_ids_by_topic[topic] for topic in topics}
    fetched_chunks = [
        chunk_index
        for chunk_index in summary.chunk_indexes
        if chunk_index.message_start_time <= end_ns
        and chunk_index.message_end_time >= start_ns
        and any(
            channel_id in chunk_index.message_index_offsets for channel_id in needed_channel_ids
        )
    ]
    return len(fetched_chunks), sum(chunk.compressed_size for chunk in fetched_chunks)


def _timed_reads(path: Path, reads: Sequence[tuple[Sequence[str], int, int]]) -> tuple[float, int]:
    """Wall-clock seconds and message bytes for a sequence of (topics, t0, t1) reads."""
    with path.open("rb") as stream:
        reader = make_reader(stream)
        message_bytes = 0
        started = time.perf_counter()
        for topics, start_ns, end_ns in reads:
            for _schema, _channel, message in reader.iter_messages(
                topics=list(topics), start_time=start_ns, end_time=end_ns
            ):
                message_bytes += len(message.data)
        elapsed = time.perf_counter() - started
    return elapsed, message_bytes


def _measure_pattern(
    path: Path,
    layout: str,
    reads: Sequence[tuple[Sequence[str], int, int]],
) -> PatternStats:
    with path.open("rb") as stream:
        summary = _read_summary(stream, path)
    total_fetches = 0
    total_compressed_bytes = 0
    for topics, start_ns, end_ns in reads:
        fetches, compressed_bytes = _chunk_cost(summary, topics, start_ns, end_ns)
        total_fetches += fetches
        total_compressed_bytes += compressed_bytes
    _timed_reads(path, reads)  # warm the OS page cache
    read_seconds, message_bytes = _timed_reads(path, reads)
    return PatternStats(
        layout=layout,
        fetches_per_sample=total_fetches / len(reads),
        compressed_bytes_per_sample=total_compressed_bytes / len(reads),
        read_seconds=read_seconds,
        message_bytes_read=message_bytes,
    )


def _print_pattern_table(title: str, stats_rows: list[PatternStats]) -> None:
    print(f"\n{title}")
    print("| layout | fetches/sample | compressed MB fetched/sample | wall-clock |")
    print("|---|---|---|---|")
    for stats in stats_rows:
        print(
            f"| {stats.layout} | {stats.fetches_per_sample:.2f} "
            f"| {stats.compressed_bytes_per_sample / 1_000_000:.3f} "
            f"| {stats.read_seconds * 1000:.0f} ms |"
        )
    baseline = stats_rows[0]
    ours = stats_rows[-1]
    if ours.fetches_per_sample and ours.read_seconds:
        print(
            f"topic-group vs {baseline.layout}: "
            f"{baseline.fetches_per_sample / ours.fetches_per_sample:.2f}x fewer fetches, "
            f"{baseline.compressed_bytes_per_sample / ours.compressed_bytes_per_sample:.2f}x "
            f"fewer bytes, {baseline.read_seconds / ours.read_seconds:.2f}x faster locally"
        )


DERIVED_CHUNK_SIZE_ARGUMENT = "derived"


def _parse_chunk_size_argument(raw_value: str) -> int | None:
    """``--chunk-size-bytes``: a flat integer target, or the derived default.

    ``None`` is what :class:`hflow.TransformConfig` reads as "size each group
    from its own byte rate", so the benchmark can measure the layout the
    library actually ships rather than an approximation of it.
    """
    if raw_value == DERIVED_CHUNK_SIZE_ARGUMENT:
        return None
    return int(raw_value)


def _describe_chunk_size(chunk_size_bytes: int | None) -> str:
    if chunk_size_bytes is None:
        return f"{DERIVED_CHUNK_SIZE_ARGUMENT} per group (baselines at {DEFAULT_CHUNK_SIZE_BYTES})"
    return str(chunk_size_bytes)


def _measure_source(
    source_path: Path,
    working_dir: Path,
    sample_count: int,
    header: str,
    topic_groups: dict[str, str] | None = None,
    chunk_size_bytes: int | None = DEFAULT_CHUNK_SIZE_BYTES,
) -> None:
    """The full three-layout comparison over one source recording."""
    grouped_path = working_dir / "topic-group.mcap"
    transform_started = time.perf_counter()
    write_canonical_episode(
        source_path,
        grouped_path,
        TransformConfig(
            gop_preset=GopPreset.VLA,
            topic_groups=topic_groups or {},
            chunk_size_bytes=chunk_size_bytes,
        ),
    )
    transform_seconds = time.perf_counter() - transform_started
    # The baselines have no per-group concept to derive from: per-topic groups
    # are single topics, and the stock writer has exactly one chunk buffer. So
    # a derived run measures them at the flat default, and the header says so.
    baseline_chunk_size = (
        chunk_size_bytes if chunk_size_bytes is not None else DEFAULT_CHUNK_SIZE_BYTES
    )
    per_topic_path = working_dir / "per-topic.mcap"
    _rewrite(
        grouped_path,
        per_topic_path,
        "per-topic rewrite",
        group_for_topic=lambda t: t,
        chunk_size_bytes=baseline_chunk_size,
    )
    interleaved_path = working_dir / "interleaved.mcap"
    _rewrite(
        grouped_path,
        interleaved_path,
        "interleaved rewrite",
        group_for_topic=None,
        chunk_size_bytes=baseline_chunk_size,
    )

    # Camera/state discovery runs on the CANONICAL file so it works for both
    # the synthetic fixture and arbitrary real recordings (the training
    # sample reads all cameras + the state topic, never every topic).
    topics = discover_topics(grouped_path)
    sample_topics = [*topics.camera_topics, topics.state_topic]

    with grouped_path.open("rb") as stream:
        summary = _read_summary(stream, grouped_path)
    statistics = summary.statistics
    if statistics is None:
        raise ValueError("no statistics record")

    window_ns = int(SAMPLE_WINDOW_S * 1e9)
    generator = random.Random(RANDOM_SEED)
    windows = [
        SampleWindow(start_ns=start, end_ns=start + window_ns)
        for start in (
            generator.randint(
                statistics.message_start_time, statistics.message_end_time - window_ns
            )
            for _ in range(sample_count)
        )
    ]
    training_reads: list[tuple[Sequence[str], int, int]] = [
        (sample_topics, window.start_ns, window.end_ns) for window in windows
    ]
    state_scan_reads: list[tuple[Sequence[str], int, int]] = [
        ([topics.state_topic], statistics.message_start_time, statistics.message_end_time + 1)
    ] * 10

    layouts = [
        ("per-topic (Dyna's baseline)", per_topic_path),
        ("interleaved (stock python writer)", interleaved_path),
        ("topic-group (ours)", grouped_path),
    ]
    print(
        f"read benchmark: {header}, {len(topics.camera_topics)} cameras + {topics.state_topic}, "
        f"chunk_size={_describe_chunk_size(chunk_size_bytes)}, local disk "
        f"(transform to canonical: {transform_seconds:.1f} s)"
    )
    if topic_groups:
        group_members: dict[str, int] = {}
        for group_name in topic_groups.values():
            group_members[group_name] = group_members.get(group_name, 0) + 1
        assignments = ", ".join(
            f"{name}: {count} topics" for name, count in sorted(group_members.items())
        )
        print(f"  topic-group assignment (read-pattern): cameras + {assignments}")
    for label, path in layouts:
        print(f"  {label}: {path.stat().st_size / 1_000_000:.2f} MB")

    training_stats = [_measure_pattern(path, label, training_reads) for label, path in layouts]
    expected_bytes = {stats.message_bytes_read for stats in training_stats}
    if len(expected_bytes) != 1:
        raise AssertionError(f"layouts read different content: {expected_bytes}")
    _print_pattern_table(
        f"training samples ({sample_count} seeded 1s windows, all cameras + state):",
        training_stats,
    )

    state_stats = [_measure_pattern(path, label, state_scan_reads) for label, path in layouts]
    _print_pattern_table(f"state-only scan (full {topics.state_topic} stream, x10):", state_stats)
    print("\nDyna's at-scale numbers for comparison: ~3.4x fewer fetches, ~2.9x faster reads.")


def run_read_benchmark(duration_s: float, sample_count: int) -> None:
    spec = SyntheticEpisodeSpec(
        duration_s=duration_s,
        cameras=CAMERA_NAMES,
        black_segment=None,
        joint_jump_at_s=None,
        timestamp_offset_segment=None,
    )
    working_dir = Path(tempfile.mkdtemp(prefix="read-benchmark-"))
    try:
        source_path = synthesize_episode(working_dir / "source.mcap", spec)
        _measure_source(
            source_path,
            working_dir,
            sample_count,
            header=f"{duration_s:g}s synthetic episode",
        )
    finally:
        shutil.rmtree(working_dir, ignore_errors=True)


def run_read_benchmark_on_input(
    input_path: Path,
    sample_count: int,
    grouping: str,
    chunk_size_bytes: int | None,
) -> None:
    source_topics = discover_topics(input_path)
    topic_groups = (
        plan_topic_groups(input_path, source_topics.camera_topics)
        if grouping == "read-pattern"
        else None
    )
    working_dir = Path(tempfile.mkdtemp(prefix="read-benchmark-"))
    try:
        _measure_source(
            input_path,
            working_dir,
            sample_count,
            header=(
                f"{input_path.name} ({source_topics.duration_s:.1f}s real recording, "
                f"grouping={grouping})"
            ),
            topic_groups=topic_groups,
            chunk_size_bytes=chunk_size_bytes,
        )
    finally:
        shutil.rmtree(working_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="short fixture, fewer samples")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="measure a real recording instead of the synthetic fixture",
    )
    parser.add_argument(
        "--grouping",
        choices=("schema-default", "read-pattern"),
        default="read-pattern",
        help=(
            "with --input: group assignment for the topic-group layout. "
            "schema-default = cameras vs everything-else (the config default, "
            "wrong for sensor-heavy recordings); read-pattern = bulk modalities "
            "in their own group via topic_groups overrides"
        ),
    )
    parser.add_argument(
        "--chunk-size-bytes",
        type=_parse_chunk_size_argument,
        default=DEFAULT_CHUNK_SIZE_BYTES,
        help=(
            "uncompressed chunk-size target: an integer pins one flat target on "
            "every group, `derived` runs the SHIPPED DEFAULT, which sizes each "
            "group from its own byte rate. A flat number equal to the derived "
            "camera target is not the same layout -- it also moves every state "
            "group off the floor -- so a published `derived` row has to be "
            "measured this way"
        ),
    )
    arguments = parser.parse_args()
    sample_count = QUICK_SAMPLE_COUNT if arguments.quick else FULL_SAMPLE_COUNT
    if arguments.input is not None:
        run_read_benchmark_on_input(
            arguments.input, sample_count, arguments.grouping, arguments.chunk_size_bytes
        )
        return
    duration_s = QUICK_DURATION_S if arguments.quick else FULL_DURATION_S
    run_read_benchmark(duration_s, sample_count)


if __name__ == "__main__":
    main()
