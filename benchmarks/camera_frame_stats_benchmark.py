"""Benchmark cold ``camera_frame_stats`` throughput on canonical 1080p30 video.

The default fixture is one deterministic 30-second, one-camera episode. Its
JPEG-to-H.264 transform is timed separately, then every check repetition gets
a fresh ``Episode`` workdir so neither the remuxed MP4 nor FFmpeg instrument
cache can turn a cold measurement into a cache hit.

Run: ``uv run python benchmarks/camera_frame_stats_benchmark.py``
(``--quick`` for a small development fixture). Timings are wall-clock.
"""

import argparse
import os
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import hflow
from hflow._video_measurements import FrameStatisticsSettings
from hflow._video_measurements._frame_statistics import frame_statistics_filter_graph
from hflow.checks import camera_frame_stats
from hflow.ffmpeg import ffmpeg_path, ffmpeg_version
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode
from hflow.transform import TransformConfig, write_canonical_episode

FULL_DURATION_S = 30.0
FULL_FRAMES_PER_SECOND = 30.0
FULL_WIDTH = 1920
FULL_HEIGHT = 1080
FULL_REPETITIONS = 3

QUICK_DURATION_S = 3.0
QUICK_FRAMES_PER_SECOND = 10.0
QUICK_WIDTH = 320
QUICK_HEIGHT = 180
QUICK_REPETITIONS = 2


@dataclass(frozen=True)
class BenchmarkSettings:
    duration_s: float
    frames_per_second: float
    width: int
    height: int
    repetitions: int
    profile_filters: bool = False


@dataclass(frozen=True)
class CheckRun:
    seconds: float
    decoded_frame_count: int
    instrument_cache_path: Path
    instrument_cache_bytes: int


@dataclass(frozen=True)
class FilterProfile:
    name: str
    seconds: float


@dataclass(frozen=True)
class BenchmarkResult:
    transform_seconds: float
    camera_topic: str
    ffmpeg_version: str
    logical_cpu_count: int
    check_runs: tuple[CheckRun, ...]
    filter_profiles: tuple[FilterProfile, ...]


def _measure_cold_check(
    canonical_path: Path,
    camera_topic: str,
    frames_per_second: float,
    workdir: Path,
) -> CheckRun:
    started = time.perf_counter()
    with hflow.Episode(canonical_path, workdir=workdir) as episode:
        result = camera_frame_stats(
            episode,
            cameras=[camera_topic],
            expected_hz={camera_topic: frames_per_second},
        )
    seconds = time.perf_counter() - started
    instrument_cache_paths = list(workdir.glob("*.instrument.*.txt"))
    if len(instrument_cache_paths) != 1:
        raise RuntimeError(
            f"expected one cold instrument cache in {workdir}, got {instrument_cache_paths}"
        )
    decoded_frame_count = result.measurements[f"{camera_topic}/decoded_frame_count"]
    if not isinstance(decoded_frame_count, int):
        raise TypeError(f"decoded frame count is not an integer: {decoded_frame_count!r}")
    return CheckRun(
        seconds=seconds,
        decoded_frame_count=decoded_frame_count,
        instrument_cache_path=instrument_cache_paths[0],
        instrument_cache_bytes=instrument_cache_paths[0].stat().st_size,
    )


def _profile_filter_paths(video_path: Path) -> tuple[FilterProfile, ...]:
    settings = FrameStatisticsSettings()
    filters: tuple[tuple[str, str | None], ...] = (
        ("decode only", None),
        ("decode + format=yuv420p", "format=pix_fmts=yuv420p"),
        (
            "decode + blackframe",
            "format=pix_fmts=yuv420p,"
            f"blackframe=amount=0:threshold={settings.black_pixel_luma_threshold}",
        ),
        (
            "decode + freezedetect",
            "format=pix_fmts=yuv420p,"
            "freezedetect="
            f"n={settings.freeze_noise_tolerance_decibels}dB:"
            f"d={settings.freeze_minimum_duration_seconds}",
        ),
        (
            "decode + signalstats=stat=tout+brng",
            "format=pix_fmts=yuv420p,signalstats=stat=tout+brng",
        ),
        ("complete shipped graph", frame_statistics_filter_graph(settings)),
    )
    profiles: list[FilterProfile] = []
    for name, filter_graph in filters:
        command = [
            str(ffmpeg_path()),
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostats",
            "-i",
            str(video_path),
        ]
        if filter_graph is not None:
            command.extend(("-vf", filter_graph))
        command.extend(("-f", "null", "-"))
        started = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, check=False)
        seconds = time.perf_counter() - started
        if completed.returncode != 0:
            raise RuntimeError(
                f"ffmpeg filter profile {name!r} failed: "
                f"{completed.stderr.decode(errors='replace')}"
            )
        profiles.append(FilterProfile(name=name, seconds=seconds))
    return tuple(profiles)


def run_synthetic_benchmark(settings: BenchmarkSettings) -> BenchmarkResult:
    with tempfile.TemporaryDirectory(prefix="camera-frame-stats-benchmark-") as directory_name:
        working_dir = Path(directory_name)
        source_path = synthesize_episode(
            working_dir / "source.mcap",
            SyntheticEpisodeSpec(
                duration_s=settings.duration_s,
                cameras=("camera_0",),
                image_hz=settings.frames_per_second,
                image_width=settings.width,
                image_height=settings.height,
                black_segment=None,
                joint_jump_at_s=None,
                timestamp_offset_segment=None,
            ),
        )
        canonical_path = working_dir / "canonical.mcap"
        transform_started = time.perf_counter()
        write_canonical_episode(source_path, canonical_path, TransformConfig())
        transform_seconds = time.perf_counter() - transform_started
        camera_topic = "/camera_0/compressed"
        check_runs = tuple(
            _measure_cold_check(
                canonical_path,
                camera_topic,
                settings.frames_per_second,
                working_dir / f"cold-check-{repetition + 1}",
            )
            for repetition in range(settings.repetitions)
        )
        if settings.profile_filters:
            with hflow.Episode(canonical_path, workdir=working_dir / "filter-profile") as episode:
                filter_profiles = _profile_filter_paths(episode.video(camera_topic))
        else:
            filter_profiles = ()
    return BenchmarkResult(
        transform_seconds=transform_seconds,
        camera_topic=camera_topic,
        ffmpeg_version=ffmpeg_version(),
        logical_cpu_count=os.cpu_count() or 1,
        check_runs=check_runs,
        filter_profiles=filter_profiles,
    )


def _settings(quick: bool, profile_filters: bool) -> BenchmarkSettings:
    if quick:
        return BenchmarkSettings(
            duration_s=QUICK_DURATION_S,
            frames_per_second=QUICK_FRAMES_PER_SECOND,
            width=QUICK_WIDTH,
            height=QUICK_HEIGHT,
            repetitions=QUICK_REPETITIONS,
            profile_filters=profile_filters,
        )
    return BenchmarkSettings(
        duration_s=FULL_DURATION_S,
        frames_per_second=FULL_FRAMES_PER_SECOND,
        width=FULL_WIDTH,
        height=FULL_HEIGHT,
        repetitions=FULL_REPETITIONS,
        profile_filters=profile_filters,
    )


def _print_result(settings: BenchmarkSettings, result: BenchmarkResult) -> None:
    print(
        "camera_frame_stats benchmark: "
        f"{settings.duration_s:g}s synthetic episode, one camera @ "
        f"{settings.frames_per_second:g} Hz, {settings.width}x{settings.height}\n"
    )
    print(f"logical CPUs: {result.logical_cpu_count}")
    print(f"ffmpeg: {result.ffmpeg_version}\n")
    print("| phase | wall-clock | decoded frames | instrument cache |")
    print("|---|---:|---:|---:|")
    print(f"| transform to canonical | {result.transform_seconds:.3f} s | - | - |")
    for run_index, run in enumerate(result.check_runs, start=1):
        print(
            f"| cold camera_frame_stats #{run_index} | {run.seconds:.3f} s "
            f"| {run.decoded_frame_count} | {run.instrument_cache_bytes / 1000:.1f} KB |"
        )
    median_seconds = statistics.median(run.seconds for run in result.check_runs)
    print(f"\ncold check median: {median_seconds:.3f} s per camera")
    if result.filter_profiles:
        print("\n| filter path | wall-clock |")
        print("|---|---:|")
        for profile in result.filter_profiles:
            print(f"| {profile.name} | {profile.seconds:.3f} s |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="use a small development fixture")
    parser.add_argument(
        "--profile-filters",
        action="store_true",
        help="time decode and each evidence-filter component separately",
    )
    arguments = parser.parse_args()
    settings = _settings(arguments.quick, arguments.profile_filters)
    _print_result(settings, run_synthetic_benchmark(settings))


if __name__ == "__main__":
    main()
