import importlib.util
from pathlib import Path
from types import ModuleType

BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "benchmarks/camera_frame_stats_benchmark.py"


def _load_benchmark() -> ModuleType:
    spec = importlib.util.spec_from_file_location("camera_frame_stats_benchmark", BENCHMARK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_camera_frame_stats_benchmark_measures_each_cold_run_from_canonical_input() -> None:
    """Reusing one workdir or measuring the pre-transform source breaks this result."""
    benchmark = _load_benchmark()

    result = benchmark.run_synthetic_benchmark(
        benchmark.BenchmarkSettings(
            duration_s=1.0,
            frames_per_second=10.0,
            width=160,
            height=90,
            repetitions=2,
            profile_filters=True,
        )
    )

    assert result.transform_seconds > 0
    assert result.camera_topic == "/camera_0/compressed"
    assert result.ffmpeg_version
    assert result.logical_cpu_count > 0
    assert len(result.check_runs) == 2
    assert [run.decoded_frame_count for run in result.check_runs] == [10, 10]
    assert all(run.seconds > 0 for run in result.check_runs)
    assert all(run.instrument_cache_bytes > 0 for run in result.check_runs)
    assert len({run.instrument_cache_path.parent for run in result.check_runs}) == 2
    assert [profile.name for profile in result.filter_profiles] == [
        "decode only",
        "decode + format=yuv420p",
        "decode + blackframe",
        "decode + freezedetect",
        "decode + signalstats=stat=tout+brng",
        "complete shipped graph",
    ]
    assert all(profile.seconds > 0 for profile in result.filter_profiles)
