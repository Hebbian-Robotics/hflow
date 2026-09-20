"""Verify actual selected footage and timing using small synthetic videos."""

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import pytest

from hflow import (
    KeyframeFallbackReason,
    SourceFrameSampling,
    SourceSamplingError,
    SourceSamplingMode,
    SourceWindow,
    plan_source_windows,
    sample_source_frames,
)
from hflow.ffmpeg import ffmpeg_path, ffprobe_path


@pytest.fixture
def color_video(tmp_path: Path) -> Path:
    source_path = tmp_path / "colors.mp4"
    arguments = [str(ffmpeg_path()), "-hide_banner", "-loglevel", "error", "-nostdin"]
    for color in ("red", "green", "blue"):
        arguments.extend(("-f", "lavfi", "-i", f"color=c={color}:s=96x64:r=10:d=2"))
    arguments.extend(
        (
            "-filter_complex",
            "[0:v][1:v][2:v]concat=n=3:v=1:a=0[out]",
            "-map",
            "[out]",
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-g",
            "20",
            "-keyint_min",
            "20",
            "-sc_threshold",
            "0",
            "-bf",
            "2",
            str(source_path),
        )
    )
    subprocess.run(arguments, check=True, capture_output=True, timeout=30)
    return source_path


@pytest.mark.parametrize("canvas", [(640, 360), (1280, 960)])
def test_keyframes_preserve_timestamps_content_and_aspect_after_a_seek(
    color_video: Path, tmp_path: Path, canvas: tuple[int, int]
) -> None:
    original_digest = hashlib.sha256(color_video.read_bytes()).digest()
    samples = sample_source_frames(
        color_video,
        tmp_path / "samples",
        window=SourceWindow(250, 4750),
        settings=SourceFrameSampling(
            mode=SourceSamplingMode.KEYFRAMES, width=canvas[0], height=canvas[1]
        ),
    )
    assert [frame.timestamp_seconds for frame in samples.frames] == [2, 4]
    assert samples.actual_mode is SourceSamplingMode.KEYFRAMES
    assert samples.fallback_reason is None
    for frame, expected_channel in zip(samples.frames, (1, 0), strict=True):
        pixels = cv2.imread(str(frame.path))
        assert pixels is not None
        assert pixels.shape == (canvas[1], canvas[0], 3)
        center_pixel = pixels[canvas[1] // 2, canvas[0] // 2].astype(int)
        assert center_pixel[expected_channel] > np.delete(center_pixel, expected_channel).max() + 80
        content_rows, content_columns = np.where(pixels.max(axis=2) > 30)
        assert (np.ptp(content_columns) + 1) / (np.ptp(content_rows) + 1) == pytest.approx(
            1.5, abs=0.02
        )
    assert hashlib.sha256(color_video.read_bytes()).digest() == original_digest


@pytest.mark.parametrize("start_millis", [250, 251])
def test_uniform_sampling_reports_actual_frames_instead_of_requested_ticks(
    color_video: Path, tmp_path: Path, start_millis: int
) -> None:
    samples = sample_source_frames(
        color_video, tmp_path / "samples", window=SourceWindow(start_millis, start_millis + 4000)
    )
    assert [frame.timestamp_seconds for frame in samples.frames] == [
        Fraction(3, 10),
        Fraction(13, 10),
        Fraction(23, 10),
        Fraction(33, 10),
    ]
    assert samples.actual_mode is SourceSamplingMode.UNIFORM
    assert samples.fallback_reason is None


@pytest.mark.parametrize(
    ("window", "expected_mode", "expected_reason"),
    [
        (
            SourceWindow(250, 1250),
            SourceSamplingMode.UNIFORM,
            KeyframeFallbackReason.TOO_FEW_KEYFRAMES,
        ),
        (
            SourceWindow(250, 4750),
            SourceSamplingMode.UNIFORM,
            KeyframeFallbackReason.INSUFFICIENT_SPAN,
        ),
        (SourceWindow(0, 4000), SourceSamplingMode.KEYFRAMES, None),
    ],
)
def test_keyframe_fallback_depends_only_on_temporal_coverage(
    color_video: Path,
    tmp_path: Path,
    window: SourceWindow,
    expected_mode: SourceSamplingMode,
    expected_reason: KeyframeFallbackReason | None,
) -> None:
    samples = sample_source_frames(
        color_video,
        tmp_path / "fallback",
        window=window,
        settings=SourceFrameSampling(mode=SourceSamplingMode.KEYFRAMES_FIRST),
    )
    assert samples.actual_mode is expected_mode
    assert samples.fallback_reason is expected_reason
    assert samples.frames
    assert all(
        Fraction(window.start_millis, 1000)
        <= frame.timestamp_seconds
        < Fraction(window.end_millis, 1000)
        for frame in samples.frames
    )


def test_a_window_without_keyframes_is_an_explicit_empty_result(
    color_video: Path, tmp_path: Path
) -> None:
    samples = sample_source_frames(
        color_video,
        tmp_path / "empty",
        window=SourceWindow(250, 1250),
        settings=SourceFrameSampling(mode=SourceSamplingMode.KEYFRAMES),
    )
    assert samples.frames == ()
    assert samples.actual_mode is SourceSamplingMode.KEYFRAMES
    assert samples.fallback_reason is None


def test_planned_windows_include_each_source_frame_once(color_video: Path, tmp_path: Path) -> None:
    windows = plan_source_windows(6000, maximum_window_millis=2500)
    frames = tuple(
        frame
        for window_index, window in enumerate(windows)
        for frame in sample_source_frames(
            color_video,
            tmp_path / f"window-{window_index}",
            window=window,
            settings=SourceFrameSampling(maximum_frames=30, minimum_interval_millis=100),
        ).frames
    )
    assert [frame.timestamp_seconds for frame in frames] == [
        Fraction(index, 10) for index in range(60)
    ]


def test_sparse_variable_rate_video_does_not_fill_gaps_or_include_the_end(tmp_path: Path) -> None:
    source_path = tmp_path / "variable.mp4"
    subprocess.run(
        [
            str(ffmpeg_path()),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=96x64:rate=10:duration=0.6",
            "-vf",
            "setpts=(N+8*floor(N/2))/(10*TB)",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-g",
            "2",
            "-bf",
            "0",
            str(source_path),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    samples = sample_source_frames(
        source_path,
        tmp_path / "samples",
        window=SourceWindow(0, 2000),
        settings=SourceFrameSampling(minimum_interval_millis=200),
    )
    assert [frame.timestamp_seconds for frame in samples.frames] == [0, 1]


def test_fractional_frame_times_and_bin_boundaries_are_exact(tmp_path: Path) -> None:
    source_path = tmp_path / "thirds.mp4"
    subprocess.run(
        [
            str(ffmpeg_path()),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=96x64:rate=3:duration=2",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            str(source_path),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    samples = sample_source_frames(
        source_path,
        tmp_path / "samples",
        window=SourceWindow(0, 2000),
        settings=SourceFrameSampling(maximum_frames=6, minimum_interval_millis=1),
    )
    assert [frame.timestamp_seconds for frame in samples.frames] == [
        Fraction(index, 3) for index in range(6)
    ]


@pytest.mark.parametrize("maximum_frames", [1, 4, 16])
def test_dense_keyframes_cover_the_window_with_a_configurable_cap(
    tmp_path: Path, maximum_frames: int
) -> None:
    source_path = tmp_path / "dense.mp4"
    subprocess.run(
        [
            str(ffmpeg_path()),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=96x64:rate=10:duration=4",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-g",
            "1",
            str(source_path),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    samples = sample_source_frames(
        source_path,
        tmp_path / "samples",
        window=SourceWindow(0, 4000),
        settings=SourceFrameSampling(
            mode=SourceSamplingMode.KEYFRAMES, maximum_frames=maximum_frames
        ),
    )
    timestamps = [frame.timestamp_seconds for frame in samples.frames]
    assert len(set(timestamps)) == maximum_frames
    assert timestamps[0] == 0
    assert 4 - Fraction(4, maximum_frames) <= timestamps[-1] < 4


def test_playback_origin_handles_a_nonzero_container_start(
    color_video: Path, tmp_path: Path
) -> None:
    shifted_path = tmp_path / "shifted.mp4"
    subprocess.run(
        [
            str(ffmpeg_path()),
            "-v",
            "error",
            "-i",
            str(color_video),
            "-c",
            "copy",
            "-output_ts_offset",
            "5",
            str(shifted_path),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    probe = subprocess.run(
        [
            str(ffprobe_path()),
            "-v",
            "error",
            "-show_entries",
            "format=start_time",
            "-of",
            "json",
            str(shifted_path),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    assert float(json.loads(probe.stdout)["format"]["start_time"]) == 5
    samples = sample_source_frames(
        shifted_path, tmp_path / "samples", window=SourceWindow(250, 1250)
    )
    assert [frame.timestamp_seconds for frame in samples.frames] == [Fraction(3, 10)]


@pytest.mark.parametrize("limit", ["maximum_log_bytes", "maximum_frame_bytes"])
def test_resource_failures_remove_partial_output(
    color_video: Path, tmp_path: Path, limit: str
) -> None:
    output_directory = tmp_path / "samples"
    with pytest.raises(SourceSamplingError):
        sample_source_frames(
            color_video,
            output_directory,
            window=SourceWindow(0, 1000),
            settings=replace(SourceFrameSampling(), **{limit: 1}),
        )
    assert not output_directory.exists()


def test_timeout_terminates_extraction_and_removes_partial_output(
    color_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A real stalled subprocess exercises cancellation and reaping at the
    # executable boundary, without depending on the speed of a video decoder.
    stalled_executable = tmp_path / "stalled-ffmpeg"
    stalled_executable.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(30)\n")
    stalled_executable.chmod(0o700)
    monkeypatch.setattr("hflow.ffmpeg.ffmpeg_path", lambda: stalled_executable)
    output_directory = tmp_path / "samples"
    with pytest.raises(SourceSamplingError, match=r"timed out|time limit"):
        sample_source_frames(
            color_video,
            output_directory,
            window=SourceWindow(0, 1000),
            settings=SourceFrameSampling(timeout_seconds=0.1),
        )
    assert not output_directory.exists()


def test_failed_decode_cleans_output_and_existing_destinations_are_preserved(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "broken.mp4"
    source_path.write_bytes(b"not a video")
    output_directory = tmp_path / "samples"
    with pytest.raises(SourceSamplingError):
        sample_source_frames(source_path, output_directory, window=SourceWindow(0, 1000))
    assert not output_directory.exists()
    output_directory.mkdir()
    marker_path = output_directory / "existing.txt"
    marker_path.write_text("keep")
    with pytest.raises(FileExistsError):
        sample_source_frames(source_path, output_directory, window=SourceWindow(0, 1000))
    assert marker_path.read_text() == "keep"


def test_oversized_windows_are_rejected_before_output_creation(
    color_video: Path, tmp_path: Path
) -> None:
    output_directory = tmp_path / "samples"
    with pytest.raises(ValueError, match="maximum_window_millis"):
        sample_source_frames(color_video, output_directory, window=SourceWindow(0, 120_001))
    assert not output_directory.exists()


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("mode", "uniform"),
        ("maximum_frames", True),
        ("maximum_frames", 0),
        ("minimum_interval_millis", 0),
        ("maximum_window_millis", -1),
        ("width", 1),
        ("height", 0),
        ("timeout_seconds", float("nan")),
        ("timeout_seconds", float("inf")),
        ("maximum_frame_bytes", 0),
        ("maximum_log_bytes", 0),
    ],
)
def test_sampling_settings_refuse_invalid_limits(field_name: str, value: object) -> None:
    with pytest.raises(ValueError):
        replace(SourceFrameSampling(), **{field_name: value})


def test_long_source_sampling_retains_the_frame_cap(tmp_path: Path) -> None:
    source_path = tmp_path / "long.mp4"
    subprocess.run(
        [
            str(ffmpeg_path()),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=96x64:rate=1:duration=130",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-g",
            "130",
            "-keyint_min",
            "130",
            "-sc_threshold",
            "0",
            str(source_path),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    samples = sample_source_frames(
        source_path,
        tmp_path / "samples",
        window=SourceWindow(0, 130_000),
        settings=SourceFrameSampling(
            mode=SourceSamplingMode.KEYFRAMES_FIRST,
            maximum_window_millis=130_000,
        ),
    )
    assert len(samples.frames) == 16
    assert samples.frames[0].timestamp_seconds == 0
    assert 121 <= samples.frames[-1].timestamp_seconds < 130
    assert samples.actual_mode is SourceSamplingMode.UNIFORM
    assert samples.fallback_reason is KeyframeFallbackReason.TOO_FEW_KEYFRAMES


def test_example_emits_complete_windows_with_readable_frame_paths(
    color_video: Path, tmp_path: Path
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(repository_root / "examples" / "sample_source_video.py"),
            str(color_video),
            "--output",
            str(tmp_path / "example-output"),
            "--maximum-window-millis",
            "2500",
            "--maximum-frames",
            "4",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    records = [json.loads(line) for line in completed.stdout.splitlines()]
    assert [(record["start_millis"], record["end_millis"]) for record in records] == [
        (0, 2000),
        (2000, 4000),
        (4000, 6000),
    ]
    frames = [frame for record in records for frame in record["frames"]]
    assert [frame["timestamp_seconds"] for frame in frames] == list(range(6))
    assert all(cv2.imread(frame["path"]) is not None for frame in frames)


@pytest.mark.parametrize("timestamp_offset", [0, 5, -1])
def test_nearest_keyframes_preserve_ties_pixels_and_playback_origin(
    color_video: Path, tmp_path: Path, timestamp_offset: int
) -> None:
    from hflow.source_sampling import SourceFrameResize

    shifted_source = tmp_path / ("shifted.ts" if timestamp_offset < 0 else "shifted.mp4")
    subprocess.run(
        [
            str(ffmpeg_path()),
            "-v",
            "error",
            "-i",
            str(color_video),
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            "-output_ts_offset",
            str(timestamp_offset),
            "-avoid_negative_ts",
            "disabled",
            str(shifted_source),
        ],
        check=True,
        capture_output=True,
    )
    samples = sample_source_frames(
        shifted_source,
        tmp_path / "nearest",
        window=SourceWindow(0, 6000),
        settings=SourceFrameSampling(
            mode=SourceSamplingMode.NEAREST_KEYFRAMES,
            keyframe_positions=(0.15, 0.5, 0.85),
            resize=SourceFrameResize.FIT,
            width=960,
            height=960,
            jpeg_quality=2,
            scaling_algorithm="bicubic",
        ),
    )
    assert [frame.timestamp_seconds for frame in samples.frames] == [0, 2, 4]
    assert samples.actual_mode is SourceSamplingMode.NEAREST_KEYFRAMES
    assert samples.fallback_reason is None
    for frame, dominant_channel in zip(samples.frames, (2, 1, 0), strict=True):
        pixels = cv2.imread(str(frame.path))
        assert pixels is not None
        assert pixels.shape == (640, 960, 3)
        assert int(pixels[320, 480, dominant_channel]) > 100
        assert np.all(pixels.max(axis=2) > 30)


def test_nearest_keyframes_deduplicate_and_leave_empty_windows_empty(
    color_video: Path, tmp_path: Path
) -> None:
    settings = SourceFrameSampling(
        mode=SourceSamplingMode.NEAREST_KEYFRAMES,
        keyframe_positions=(0.2, 0.5, 0.8),
    )
    sparse = sample_source_frames(
        color_video, tmp_path / "sparse", window=SourceWindow(0, 1900), settings=settings
    )
    empty = sample_source_frames(
        color_video, tmp_path / "empty-nearest", window=SourceWindow(250, 1250), settings=settings
    )
    assert [frame.timestamp_seconds for frame in sparse.frames] == [0]
    assert empty.frames == ()
    assert empty.fallback_reason is None
