"""One shared window decode must measure exactly what the file-level measurements see."""

from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from hflow.blur import BlurSummary, measure_video_blur
from hflow.camera_motion import (
    CameraMotionStreamSettings,
    CameraShakeSettings,
    filter_camera_shake,
    stream_camera_motion,
    summarize_camera_shake,
)
from hflow.ffmpeg import ffmpeg_path
from hflow.media import UnreadableVideo, VideoProperties, VideoWindow, probe_video
from hflow.video_statistics import (
    FrameStatisticsSettings,
    LumaRangePolicy,
    VideoFrameStatistics,
    VideoMeasurementToolchain,
    measure_video_frame_statistics,
)
from hflow.window_measurements import (
    IndependentVideoWindowMeasurements,
    MeasurementFailure,
    VideoWindowMeasurements,
    WindowMeasurementSelection,
    measure_video_window,
    measure_video_window_independently,
)

FRAMES_PER_SECOND = 16.0
MEASURED_WINDOW = VideoWindow(start_seconds=1.0, duration_seconds=3.0, frames_per_second=16.0)
FULL_RANGE_STATISTICS = FrameStatisticsSettings(luma_range=LumaRangePolicy.FULL)
SHAKE_SETTINGS = CameraShakeSettings(half_window_pairs=8, horizontal_field_of_view_degrees=90.0)


def run_ffmpeg(*arguments: str) -> None:
    subprocess.run(
        [str(ffmpeg_path()), "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *arguments],
        check=True,
    )


def write_moving_colour_source(output: Path, *, display_rotation_degrees: int | None) -> Path:
    """Moving colour footage, optionally tagged with display rotation instead of re-encoded."""
    upright = output.with_name(f"{output.stem}-upright.mp4")
    run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=320x240:rate=30:duration=6",
        "-vf",
        "scroll=horizontal=0.004:vertical=0.002",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(upright),
    )
    if display_rotation_degrees is None:
        return upright
    run_ffmpeg(
        "-display_rotation",
        str(display_rotation_degrees),
        "-i",
        str(upright),
        "-c",
        "copy",
        str(output),
    )
    return output


def write_uncompressed_window(source: Path, output: Path) -> Path:
    """The frames the shared decode must see: same seek, fps and default rotation."""
    run_ffmpeg(
        "-ss",
        f"{MEASURED_WINDOW.start_seconds:.3f}",
        "-t",
        f"{MEASURED_WINDOW.duration_seconds:.3f}",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-vf",
        f"fps={FRAMES_PER_SECOND:g}",
        "-an",
        "-c:v",
        "rawvideo",
        str(output),
    )
    return output


@pytest.mark.parametrize("display_rotation_degrees", [None, 90], ids=["upright", "rotated"])
def test_shared_decode_equals_file_measurements_of_the_same_frames(
    tmp_path: Path, display_rotation_degrees: int | None
) -> None:
    source = write_moving_colour_source(
        tmp_path / "source.mp4", display_rotation_degrees=display_rotation_degrees
    )
    reference_window = write_uncompressed_window(source, tmp_path / "reference.mkv")
    reference_properties = probe_video(reference_window)
    assert isinstance(reference_properties, VideoProperties)
    assert (reference_properties.width, reference_properties.height) == (
        (240, 320) if display_rotation_degrees == 90 else (320, 240)
    )

    measurements = measure_video_window(
        source,
        MEASURED_WINDOW,
        WindowMeasurementSelection(
            frame_statistics=FULL_RANGE_STATISTICS, blur=True, camera_shake=SHAKE_SETTINGS
        ),
    )

    assert isinstance(measurements, VideoWindowMeasurements)
    reference_statistics = measure_video_frame_statistics(
        reference_window, settings=FULL_RANGE_STATISTICS
    )
    with stream_camera_motion(
        reference_window,
        settings=CameraMotionStreamSettings(frames_per_second=FRAMES_PER_SECOND),
    ) as motion_observations:
        reference_shake = summarize_camera_shake(
            filter_camera_shake(motion_observations, settings=SHAKE_SETTINGS)
        )
    assert measurements.frame_statistics is not None
    # The shared decode keeps the exact fps cadence; the reference file's millisecond
    # container timestamps can lengthen the final frame by up to one millisecond.
    assert measurements.frame_statistics.duration_seconds == 48 / FRAMES_PER_SECOND
    assert 0 <= reference_statistics.duration_seconds - 48 / FRAMES_PER_SECOND <= 0.0011
    assert (
        replace(
            measurements.frame_statistics, duration_seconds=reference_statistics.duration_seconds
        )
        == reference_statistics
    )
    assert measurements.blur == measure_video_blur(reference_window)
    assert measurements.camera_shake == reference_shake
    assert reference_shake.measured_shake_pair_count > 0
    assert measurements.decoded_frame_count == reference_statistics.decoded_frame_count == 48


def test_only_selected_measurements_are_computed(tmp_path: Path) -> None:
    source = write_moving_colour_source(tmp_path / "source.mp4", display_rotation_degrees=None)

    measurements = measure_video_window(
        source, MEASURED_WINDOW, WindowMeasurementSelection(blur=True)
    )

    assert isinstance(measurements, VideoWindowMeasurements)
    assert measurements.frame_statistics is None
    assert measurements.camera_shake is None
    assert measurements.blur is not None
    assert (measurements.blur.frame_count, measurements.blur.scored_frame_count) == (48, 48)
    with pytest.raises(ValueError, match="at least one"):
        WindowMeasurementSelection()


def test_independent_measurements_preserve_other_branches_after_motion_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hflow.window_measurements as window_measurements

    source = write_moving_colour_source(tmp_path / "source.mp4", display_rotation_degrees=None)

    def fail_motion_stream(*_arguments: object, **_settings: object) -> None:
        raise RuntimeError("motion calculation failed")

    monkeypatch.setattr(window_measurements, "iter_frame_motion", fail_motion_stream)
    result = measure_video_window_independently(
        source,
        MEASURED_WINDOW,
        WindowMeasurementSelection(
            frame_statistics=FULL_RANGE_STATISTICS, blur=True, camera_shake=SHAKE_SETTINGS
        ),
    )
    assert isinstance(result, IndependentVideoWindowMeasurements)
    assert result.decoded_frame_count == 48
    assert isinstance(result.frame_statistics, VideoFrameStatistics)
    assert isinstance(result.blur, BlurSummary)
    assert result.camera_shake == MeasurementFailure("RuntimeError")


def test_windows_without_decodable_frames_are_unreadable(tmp_path: Path) -> None:
    source = write_moving_colour_source(tmp_path / "source.mp4", display_rotation_degrees=None)
    truncated = tmp_path / "truncated.mp4"
    truncated.write_bytes(source.read_bytes()[:2048])
    selection = WindowMeasurementSelection(
        frame_statistics=FULL_RANGE_STATISTICS, blur=True, camera_shake=SHAKE_SETTINGS
    )

    after_source_end = measure_video_window(
        source, replace(MEASURED_WINDOW, start_seconds=7.0), selection
    )
    damaged_source = measure_video_window(truncated, MEASURED_WINDOW, selection)

    assert isinstance(after_source_end, UnreadableVideo)
    assert isinstance(damaged_source, UnreadableVideo)


def test_caller_toolchain_probes_the_source(tmp_path: Path) -> None:
    source = write_moving_colour_source(tmp_path / "source.mp4", display_rotation_degrees=None)
    rejecting_ffprobe = tmp_path / "ffprobe"
    rejecting_ffprobe.write_text(
        "#!/bin/sh\necho 'Invalid data found when processing input' >&2\nexit 1\n"
    )
    rejecting_ffprobe.chmod(0o755)
    toolchain = VideoMeasurementToolchain(
        ffmpeg_executable=ffmpeg_path(),
        ffprobe_executable=rejecting_ffprobe,
        ffmpeg_version="caller ffmpeg",
        ffprobe_version="caller ffprobe",
    )

    # The default toolchain reads this source; only the caller's probe rejects it.
    measurements = measure_video_window(
        source, MEASURED_WINDOW, WindowMeasurementSelection(blur=True), toolchain=toolchain
    )

    assert isinstance(measurements, UnreadableVideo)
