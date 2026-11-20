"""Oversized declared frame dimensions must be refused before a raw-frame read.

``probe_video`` and the video importer already reject a frame whose
``width * height`` exceeds ``VideoLimits.maximum_frame_pixels``. Camera-motion
streaming, camera-motion measurement, raw RGB sampling, and the opt-in
``camera_stability`` check do not: they trust the dimensions ffprobe reports
and read one raw frame of that size. These tests fail until those readers
apply the same budget and never allocate a raw frame larger than it.

``window_measurements`` is not covered here. Its luma reader runs only after
``probe_video`` has already returned ``UnsupportedVideo`` for the same budget.
``mediapipe_hands`` is covered by the ``rgb_frames`` cases: it calls that
reader with no pixel check of its own, both when ``inference_long_edge_pixels``
is unset and when it scales the sample.
"""

from __future__ import annotations

import asyncio
import subprocess
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from mcap_test_helpers import write_compressed_video_mcap
from media_test_helpers import render_lavfi, run_ffprobe

from hflow._video_measurement_toolchain import resolved_video_measurement_toolchain
from hflow._video_measurements._camera_motion import CameraMotionSettings, measure_camera_motion
from hflow._video_measurements._raw_frames import (
    LUMA_FRAME_FILTER_GRAPH,
    RawFrameError,
    _raw_frame_stream,
    luma_frames,
    rgb_frames,
)
from hflow._video_measurements._toolchain import VideoMeasurementToolchain
from hflow.camera_motion import CameraMotionStreamSettings, stream_camera_motion
from hflow.checks import camera_stability
from hflow.episode import Episode
from hflow.media import VideoLimits
from hflow.video import canonical_x264_parameters, split_annex_b_stream

pytestmark = pytest.mark.requires_system_ffmpeg

_PIXEL_BUDGET = VideoLimits().maximum_frame_pixels
_OVER_BUDGET_WIDTH = 8192
# 8192 * 4096 == 33554432, the default budget. Two extra rows is the smallest
# even height that crosses it, so H.264 4:2:0 and a gray header share one size.
_OVER_BUDGET_HEIGHT = 4098
_EXACT_BUDGET_HEIGHT = 4096
_FRAME_RATE = 10


def _over_budget_pixels() -> int:
    pixels = _OVER_BUDGET_WIDTH * _OVER_BUDGET_HEIGHT
    assert pixels > _PIXEL_BUDGET
    assert _OVER_BUDGET_WIDTH * _EXACT_BUDGET_HEIGHT == _PIXEL_BUDGET
    return pixels


def _require_probed_dimensions(video: Path, width: int, height: int) -> None:
    """The fixture must be a file ffprobe actually reports at this size."""
    probed = run_ffprobe(
        "-hide_banner",
        "-loglevel",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0",
        str(video),
    ).strip()
    assert probed == f"{width},{height}"
    # The read size is this product. Guard the assertion against a numpy
    # overflow that would make a smaller allocation look like a refusal.
    assert int(np.prod((height, width), dtype=np.int64)) == width * height


def _requests_raw_frame_decode(command: tuple[str, ...]) -> bool:
    """True when an ffmpeg command emits a raw frame stream this process will read."""
    for index, flag in enumerate(command[:-1]):
        if flag == "-f" and command[index + 1] in {"rawvideo", "framemd5"}:
            return True
    return False


def _assert_refused_before_allocating(
    declared_frame_bytes: int, operation: Callable[[], None]
) -> None:
    """The call must raise before a raw frame of the declared size is read.

    A refusal that still performs that read fails this assertion: CPython
    allocates the requested size before it learns the producer sent less, and
    ffmpeg allocates the decoded frame before it can scale one down. The
    message has to name the pixel budget so a truncated-frame error, which
    also raises ``RawFrameError``, does not satisfy the test. ffprobe may run.
    A rawvideo or framemd5 ffmpeg process may not.
    """
    assert declared_frame_bytes > _PIXEL_BUDGET
    tracing = tracemalloc.is_tracing()
    if not tracing:
        tracemalloc.start()
    tracemalloc.reset_peak()
    commands: list[tuple[str, ...]] = []
    real_popen = subprocess.Popen

    def recording_popen(*popen_args: Any, **popen_kwargs: Any) -> subprocess.Popen[bytes]:
        args = popen_args[0]
        commands.append(tuple(str(part) for part in args))
        return cast("subprocess.Popen[bytes]", real_popen(*popen_args, **popen_kwargs))

    subprocess.Popen = recording_popen  # ty: ignore[invalid-assignment]
    try:
        with pytest.raises(RawFrameError, match=str(_PIXEL_BUDGET)):
            operation()
    finally:
        subprocess.Popen = real_popen
        _current, peak = tracemalloc.get_traced_memory()
        if not tracing:
            tracemalloc.stop()
    assert peak < declared_frame_bytes, (
        f"allocated {peak} bytes before refusing a {declared_frame_bytes}-byte frame"
    )
    raw_decodes = [command for command in commands if _requests_raw_frame_decode(command)]
    assert not raw_decodes, f"decoded a raw frame before refusing it: {raw_decodes}"


def _write_declared_header(path: Path, width: int, height: int) -> None:
    """A Y4M whose header names ``width`` x ``height`` and whose payload does not.

    Sixteen payload bytes cannot fill the declared frame. ffprobe still
    reports the header dimensions, which is what the raw reader multiplies.
    """
    header = f"YUV4MPEG2 W{width} H{height} F{_FRAME_RATE}:1 Ip A1:1 Cmono\nFRAME\n"
    path.write_bytes(header.encode("ascii") + b"\x00" * 16)
    _require_probed_dimensions(path, width, height)


def _write_encoded_video(path: Path, width: int, height: int, *, frames: int) -> None:
    render_lavfi(
        path,
        f"color=c=black:s={width}x{height}:r={_FRAME_RATE}:d=1",
        output_arguments=(
            "-frames:v",
            str(frames),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "1",
            "-bf",
            "0",
            "-x264-params",
            canonical_x264_parameters(1),
        ),
    )
    _require_probed_dimensions(path, width, height)


@pytest.fixture(scope="module")
def lying_over_budget_header(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("lying-header") / "declared-over-budget.y4m"
    _write_declared_header(path, _OVER_BUDGET_WIDTH, _OVER_BUDGET_HEIGHT)
    return path


@pytest.fixture(scope="module")
def decodable_over_budget_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("decodable-over-budget") / "over-budget.mp4"
    _write_encoded_video(path, _OVER_BUDGET_WIDTH, _OVER_BUDGET_HEIGHT, frames=2)
    return path


@pytest.fixture(scope="module")
def toolchain() -> VideoMeasurementToolchain:
    return resolved_video_measurement_toolchain()


def test_luma_frames_refuse_a_header_that_declares_an_over_budget_frame(
    lying_over_budget_header: Path, toolchain: VideoMeasurementToolchain
) -> None:
    def read_frames() -> None:
        with luma_frames(lying_over_budget_header, toolchain=toolchain) as frames:
            list(frames)

    _assert_refused_before_allocating(_over_budget_pixels(), read_frames)


def test_luma_frames_refuse_a_decodable_frame_over_the_pixel_budget(
    decodable_over_budget_video: Path, toolchain: VideoMeasurementToolchain
) -> None:
    def read_frames() -> None:
        with luma_frames(decodable_over_budget_video, toolchain=toolchain) as frames:
            list(frames)

    _assert_refused_before_allocating(_over_budget_pixels(), read_frames)


def test_rgb_frames_refuse_a_decodable_frame_over_the_pixel_budget(
    decodable_over_budget_video: Path, toolchain: VideoMeasurementToolchain
) -> None:
    def read_frames() -> None:
        with rgb_frames(
            decodable_over_budget_video,
            toolchain=toolchain,
            source_fps=float(_FRAME_RATE),
        ) as frames:
            list(frames)

    _assert_refused_before_allocating(_over_budget_pixels() * 3, read_frames)


def test_rgb_frames_refuse_an_over_budget_source_when_the_sample_is_scaled(
    decodable_over_budget_video: Path, toolchain: VideoMeasurementToolchain
) -> None:
    """Scaling the sample does not make the probed frame legal.

    ``mediapipe_hands`` passes ``long_edge_pixels`` through to this reader.
    The budget applies to the dimensions ffprobe reported, before the scale.
    """

    def read_frames() -> None:
        with rgb_frames(
            decodable_over_budget_video,
            toolchain=toolchain,
            frames_per_second=1.0,
            long_edge_pixels=64,
            source_fps=float(_FRAME_RATE),
        ) as frames:
            list(frames)

    _assert_refused_before_allocating(_over_budget_pixels() * 3, read_frames)


def test_rgb_frames_refuse_a_header_that_declares_an_over_budget_frame(
    lying_over_budget_header: Path, toolchain: VideoMeasurementToolchain
) -> None:
    def read_frames() -> None:
        with rgb_frames(
            lying_over_budget_header,
            toolchain=toolchain,
            source_fps=float(_FRAME_RATE),
        ) as frames:
            list(frames)

    _assert_refused_before_allocating(_over_budget_pixels() * 3, read_frames)


def test_stream_camera_motion_refuses_a_header_that_declares_an_over_budget_frame(
    lying_over_budget_header: Path, toolchain: VideoMeasurementToolchain
) -> None:
    pytest.importorskip("cv2", reason="camera-motion streaming needs the 'motion' extra")

    def read_motion() -> None:
        with stream_camera_motion(
            lying_over_budget_header,
            settings=CameraMotionStreamSettings(frames_per_second=float(_FRAME_RATE)),
            toolchain=toolchain,
        ) as observations:
            list(observations)

    _assert_refused_before_allocating(_over_budget_pixels(), read_motion)


def test_stream_camera_motion_refuses_a_decodable_frame_over_the_pixel_budget(
    decodable_over_budget_video: Path, toolchain: VideoMeasurementToolchain
) -> None:
    pytest.importorskip("cv2", reason="camera-motion streaming needs the 'motion' extra")

    def read_motion() -> None:
        with stream_camera_motion(
            decodable_over_budget_video,
            settings=CameraMotionStreamSettings(frames_per_second=float(_FRAME_RATE)),
            toolchain=toolchain,
        ) as observations:
            list(observations)

    _assert_refused_before_allocating(_over_budget_pixels(), read_motion)


def test_measure_camera_motion_refuses_a_header_that_declares_an_over_budget_frame(
    lying_over_budget_header: Path, toolchain: VideoMeasurementToolchain
) -> None:
    pytest.importorskip("cv2", reason="camera-motion measurement needs the 'motion' extra")

    def measure() -> None:
        measure_camera_motion(
            lying_over_budget_header,
            toolchain=toolchain,
            settings=CameraMotionSettings(frames_per_second=float(_FRAME_RATE)),
        )

    _assert_refused_before_allocating(_over_budget_pixels(), measure)


def test_measure_camera_motion_refuses_a_decodable_frame_over_the_pixel_budget(
    decodable_over_budget_video: Path, toolchain: VideoMeasurementToolchain
) -> None:
    pytest.importorskip("cv2", reason="camera-motion measurement needs the 'motion' extra")

    def measure() -> None:
        measure_camera_motion(
            decodable_over_budget_video,
            toolchain=toolchain,
            settings=CameraMotionSettings(frames_per_second=float(_FRAME_RATE)),
        )

    _assert_refused_before_allocating(_over_budget_pixels(), measure)


def test_camera_stability_refuses_an_over_budget_episode_after_remux(
    tmp_path: Path, toolchain: VideoMeasurementToolchain
) -> None:
    """The opt-in check remuxes in-band H.264, then uses the same raw reader.

    The episode is a canonical compressed-video file. Its access units are the
    input; nothing here goes through the importer's pixel check.
    """
    pytest.importorskip("cv2", reason="camera_stability needs the 'motion' extra")
    annex_b = tmp_path / "over-budget.h264"
    _write_encoded_video(annex_b, _OVER_BUDGET_WIDTH, _OVER_BUDGET_HEIGHT, frames=2)
    access_units = split_annex_b_stream(annex_b.read_bytes())
    assert len(access_units) == 2
    episode_path = tmp_path / "episode.mcap"
    start_ns = 1_700_000_000_000_000_000
    write_compressed_video_mcap(
        episode_path,
        [
            ("/camera", start_ns + index * 1_000_000_000 // _FRAME_RATE, unit.data)
            for index, unit in enumerate(access_units)
        ],
    )

    def measure() -> None:
        with Episode(episode_path, workdir=tmp_path / "work") as episode:
            remux = episode.video()
            _require_probed_dimensions(remux, _OVER_BUDGET_WIDTH, _OVER_BUDGET_HEIGHT)
            asyncio.run(camera_stability(episode))

    # ``toolchain`` is unused: the check resolves HFlow's own binaries.
    # Resolving it here still pins the test to a toolchain the suite can run.
    assert toolchain.ffprobe_executable.is_file()
    _assert_refused_before_allocating(_over_budget_pixels(), measure)


def test_raw_frame_stream_refuses_a_read_larger_than_the_pixel_budget(
    tmp_path: Path, toolchain: VideoMeasurementToolchain
) -> None:
    """The read size itself is capped, not only a probe that happens earlier.

    The file's own frames are inside the budget. The shape passed to the
    reader is not. Re-probing the file and then reading the oversized shape
    still fails this test, because the allocation is the value that was used.
    """
    legal = tmp_path / "legal.mp4"
    _write_encoded_video(legal, 160, 120, frames=1)
    oversized_shape = (_OVER_BUDGET_HEIGHT, _OVER_BUDGET_WIDTH)
    declared_frame_bytes = int(np.prod(oversized_shape, dtype=np.int64))

    def read_frames() -> None:
        with _raw_frame_stream(
            legal,
            toolchain=toolchain,
            filter_graph=LUMA_FRAME_FILTER_GRAPH,
            frame_shape=oversized_shape,
        ) as frames:
            list(frames)

    _assert_refused_before_allocating(declared_frame_bytes, read_frames)


def test_a_frame_at_the_pixel_budget_still_decodes(
    tmp_path: Path, toolchain: VideoMeasurementToolchain
) -> None:
    """The budget matches ``probe_video``: equality is allowed, one row over is not."""
    video = tmp_path / "exact-budget.mp4"
    _write_encoded_video(video, _OVER_BUDGET_WIDTH, _EXACT_BUDGET_HEIGHT, frames=1)
    with luma_frames(video, toolchain=toolchain) as frames:
        decoded = list(frames)
    assert len(decoded) == 1
    assert decoded[0].shape == (_EXACT_BUDGET_HEIGHT, _OVER_BUDGET_WIDTH)


def test_a_small_declared_header_still_decodes(
    tmp_path: Path, toolchain: VideoMeasurementToolchain
) -> None:
    width, height = 160, 120
    video = tmp_path / "legal.y4m"
    header = f"YUV4MPEG2 W{width} H{height} F{_FRAME_RATE}:1 Ip A1:1 Cmono\nFRAME\n"
    video.write_bytes(header.encode("ascii") + bytes(width * height))
    _require_probed_dimensions(video, width, height)
    with luma_frames(video, toolchain=toolchain) as frames:
        decoded = list(frames)
    assert len(decoded) == 1
    assert decoded[0].shape == (height, width)
