"""Measure one fixed-rate source window with a single decode, without an intermediate file.

``measure_video_window`` seeks the original source, applies the window's ``fps``
filter, and splits the decoded frames inside one FFmpeg process between the
selected frame-statistics, blur, and camera-shake measurements. Each branch applies
exactly the filters its file-level measurement applies, so results equal measuring
an uncompressed copy of the same window with ``measure_video_frame_statistics``,
``measure_video_blur``, and ``stream_camera_motion``. Unlike ``prepare_video_window``,
no lossy H.264 window is encoded and measured.

Unreadable and unsupported media are expected outcomes and are returned, matching
``prepare_video_window``. Process, timeout, and filesystem failures raise.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import IO

import numpy as np

from hflow._video_measurement_toolchain import resolved_video_measurement_toolchain
from hflow._video_measurements._frame_statistics import (
    FRAME_STATISTICS_DEFINITION_VERSION,
    FrameStatisticsProvenance,
    FrameStatisticsSettings,
    VideoFrameStatistics,
    _aggregate_frame_statistics_lines,
    _attach_provenance,
    _NoInstrumentFramesError,
    _validate_frame_statistics_filters,
    frame_statistics_filter_chain,
    frame_statistics_filter_graph,
)
from hflow._video_measurements._motion_stream import (
    CameraMotionStreamSettings,
    CameraShakeSettings,
    filter_camera_shake,
    iter_frame_motion,
)
from hflow._video_measurements._toolchain import VideoMeasurementToolchain
from hflow.blur import BlurSummary, summarize_blur_scores
from hflow.camera_motion import CameraShakeSummary, summarize_camera_shake
from hflow.ffmpeg._process import MediaCommandResult, MediaToolError, media_input_was_rejected
from hflow.media import (
    UnreadableVideo,
    UnsupportedVideo,
    VideoLimits,
    VideoProperties,
    VideoWindow,
    probe_video,
)

__all__ = [
    "IndependentVideoWindowMeasurements",
    "MeasurementFailure",
    "VideoWindowMeasurements",
    "WindowMeasurementSelection",
    "measure_video_window",
    "measure_video_window_independently",
]

_MAXIMUM_DIAGNOSTIC_BYTES = 65536
# Metadata destinations are FFmpeg filter option values; restricting the private
# temporary path avoids filter-graph escaping rules for ':', ',', ';', '[' and quotes.
_FILTER_SAFE_PATH = re.compile(r"^[A-Za-z0-9/._-]+$")
_Y4M_DIMENSIONS = re.compile(rb"\bW(?P<width>\d+) H(?P<height>\d+)\b")


@dataclass(frozen=True)
class WindowMeasurementSelection:
    """Which measurements one decode produces; unselected results are ``None``.

    ``camera_shake`` settings are paired with the window's frame rate, which is the
    constant cadence of the decoded frames.
    """

    frame_statistics: FrameStatisticsSettings | None = None
    blur: bool = False
    camera_shake: CameraShakeSettings | None = None

    def __post_init__(self) -> None:
        if self.frame_statistics is not None and not isinstance(
            self.frame_statistics, FrameStatisticsSettings
        ):
            raise ValueError("frame_statistics must be FrameStatisticsSettings or None")
        if type(self.blur) is not bool:
            raise ValueError("blur must be a bool")
        if self.camera_shake is not None and not isinstance(self.camera_shake, CameraShakeSettings):
            raise ValueError("camera_shake must be CameraShakeSettings or None")
        if self.frame_statistics is None and not self.blur and self.camera_shake is None:
            raise ValueError("select at least one window measurement")


@dataclass(frozen=True)
class VideoWindowMeasurements:
    """Results for the measured window, which may end before the requested one."""

    window: VideoWindow
    decoded_frame_count: int
    frame_statistics: VideoFrameStatistics | None
    blur: BlurSummary | None
    camera_shake: CameraShakeSummary | None


@dataclass(frozen=True)
class MeasurementFailure:
    """A selected branch failed after the shared source decode began.

    The exception type is retained for diagnosis without publishing raw process
    diagnostics or any model or source content.
    """

    error_type: str


@dataclass(frozen=True)
class IndependentVideoWindowMeasurements:
    """Branch outcomes from one decode; ``None`` means unselected."""

    window: VideoWindow
    decoded_frame_count: int | None
    frame_statistics: VideoFrameStatistics | MeasurementFailure | None
    blur: BlurSummary | MeasurementFailure | None
    camera_shake: CameraShakeSummary | MeasurementFailure | None


def measure_video_window_independently(
    source: Path,
    window: VideoWindow,
    selection: WindowMeasurementSelection,
    *,
    limits: VideoLimits = VideoLimits(),
    toolchain: VideoMeasurementToolchain | None = None,
) -> IndependentVideoWindowMeasurements | UnreadableVideo | UnsupportedVideo:
    """Keep usable branch measurements when another branch's calculation fails.

    A source probe, FFmpeg decode, timeout, or filesystem failure remains shared
    and raises or returns a shared media outcome. This API isolates failures in
    the Python motion calculation and in parsing each branch's output.
    """
    result = _measure_video_window(
        source, window, selection, limits=limits, toolchain=toolchain, isolate_failures=True
    )
    if isinstance(result, VideoWindowMeasurements):
        return IndependentVideoWindowMeasurements(
            result.window,
            result.decoded_frame_count,
            result.frame_statistics,
            result.blur,
            result.camera_shake,
        )
    return result


def measure_video_window(
    source: Path,
    window: VideoWindow,
    selection: WindowMeasurementSelection,
    *,
    limits: VideoLimits = VideoLimits(),
    toolchain: VideoMeasurementToolchain | None = None,
) -> VideoWindowMeasurements | UnreadableVideo | UnsupportedVideo:
    """Decode ``window`` from ``source`` once and compute every selected measurement.

    The window is clamped to the source end and judged unreadable when it decodes to
    less than half its duration (at most 0.5 seconds), as ``prepare_video_window`` does.
    FFmpeg's default display-rotation handling applies, so frames are measured upright.
    """
    result = _measure_video_window(
        source, window, selection, limits=limits, toolchain=toolchain, isolate_failures=False
    )
    assert not isinstance(result, IndependentVideoWindowMeasurements)
    return result


def _measure_video_window(
    source: Path,
    window: VideoWindow,
    selection: WindowMeasurementSelection,
    *,
    limits: VideoLimits,
    toolchain: VideoMeasurementToolchain | None,
    isolate_failures: bool,
) -> (
    VideoWindowMeasurements
    | IndependentVideoWindowMeasurements
    | UnreadableVideo
    | UnsupportedVideo
):
    resolved_toolchain = (
        toolchain if toolchain is not None else resolved_video_measurement_toolchain()
    )
    inspection = probe_video(
        source, limits=limits, executable=resolved_toolchain.ffprobe_executable
    )
    if not isinstance(inspection, VideoProperties):
        return inspection
    if (
        window.frames_per_second > limits.maximum_frames_per_second
        or window.duration_seconds > limits.maximum_duration_seconds
    ):
        return UnsupportedVideo()
    remaining_source_seconds = inspection.duration_seconds - Decimal(str(window.start_seconds))
    if remaining_source_seconds <= 0:
        return UnreadableVideo()
    measured_window = VideoWindow(
        start_seconds=window.start_seconds,
        duration_seconds=min(window.duration_seconds, float(remaining_source_seconds)),
        frames_per_second=window.frames_per_second,
    )
    if selection.frame_statistics is not None:
        _validate_frame_statistics_filters(resolved_toolchain)

    with tempfile.TemporaryDirectory(prefix="hflow-window-") as directory:
        working_directory = Path(directory).resolve()
        if not _FILTER_SAFE_PATH.match(str(working_directory)):
            raise MediaToolError("temporary directory path is not safe for an FFmpeg filter")
        statistics_path = working_directory / "frame-statistics.txt"
        blur_path = working_directory / "blur.txt"
        diagnostics_path = working_directory / "diagnostics.txt"
        filter_graph, output_arguments = _shared_decode_graph(
            selection,
            frames_per_second=measured_window.frames_per_second,
            statistics_path=statistics_path,
            blur_path=blur_path,
        )
        command = [
            str(resolved_toolchain.ffmpeg_executable),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-protocol_whitelist",
            "file",
            "-ss",
            f"{measured_window.start_seconds:.3f}",
            "-t",
            f"{measured_window.duration_seconds:.3f}",
            "-i",
            str(source.resolve()),
            "-filter_complex",
            filter_graph,
            *output_arguments,
        ]
        with diagnostics_path.open("wb") as diagnostics_output:
            decoding_process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE
                if selection.camera_shake is not None
                else subprocess.DEVNULL,
                stderr=diagnostics_output,
            )
            timed_out = threading.Event()

            def stop_after_timeout() -> None:
                timed_out.set()
                decoding_process.kill()

            watchdog = threading.Timer(limits.timeout_seconds, stop_after_timeout)
            watchdog.start()
            camera_shake: CameraShakeSummary | MeasurementFailure | None = None
            luma_frame_count: int | None = None
            try:
                if selection.camera_shake is not None:
                    assert decoding_process.stdout is not None
                    luma_frames = _Y4mLumaFrames(decoding_process.stdout)
                    try:
                        camera_shake = summarize_camera_shake(
                            filter_camera_shake(
                                iter_frame_motion(
                                    luma_frames,
                                    settings=CameraMotionStreamSettings(
                                        frames_per_second=measured_window.frames_per_second
                                    ),
                                ),
                                settings=selection.camera_shake,
                            )
                        )
                        luma_frame_count = luma_frames.frame_count
                    except Exception as error:
                        if not isolate_failures or isinstance(
                            error, (OSError, MemoryError, TimeoutError)
                        ):
                            raise
                        camera_shake = MeasurementFailure(type(error).__name__)
                        # FFmpeg still has other output branches to finish. Drain
                        # the pipe so their measurements can complete.
                        while decoding_process.stdout.read(65536):
                            pass
                return_code = decoding_process.wait()
            except BaseException:
                decoding_process.kill()
                decoding_process.wait()
                raise
            finally:
                watchdog.cancel()
                if decoding_process.stdout is not None:
                    decoding_process.stdout.close()
        if timed_out.is_set():
            raise MediaToolError("window measurement timed out")
        if return_code != 0:
            with diagnostics_path.open("rb") as diagnostics_input:
                diagnostics = diagnostics_input.read(_MAXIMUM_DIAGNOSTIC_BYTES)
            if media_input_was_rejected(
                MediaCommandResult(returncode=return_code, stdout=b"", stderr=diagnostics)
            ):
                return UnreadableVideo()

        frame_statistics: VideoFrameStatistics | MeasurementFailure | None = None
        if selection.frame_statistics is not None:
            try:
                with statistics_path.open(encoding="utf-8") as statistics_lines:
                    aggregate = _aggregate_frame_statistics_lines(
                        statistics_lines, selection.frame_statistics
                    )
            except _NoInstrumentFramesError:
                # A window that decodes no frames is unreadable, not a parse failure.
                return UnreadableVideo()
            except Exception as error:
                if not isolate_failures or isinstance(error, (OSError, MemoryError, TimeoutError)):
                    raise
                frame_statistics = MeasurementFailure(type(error).__name__)
            else:
                frame_statistics = _attach_provenance(
                    aggregate,
                    FrameStatisticsProvenance(
                        measurement_definition_version=FRAME_STATISTICS_DEFINITION_VERSION,
                        ffmpeg_version=resolved_toolchain.ffmpeg_version,
                        filter_graph=frame_statistics_filter_graph(selection.frame_statistics),
                        settings=selection.frame_statistics,
                    ),
                )
        blur: BlurSummary | MeasurementFailure | None = None
        if selection.blur:
            try:
                blur = _read_blur_summary(blur_path)
            except Exception as error:
                if not isolate_failures or isinstance(error, (OSError, MemoryError, TimeoutError)):
                    raise
                blur = MeasurementFailure(type(error).__name__)

    decoded_frame_counts = {
        count
        for count in (
            frame_statistics.decoded_frame_count
            if isinstance(frame_statistics, VideoFrameStatistics)
            else None,
            blur.frame_count if isinstance(blur, BlurSummary) else None,
            luma_frame_count,
        )
        if count is not None
    }
    if len(decoded_frame_counts) > 1 or (not isolate_failures and not decoded_frame_counts):
        raise MediaToolError("window measurements observed different frame counts")
    decoded_frame_count = decoded_frame_counts.pop() if decoded_frame_counts else None
    if (
        decoded_frame_count is not None
        and decoded_frame_count / measured_window.frames_per_second
        < min(0.5, measured_window.duration_seconds * 0.5)
    ):
        return UnreadableVideo()
    if isolate_failures:
        return IndependentVideoWindowMeasurements(
            window=measured_window,
            decoded_frame_count=decoded_frame_count,
            frame_statistics=frame_statistics,
            blur=blur,
            camera_shake=camera_shake,
        )
    assert decoded_frame_count is not None
    assert not isinstance(frame_statistics, MeasurementFailure)
    assert not isinstance(blur, MeasurementFailure)
    assert not isinstance(camera_shake, MeasurementFailure)
    return VideoWindowMeasurements(
        window=measured_window,
        decoded_frame_count=decoded_frame_count,
        frame_statistics=frame_statistics,
        blur=blur,
        camera_shake=camera_shake,
    )


def _shared_decode_graph(
    selection: WindowMeasurementSelection,
    *,
    frames_per_second: float,
    statistics_path: Path,
    blur_path: Path,
) -> tuple[str, list[str]]:
    """Return one decode's filter graph and its per-measurement output arguments."""
    branches: list[tuple[str, str, list[str]]] = []
    if selection.frame_statistics is not None:
        branches.append(
            (
                "frame_statistics",
                frame_statistics_filter_chain(
                    selection.frame_statistics, metadata_destination=str(statistics_path)
                ),
                ["-f", "null", "-"],
            )
        )
    if selection.blur:
        branches.append(
            (
                "blur",
                f"blurdetect,metadata=mode=print:key=lavfi.blur:file={blur_path}",
                ["-f", "null", "-"],
            )
        )
    if selection.camera_shake is not None:
        # An explicit scale converts inside this branch, matching luma_frames'
        # format=gray. A bare format=gray would let FFmpeg negotiate grey frames
        # for split itself, silently removing chroma from every other branch.
        branches.append(("luma", "scale,format=pix_fmts=gray", ["-f", "yuv4mpegpipe", "pipe:1"]))
    split_outputs = "".join(f"[{name}_input]" for name, _, _ in branches)
    filter_graph = ";".join(
        (
            f"[0:v:0]fps={frames_per_second:g},split={len(branches)}{split_outputs}",
            *(f"[{name}_input]{chain}[{name}_output]" for name, chain, _ in branches),
        )
    )
    output_arguments = [
        argument
        for name, _, muxer_arguments in branches
        for argument in ("-map", f"[{name}_output]", *muxer_arguments)
    ]
    return filter_graph, output_arguments


class _Y4mLumaFrames(Iterator[np.ndarray]):
    """Read grey YUV4MPEG2 frames; the header carries the rotated output size."""

    def __init__(self, stream: IO[bytes]) -> None:
        self._stream = stream
        self._frame_shape: tuple[int, int] | None = None
        self.frame_count = 0

    def __next__(self) -> np.ndarray:
        if self._frame_shape is None:
            header = self._stream.readline()
            if not header:
                raise StopIteration
            dimensions = _Y4M_DIMENSIONS.search(header)
            if not header.startswith(b"YUV4MPEG2 ") or dimensions is None:
                raise MediaToolError("window decode produced an invalid luma stream header")
            self._frame_shape = (int(dimensions["height"]), int(dimensions["width"]))
        frame_header = self._stream.readline()
        if not frame_header:
            raise StopIteration
        if not frame_header.startswith(b"FRAME"):
            raise MediaToolError("window decode produced an invalid luma frame header")
        frame_byte_count = self._frame_shape[0] * self._frame_shape[1]
        frame_payload = self._stream.read(frame_byte_count)
        if len(frame_payload) != frame_byte_count:
            raise MediaToolError("window decode produced a truncated luma frame")
        self.frame_count += 1
        return np.frombuffer(frame_payload, dtype=np.uint8).reshape(self._frame_shape)


def _read_blur_summary(blur_path: Path) -> BlurSummary:
    metadata_lines = blur_path.read_bytes().splitlines()
    summary = summarize_blur_scores(
        float(line.removeprefix(b"lavfi.blur="))
        for line in metadata_lines
        if line.startswith(b"lavfi.blur=")
    )
    emitted_frame_count = sum(line.startswith(b"frame:") for line in metadata_lines)
    if summary.frame_count != emitted_frame_count:
        raise MediaToolError("blur measurement returned incomplete frame scores")
    return summary
