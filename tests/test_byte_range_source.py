"""Keyframe sampling reads only what it needs, locally and through a byte-range reader."""

import hashlib
import http.client
import math
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from fractions import Fraction
from pathlib import Path

import pytest

from hflow import SourceFrameSampling, SourceSamplingMode, SourceWindow, sample_source_frames
from hflow.byte_range_source import LoopbackVideoSource, serve_byte_ranges
from hflow.ffmpeg import ffmpeg_path, ffprobe_path
from hflow.media import VideoProperties, probe_video
from hflow.source_sampling import SourceFrameResize


class FileRangeReader:
    """Reads one local file by range and counts the bytes it returned."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.bytes_read = 0

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size

    def read_range(self, start: int, stop: int) -> bytes:
        with self.path.open("rb") as source_file:
            source_file.seek(start)
            data = source_file.read(stop - start)
        self.bytes_read += len(data)
        return data


class StorageUnavailable(Exception):
    """Stands in for a storage or credential failure raised by a caller's reader."""


class FailingRangeReader(FileRangeReader):
    """Serves bytes before ``failing_from_byte``, then fails like an expired credential."""

    def __init__(self, path: Path, *, failing_from_byte: int) -> None:
        super().__init__(path)
        self.failing_from_byte = failing_from_byte

    def read_range(self, start: int, stop: int) -> bytes:
        if stop > self.failing_from_byte:
            raise StorageUnavailable("the object store refused the request")
        return super().read_range(start, stop)


def _encode(path: Path, *arguments: str) -> Path:
    subprocess.run(
        [str(ffmpeg_path()), "-v", "error", "-nostdin", "-y", *arguments, str(path)],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return path


@pytest.fixture(scope="module")
def recordings(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    directory = tmp_path_factory.mktemp("recordings")
    x264 = ("-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p")
    fixed = _encode(
        directory / "fixed_gop_b_frames.mp4",
        "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=30:duration=12",
        *x264, "-g", "45", "-keyint_min", "45", "-sc_threshold", "0", "-bf", "3",
    )  # fmt: skip
    # Hard cuts between sources give scene-cut keyframes at irregular times.
    scene_cuts = _encode(
        directory / "scene_cut_keyframes.mov",
        "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=25:duration=4",
        "-f", "lavfi", "-i", "mandelbrot=size=160x120:rate=25",
        "-f", "lavfi", "-i", "smptebars=size=160x120:rate=25:duration=5",
        "-filter_complex",
        "[1:v]trim=duration=3.3,setpts=PTS-STARTPTS[m];[0:v][m][2:v]concat=n=3:v=1:a=0[out]",
        "-map", "[out]", *x264, "-g", "250",
    )  # fmt: skip
    shifted = _encode(
        directory / "nonzero_start.mp4",
        "-i", str(fixed), "-c", "copy", "-output_ts_offset", "5.033333",
    )  # fmt: skip
    matroska = _encode(directory / "matroska.mkv", "-i", str(fixed), "-c", "copy")
    # The last bin starts 5.5 s after the keyframe before it and 4.5 s before its own.
    late_keyframe = _encode(
        directory / "late_keyframe.mp4",
        "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=10:duration=20",
        *x264, "-g", "1000", "-sc_threshold", "0", "-force_key_frames", "0,10,19.5",
    )  # fmt: skip
    return {
        "late_keyframe": late_keyframe,
        "fixed": fixed,
        "scene_cuts": scene_cuts,
        "shifted": shifted,
        "matroska": matroska,
    }


def _single_pass_keyframes(
    source_path: Path, output_directory: Path, window: SourceWindow, settings: SourceFrameSampling
) -> list[tuple[Fraction, str]]:
    """The one-pass keyframe extraction the sampler used before it seeked per bin."""
    time_base = Fraction(
        subprocess.run(
            [
                str(ffprobe_path()), "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=time_base", "-of", "default=noprint_wrappers=1:nokey=1",
                str(source_path),
            ],
            check=True, capture_output=True, timeout=60,
        ).stdout.decode().strip()
    )  # fmt: skip
    interval_millis = Fraction(window.duration_millis, settings.maximum_frames)
    bin_scale = time_base * 1000 / interval_millis
    bin_offset = Fraction(window.start_millis) / interval_millis
    denominator = math.lcm(bin_scale.denominator, bin_offset.denominator)
    multiplier = bin_scale.numerator * (denominator // bin_scale.denominator)
    offset = bin_offset.numerator * (denominator // bin_offset.denominator)
    first_tick = math.ceil(Fraction(window.start_millis, 1000) / time_base)
    end_tick = math.ceil(Fraction(window.end_millis, 1000) / time_base)
    current_bin = f"floor((pts*{multiplier}-{offset})/{denominator})"
    previous_bin = f"floor((prev_selected_pts*{multiplier}-{offset})/{denominator})"
    resize = (
        f"scale={settings.width}:{settings.height}:force_original_aspect_ratio=decrease"
        f":flags={settings.scaling_algorithm}"
    )
    if settings.resize is SourceFrameResize.PAD:
        resize += f",pad={settings.width}:{settings.height}:(ow-iw)/2:(oh-ih)/2:black"
    output_directory.mkdir()
    completed = subprocess.run(
        [
            str(ffmpeg_path()), "-hide_banner", "-loglevel", "info", "-nostats", "-nostdin",
            "-xerror", "-protocol_whitelist", "file", "-threads", "2", "-skip_frame", "nokey",
            "-copyts", "-start_at_zero", "-ss", f"{window.start_millis / 1000:.3f}",
            "-t", f"{window.duration_millis / 1000:.3f}", "-i", str(source_path),
            "-map", "0:v:0", "-an", "-map_metadata", "-1",
            "-vf",
            f"select=gte(pts\\,{first_tick})*lt(pts\\,{end_tick})*"
            f"lt(selected_n\\,{settings.maximum_frames})*(isnan(prev_selected_t)+"
            f"gt({current_bin}\\,{previous_bin})),{resize},showinfo",
            "-filter_threads", "1", "-frames:v", str(settings.maximum_frames),
            "-fps_mode", "passthrough", "-pix_fmt", "yuvj420p", "-c:v", "mjpeg",
            "-q:v", str(settings.jpeg_quality), "-threads", "2", "-f", "image2",
            str(output_directory / "frame_%06d.jpg"),
        ],
        check=True, capture_output=True, timeout=120,
    )  # fmt: skip
    timestamps = [
        int(match.group(1)) * time_base
        for match in re.finditer(
            rb"\[Parsed_showinfo_[^\]]+\].*\bn:\s*\d+\s+pts:\s*(-?\d+)\s+pts_time:",
            completed.stderr,
        )
    ]
    frames = sorted(output_directory.glob("frame_*.jpg"))
    assert len(frames) == len(timestamps)
    return [
        (timestamp, hashlib.sha256(frame.read_bytes()).hexdigest())
        for timestamp, frame in zip(timestamps, frames, strict=True)
    ]


def _sampled(
    samples_path: Path,
    source: Path | LoopbackVideoSource,
    window: SourceWindow,
    settings: SourceFrameSampling,
) -> list[tuple[Fraction, str]]:
    samples = sample_source_frames(source, samples_path, window=window, settings=settings)
    return [
        (frame.timestamp_seconds, hashlib.sha256(frame.path.read_bytes()).hexdigest())
        for frame in samples.frames
    ]


@pytest.mark.parametrize(
    ("recording", "window", "maximum_frames", "resize"),
    [
        ("fixed", SourceWindow(0, 12_000), 3, SourceFrameResize.FIT),
        ("fixed", SourceWindow(1_700, 10_900), 16, SourceFrameResize.PAD),
        ("fixed", SourceWindow(250, 11_750), 1, SourceFrameResize.PAD),
        ("scene_cuts", SourceWindow(0, 12_300), 3, SourceFrameResize.FIT),
        ("scene_cuts", SourceWindow(3_100, 9_000), 16, SourceFrameResize.PAD),
        ("shifted", SourceWindow(0, 12_000), 3, SourceFrameResize.FIT),
        ("shifted", SourceWindow(2_333, 8_001), 4, SourceFrameResize.PAD),
        ("matroska", SourceWindow(0, 12_000), 3, SourceFrameResize.FIT),
        ("late_keyframe", SourceWindow(0, 20_000), 4, SourceFrameResize.PAD),
    ],
)
def test_seeking_selects_the_same_keyframes_and_pixels_as_one_pass(
    recordings: dict[str, Path],
    tmp_path: Path,
    recording: str,
    window: SourceWindow,
    maximum_frames: int,
    resize: SourceFrameResize,
) -> None:
    settings = SourceFrameSampling(
        mode=SourceSamplingMode.KEYFRAMES,
        maximum_frames=maximum_frames,
        width=320,
        height=240,
        resize=resize,
        scaling_algorithm="bicubic" if resize is SourceFrameResize.FIT else "lanczos",
        jpeg_quality=2 if resize is SourceFrameResize.FIT else 5,
    )
    source_path = recordings[recording]
    expected = _single_pass_keyframes(source_path, tmp_path / "one-pass", window, settings)
    assert expected
    assert _sampled(tmp_path / "seeking", source_path, window, settings) == expected


@pytest.mark.parametrize("recording", ["fixed", "scene_cuts", "shifted", "matroska"])
def test_a_byte_range_source_probes_and_samples_like_the_local_file(
    recordings: dict[str, Path], tmp_path: Path, recording: str
) -> None:
    source_path = recordings[recording]
    local_properties = probe_video(source_path)
    assert isinstance(local_properties, VideoProperties)
    window = SourceWindow(0, local_properties.duration_millis)
    settings = SourceFrameSampling(
        mode=SourceSamplingMode.KEYFRAMES_FIRST,
        maximum_frames=3,
        maximum_window_millis=window.duration_millis,
        width=320,
        height=240,
        resize=SourceFrameResize.FIT,
        scaling_algorithm="bicubic",
        jpeg_quality=2,
    )
    expected = _sampled(tmp_path / "local", source_path, window, settings)
    reader = FileRangeReader(source_path)
    with serve_byte_ranges(reader) as source:
        assert probe_video(source) == local_properties
        assert _sampled(tmp_path / "remote", source, window, settings) == expected
        assert source.bytes_fetched == reader.bytes_read


def test_keyframe_sampling_reads_a_fraction_of_a_long_recording(tmp_path: Path) -> None:
    source_path = _encode(
        tmp_path / "long.mp4",
        "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=90,noise=alls=12:allf=t",
        "-c:v", "libx264", "-threads", "1", "-preset", "ultrafast", "-g", "60",
        "-pix_fmt", "yuv420p",
    )  # fmt: skip
    reader = FileRangeReader(source_path)
    with serve_byte_ranges(reader) as source:
        properties = probe_video(source)
        assert isinstance(properties, VideoProperties)
        samples = sample_source_frames(
            source,
            tmp_path / "samples",
            window=SourceWindow(0, properties.duration_millis),
            settings=SourceFrameSampling(
                mode=SourceSamplingMode.KEYFRAMES,
                maximum_frames=3,
                maximum_window_millis=properties.duration_millis,
            ),
        )
    assert [frame.timestamp_seconds for frame in samples.frames] == [0, 30, 60]
    assert reader.bytes_read < source_path.stat().st_size // 10


def test_a_reader_failure_is_raised_as_itself_not_as_unreadable_media(
    recordings: dict[str, Path], tmp_path: Path
) -> None:
    source_path = recordings["fixed"]
    with (
        serve_byte_ranges(FailingRangeReader(source_path, failing_from_byte=0)) as source,
        pytest.raises(StorageUnavailable),
    ):
        probe_video(source)
    # The first block succeeds, so the failure lands mid-response.
    block_bytes = 16 * 1024
    with (
        serve_byte_ranges(
            FailingRangeReader(source_path, failing_from_byte=block_bytes), block_bytes=block_bytes
        ) as source,
        pytest.raises(StorageUnavailable),
    ):
        sample_source_frames(
            source,
            tmp_path / "samples",
            window=SourceWindow(0, 12_000),
            settings=SourceFrameSampling(mode=SourceSamplingMode.KEYFRAMES, maximum_frames=3),
        )
    assert not (tmp_path / "samples").exists()


def test_a_configured_http_proxy_does_not_capture_loopback_reads(
    recordings: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("http_proxy", "http://192.0.2.1:9")
    with serve_byte_ranges(FileRangeReader(recordings["fixed"])) as source:
        assert isinstance(probe_video(source), VideoProperties)


def test_the_loopback_server_serves_only_its_own_path(recordings: dict[str, Path]) -> None:
    with serve_byte_ranges(FileRangeReader(recordings["fixed"])) as source:
        scheme_and_host = source.url.rsplit("/", 1)[0]
        with pytest.raises(urllib.error.HTTPError) as rejected:
            urllib.request.urlopen(scheme_and_host + "/another-object", timeout=10)
        assert rejected.value.code == 404
        request = urllib.request.Request(source.url, headers={"Range": "bytes=4-11"})
        with urllib.request.urlopen(request, timeout=10) as response:
            assert response.status == 206
            assert response.read() == recordings["fixed"].read_bytes()[4:12]


class SlowRangeReader(FileRangeReader):
    """Takes a while per range and records how many reads are still running."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.reads_in_progress = 0
        self.read_started = threading.Event()

    def read_range(self, start: int, stop: int) -> bytes:
        self.reads_in_progress += 1
        self.read_started.set()
        try:
            time.sleep(0.5)
            return super().read_range(start, stop)
        finally:
            self.reads_in_progress -= 1


def test_leaving_the_block_waits_for_reads_in_progress(recordings: dict[str, Path]) -> None:
    reader = SlowRangeReader(recordings["fixed"])
    with serve_byte_ranges(reader, block_bytes=16 * 1024) as source:
        host_and_port, path = source.url.removeprefix("http://").split("/", 1)
        connection = http.client.HTTPConnection(host_and_port, timeout=10)
        connection.request("GET", "/" + path, headers={"Range": "bytes=0-"})
        assert reader.read_started.wait(timeout=10)
        # Abandon the response mid-body, as FFmpeg does once it has what it needs.
        connection.close()
    assert reader.reads_in_progress == 0
