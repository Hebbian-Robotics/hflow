"""LeRobot import must not follow playlist segments to other local files.

The dataset publisher controls ``video_path`` and the bytes downloaded for it.
``import_lerobot_dataset`` passes that local path to ffmpeg and ffprobe. These
tests place a decodable video outside the dataset cache and name it from an
HLS playlist. They fail while either tool opens that file, including when the
import later raises, and they pass once the playlist is refused before any
segment is opened.

The ``hflow import lerobot`` command calls ``import_lerobot_dataset`` and does
not open media itself (``src/hflow/cli.py``). Concat scripts are not covered
here: the pinned ffmpeg build rejects unsafe concat names before opening them.
"""

from __future__ import annotations

import ctypes
import os
import select
import subprocess
import threading
from pathlib import Path

import pytest
from lerobot_test_helpers import (
    CorpusEpisodeRow,
    stub_hub_download,
    stub_hub_repo_info,
    write_v3_corpus,
)

import hflow.importers.lerobot as lerobot_importer
from hflow.ffmpeg import ffmpeg_path

_CAMERA_KEY = "observation.images.cam"
_REVISION = "abc"
_FPS = 10
_FRAMES = 3
_IN_ACCESS = 0x00000001
_IN_OPEN = 0x00000020
_IN_NONBLOCK = 0x00000800


class _InotifyEvent(ctypes.Structure):
    _fields_ = (
        ("wd", ctypes.c_int),
        ("mask", ctypes.c_uint32),
        ("cookie", ctypes.c_uint32),
        ("len", ctypes.c_uint32),
    )


class _OpenWatch:
    """Count ``open`` and ``read`` events on one file while the body runs."""

    def __init__(self, path: Path) -> None:
        self.events = 0
        self._stop = threading.Event()
        self._libc = ctypes.CDLL("libc.so.6", use_errno=True)
        self._fd = self._libc.inotify_init1(_IN_NONBLOCK)
        if self._fd < 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number), str(path))
        watch = self._libc.inotify_add_watch(self._fd, os.fsencode(path), _IN_OPEN | _IN_ACCESS)
        if watch < 0:
            error_number = ctypes.get_errno()
            os.close(self._fd)
            raise OSError(error_number, os.strerror(error_number), str(path))
        self._watch = watch
        self._thread = threading.Thread(target=self._collect, daemon=True)

    def __enter__(self) -> _OpenWatch:
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self._drain()
        self._libc.inotify_rm_watch(self._fd, self._watch)
        os.close(self._fd)

    def _collect(self) -> None:
        while not self._stop.is_set():
            ready, _, _ = select.select([self._fd], [], [], 0.05)
            if ready:
                self._drain()

    def _drain(self) -> None:
        header = ctypes.sizeof(_InotifyEvent)
        while True:
            try:
                raw = os.read(self._fd, 4096)
            except BlockingIOError:
                return
            if not raw:
                return
            offset = 0
            while offset + header <= len(raw):
                event = _InotifyEvent.from_buffer_copy(raw[offset : offset + header])
                offset += header + event.len
                if event.mask & (_IN_OPEN | _IN_ACCESS):
                    self.events += 1


def _relative_posix(target: Path, start_directory: Path) -> str:
    target_parts = target.resolve().parts
    start_parts = start_directory.resolve().parts
    common = 0
    for left, right in zip(target_parts, start_parts, strict=False):
        if left != right:
            break
        common += 1
    upward = [".."] * (len(start_parts) - common)
    return "/".join([*upward, *target_parts[common:]])


def _write_solid_video(path: Path) -> None:
    subprocess.run(
        [
            str(ffmpeg_path()),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=red:s=32x32:r={_FPS}:d={_FRAMES / _FPS}",
            "-frames:v",
            str(_FRAMES),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-bf",
            "0",
            str(path),
        ],
        check=True,
    )


def _playlist(segment: str) -> str:
    return (
        "#EXTM3U\n"
        "#EXT-X-VERSION:3\n"
        "#EXT-X-TARGETDURATION:1\n"
        f"#EXTINF:{_FRAMES / _FPS},\n"
        f"{segment}\n"
        "#EXT-X-ENDLIST\n"
    )


def _import_playlist(
    monkeypatch: pytest.MonkeyPatch,
    output_dir: Path,
    corpus_root: Path,
    playlist_text: str,
    *,
    windowed: bool,
) -> None:
    info = {
        "fps": _FPS,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.m3u8",
        "features": {
            "action": {"dtype": "float32", "shape": [6]},
            "observation.state": {"dtype": "float32", "shape": [6]},
            _CAMERA_KEY: {"dtype": "video", "shape": [32, 32, 3]},
        },
        "robot_type": "so101",
    }
    write_v3_corpus(
        corpus_root,
        info=info,
        episode_rows=[
            CorpusEpisodeRow(
                episode_index=0,
                length=_FRAMES,
                dataset_from_index=0,
                video_to_timestamp=0.3 if windowed else 0.0,
                tasks=("push",),
            )
        ],
        camera_keys=(_CAMERA_KEY,),
        timestamps_as_double=True,
    )
    stub_hub_repo_info(monkeypatch, resolved_sha=_REVISION)
    monkeypatch.setattr(
        lerobot_importer,
        "_hf_episode_metadata_files",
        lambda *_args, **_kwargs: ["meta/episodes/chunk-000/file-000.parquet"],
    )

    def supply_file(filename: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if filename.endswith(".m3u8"):
            destination.write_text(playlist_text, encoding="utf-8")
        elif "meta/episodes" in filename:
            source = corpus_root / "meta/episodes/chunk-000/file-000.parquet"
            destination.write_bytes(source.read_bytes())
        elif filename.endswith("info.json"):
            destination.write_text((corpus_root / "meta" / "info.json").read_text(encoding="utf-8"))
        else:
            source = corpus_root / "data/chunk-000/file-000.parquet"
            destination.write_bytes(source.read_bytes())

    stub_hub_download(monkeypatch, supply_file)
    lerobot_importer.import_lerobot_dataset(
        dataset_repo="attacker/dataset",
        revision="main",
        output_dir=output_dir,
        episode_index=0,
        camera_keys=(_CAMERA_KEY,),
    )


@pytest.mark.parametrize("segment_form", ["file-url", "relative"])
@pytest.mark.parametrize("windowed", [False, True])
def test_lerobot_import_does_not_open_playlist_segments_outside_the_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    segment_form: str,
    windowed: bool,
) -> None:
    """A publisher-supplied HLS playlist must not open or publish other files.

    ``windowed`` is the episode path that seeks before ``-i`` and rebases
    packet times. The other path transcodes the whole file and probes it.
    ``file-url`` and ``relative`` are the two segment spellings ffmpeg follows
    for a local playlist. The outside file is watched for the whole import, so
    a refusal that still opens the segment fails the test.
    """
    outside_video = tmp_path / "outside-camera.mp4"
    _write_solid_video(outside_video)
    output_dir = tmp_path / "output"
    playlist_path = (
        output_dir
        / "_lerobot_cache"
        / _REVISION
        / "videos"
        / _CAMERA_KEY
        / "chunk-000"
        / "file-000.m3u8"
    )
    if segment_form == "file-url":
        segment = outside_video.resolve().as_uri()
    else:
        segment = _relative_posix(outside_video, playlist_path.parent)
        assert segment.startswith("../")

    refused = False
    with _OpenWatch(outside_video) as watch:
        try:
            _import_playlist(
                monkeypatch,
                output_dir,
                tmp_path / "corpus",
                _playlist(segment),
                windowed=windowed,
            )
        except (ValueError, RuntimeError, OSError):
            refused = True

    assert watch.events == 0, "import opened a playlist segment outside the downloaded file"
    assert refused, "import published a playlist instead of refusing it"
    assert not (output_dir / "landing").exists()
    assert not (output_dir / "prepared-manifest.json").exists()
