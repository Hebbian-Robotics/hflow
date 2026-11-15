"""FFmpeg reads interactive commands from standard input (``q`` stops it with exit status 0)."""

import os
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from episode_test_helpers import synthesize_canonical_episode

import hflow.importers.lerobot as lerobot_importer
from hflow._video_measurement_toolchain import resolved_video_measurement_toolchain
from hflow._video_measurements._frame_statistics import measure_video_frame_statistics
from hflow.episode import Episode
from hflow.ffmpeg import contact_sheet
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode

_STANDARD_INPUT_DESCRIPTOR = 0
_CAMERA_TOPIC = "/wrist_cam/compressed"
_SPEC = SyntheticEpisodeSpec(
    duration_s=10.0,
    cameras=("wrist_cam",),
    image_hz=30.0,
    image_width=640,
    image_height=480,
    black_segment=None,
    timestamp_offset_segment=None,
)
_SOURCE_FRAME_COUNT = 300


@pytest.fixture(scope="module")
def canonical_episode(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return synthesize_canonical_episode(tmp_path_factory.mktemp("standard-input"), _SPEC)


@pytest.fixture
def consumed_quit_command_count(tmp_path: Path) -> Iterator[Callable[[], int]]:
    quit_commands_path = tmp_path / "standard-input"
    quit_commands_path.write_bytes(b"q" * 4096)
    quit_commands = os.open(quit_commands_path, os.O_RDONLY)
    original_standard_input = os.dup(_STANDARD_INPUT_DESCRIPTOR)
    os.dup2(quit_commands, _STANDARD_INPUT_DESCRIPTOR)
    try:
        yield lambda: os.lseek(quit_commands, 0, os.SEEK_CUR)
    finally:
        os.dup2(original_standard_input, _STANDARD_INPUT_DESCRIPTOR)
        os.close(original_standard_input)
        os.close(quit_commands)


@pytest.mark.requires_system_ffmpeg
def test_frame_extraction_is_complete_and_leaves_standard_input_unread(
    canonical_episode: Path, tmp_path: Path, consumed_quit_command_count: Callable[[], int]
) -> None:
    with Episode(canonical_episode, workdir=tmp_path / "workdir") as episode:
        sampled_frames = episode.frames(_CAMERA_TOPIC, fps=10.0)
        indexed_frames = episode.frames_at_indices(
            _CAMERA_TOPIC, frame_indices=[0, _SOURCE_FRAME_COUNT - 1]
        )
        sheet = contact_sheet(sampled_frames, tmp_path / "sheet.jpg")

    assert len(sampled_frames) == 100
    assert sampled_frames[-1].log_time_ns == _SPEC.start_time_ns + 9_900_000_000
    assert len(indexed_frames) == 2
    assert sheet.path.stat().st_size > 0
    assert consumed_quit_command_count() == 0


@pytest.mark.requires_system_ffmpeg
def test_frame_statistics_cover_the_whole_video_and_leave_standard_input_unread(
    canonical_episode: Path, tmp_path: Path, consumed_quit_command_count: Callable[[], int]
) -> None:
    with Episode(canonical_episode, workdir=tmp_path / "workdir") as episode:
        statistics = measure_video_frame_statistics(
            episode.video(_CAMERA_TOPIC), toolchain=resolved_video_measurement_toolchain()
        )

    assert statistics.decoded_frame_count == _SOURCE_FRAME_COUNT
    assert consumed_quit_command_count() == 0


@pytest.mark.requires_system_ffmpeg
def test_synthetic_episode_renders_and_leaves_standard_input_unread(
    tmp_path: Path, consumed_quit_command_count: Callable[[], int]
) -> None:
    source = synthesize_episode(tmp_path / "episode.mcap", _SPEC)

    assert source.stat().st_size > 0
    assert consumed_quit_command_count() == 0


@pytest.mark.requires_system_ffmpeg
def test_lerobot_transcode_keeps_every_frame_and_leaves_standard_input_unread(
    canonical_episode: Path, tmp_path: Path, consumed_quit_command_count: Callable[[], int]
) -> None:
    with Episode(canonical_episode, workdir=tmp_path / "workdir") as episode:
        access_units = lerobot_importer._transcode_mp4_to_h264(
            episode.video(_CAMERA_TOPIC), gop_seconds=1.0, frames_per_second=_SPEC.image_hz
        )

    assert len(access_units) == _SOURCE_FRAME_COUNT
    assert consumed_quit_command_count() == 0
