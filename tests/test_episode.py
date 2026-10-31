import re
from pathlib import Path
from typing import Any

import pytest
from episode_test_helpers import (
    NUMBERED_VIDEO_TOPIC,
    number_in_luma,
    source_frame_number,
    write_numbered_h264_episodes,
)
from mcap.writer import CompressionType, Writer

from hflow.episode import Episode, _frame_selection_expression


def test_sparse_decoded_timestamps_keep_original_source_positions(tmp_path: Path) -> None:
    from hflow.episode import _decoded_frame_indices

    decoded_trace = tmp_path / "decoded.framecrc"
    decoded_trace.write_text(
        "#tb 0: 1/1000000\n"
        "0, 0, 0, 33333, 1, 0\n"
        "0, 1000000, 1000000, 33333, 1, 0\n"
        "0, 2000000, 2000000, 33333, 1, 0\n",
        encoding="utf-8",
    )
    assert _decoded_frame_indices(decoded_trace, source_fps=30.0) == {0, 30, 60}


@pytest.fixture(scope="module")
def numbered_h264_episodes(tmp_path_factory: pytest.TempPathFactory) -> dict[int | None, Path]:
    return write_numbered_h264_episodes(tmp_path_factory.mktemp("numbered-video"))


def _number_in_frame(path: Path) -> int:
    import cv2

    pixels = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert pixels is not None
    return number_in_luma(pixels)


@pytest.mark.requires_system_ffmpeg
@pytest.mark.parametrize(
    ("fps", "start_s", "end_s"),
    [
        (30.0, None, None),
        (10.0, None, None),
        (3.0, None, None),
        (1.0, None, None),
        (3.0, 0.0, 1.7),
        (3.0, 0.2, 1.7),
        (10.0, 0.37, 2.3),
        (1.0, 0.5, 4.2),
        (40.0, None, 0.5),
    ],
)
def test_frames_label_the_pixels_that_were_sampled(
    numbered_h264_episodes: dict[int | None, Path],
    tmp_path: Path,
    fps: float,
    start_s: float | None,
    end_s: float | None,
) -> None:
    with Episode(numbered_h264_episodes[None], workdir=tmp_path) as episode:
        frames = episode.frames(NUMBERED_VIDEO_TOPIC, fps=fps, start_s=start_s, end_s=end_s)
    assert frames
    for frame in frames:
        named = source_frame_number(frame.log_time_ns)
        shown = _number_in_frame(frame.path)
        assert abs(shown - named) <= 1, (fps, start_s, end_s, frame.path.name, named, shown)


@pytest.mark.requires_system_ffmpeg
def test_short_frame_request_bounds_decoded_trace(
    numbered_h264_episodes: dict[int | None, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hflow import episode as episode_module

    original_decoded_frame_indices = episode_module._decoded_frame_indices
    decoded_source_indices: set[int] = set()

    def capture_decoded_indices(framecrc_path: Path, *, source_fps: float) -> set[int]:
        source_indices = original_decoded_frame_indices(framecrc_path, source_fps=source_fps)
        decoded_source_indices.update(source_indices)
        return source_indices

    monkeypatch.setattr(episode_module, "_decoded_frame_indices", capture_decoded_indices)
    with Episode(numbered_h264_episodes[None], workdir=tmp_path) as episode:
        frames = episode.frames(NUMBERED_VIDEO_TOPIC, fps=10.0, end_s=0.5)
    assert len(frames) == 5
    assert [_number_in_frame(frame.path) for frame in frames] == [0, 3, 6, 9, 12]
    assert decoded_source_indices == set(range(15))


@pytest.mark.requires_system_ffmpeg
@pytest.mark.parametrize("fps", [30.0, 10.0, 3.0, 1.0])
def test_frames_omit_undecoded_ticks_without_shifting_later_samples(
    numbered_h264_episodes: dict[int | None, Path], tmp_path: Path, fps: float
) -> None:
    with Episode(numbered_h264_episodes[100], workdir=tmp_path) as episode:
        frames = episode.frames(NUMBERED_VIDEO_TOPIC, fps=fps)
    assert frames
    sampled_ticks = {int(frame.path.stem.removeprefix("frame_")) - 1 for frame in frames}
    # Original MP4 PTS 100..107 are absent after dropping IDR 100, and the decoder
    # conceals 108 from an unrelated reference until the next keyframe at 109:
    # neither filter duplication of frame 99 nor that concealed picture may
    # become a sample at those tick positions.
    for tick in range(100, 109):
        if tick % (30 / fps) == 0:
            assert int(tick * fps / 30) not in sampled_ticks
    assert max(sampled_ticks) >= int(120 * fps / 30)
    for frame in frames:
        assert abs(_number_in_frame(frame.path) - source_frame_number(frame.log_time_ns)) <= 1


@pytest.mark.requires_system_ffmpeg
def test_frames_reextract_old_cache_and_reuse_new_cache(
    numbered_h264_episodes: dict[int | None, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    from hflow.episode import _sanitize_topic

    old_cache = (
        tmp_path / f"frames_{_sanitize_topic(NUMBERED_VIDEO_TOPIC)}_f1.000000_s0.000000_eend"
    )
    old_cache.mkdir()
    (old_cache / "frame_000001.jpg").write_bytes(b"old cached pixels")

    with Episode(numbered_h264_episodes[None], workdir=tmp_path) as episode:
        first = episode.frames(NUMBERED_VIDEO_TOPIC, fps=1.0)
        assert len(first) == 5
        assert first[0].path.parent.name.startswith("frames_v3_")
        assert _number_in_frame(first[0].path) == 0
        assert {path.suffix for path in first[0].path.parent.iterdir()} == {".jpg"}

        def forbid_second_extraction(*args: Any, **kwargs: Any) -> None:
            pytest.fail("cache hit must not rerun ffmpeg")

        with monkeypatch.context() as patched:
            patched.setattr(subprocess, "run", forbid_second_extraction)
            assert episode.frames(NUMBERED_VIDEO_TOPIC, fps=1.0) == first
    with (
        Episode(numbered_h264_episodes[None], workdir=tmp_path) as episode,
        monkeypatch.context() as patched,
    ):
        patched.setattr(subprocess, "run", forbid_second_extraction)
        assert episode.frames(NUMBERED_VIDEO_TOPIC, fps=1.0) == first


@pytest.mark.requires_system_ffmpeg
def test_missing_decoded_frame_trace_does_not_publish_frame_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    from hflow.testing import SyntheticEpisodeSpec, synthesize_episode
    from hflow.transform import write_canonical_episode

    source = tmp_path / "source.mcap"
    canonical = tmp_path / "canonical.mcap"
    workdir = tmp_path / "cache"
    synthesize_episode(source, SyntheticEpisodeSpec(duration_s=1.0, cameras=("cam",), image_hz=5.0))
    write_canonical_episode(source, canonical)

    def missing_sidecar(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        frame_pattern = next(value for value in command if value.endswith("frame_%06d.jpg"))
        Path(frame_pattern.replace("%06d", "000001")).write_bytes(b"partial JPEG")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    with Episode(canonical, workdir=workdir) as episode:
        with monkeypatch.context() as patched:
            patched.setattr(subprocess, "run", missing_sidecar)
            with pytest.raises(RuntimeError, match="Invalid decoded frame timestamps"):
                episode.frames(fps=2.0)
        assert not [path for path in workdir.iterdir() if path.is_dir()]
        frames = episode.frames(fps=2.0)
        assert len(frames) == 2
        assert all(frame.path.is_file() for frame in frames)


@pytest.mark.requires_system_ffmpeg
@pytest.mark.parametrize("source_fps", [29.87, 29.97])
def test_frames_preserve_sample_ticks_with_fractional_source_rates(
    tmp_path: Path, source_fps: float
) -> None:
    from hflow.testing import SyntheticEpisodeSpec, synthesize_episode
    from hflow.transform import write_canonical_episode

    source = tmp_path / "source.mcap"
    canonical = tmp_path / "canonical.mcap"
    synthesize_episode(
        source,
        SyntheticEpisodeSpec(
            duration_s=3.4,
            cameras=("cam",),
            image_hz=source_fps,
            black_segment=None,
            timestamp_offset_segment=None,
        ),
    )
    write_canonical_episode(source, canonical)
    with Episode(canonical, workdir=tmp_path / "cache") as episode:
        for sample_fps in (0.5, 3.0, 40.0):
            frames = episode.frames(fps=sample_fps, start_s=0.27, end_s=2.7)
            assert frames
            indices = [int(frame.path.stem.removeprefix("frame_")) for frame in frames]
            assert indices == list(range(1, len(indices) + 1))
            assert all(frame.path.is_file() for frame in frames)
            assert [frame.log_time_ns for frame in frames] == sorted(
                frame.log_time_ns for frame in frames
            )


def _write_two_json_channels(
    mcap_path: Path, topics: tuple[str, str], payloads: tuple[bytes, bytes]
) -> tuple[int, int]:
    """One message per channel, written in channel order; returns the channel ids."""
    with mcap_path.open("wb") as stream:
        writer = Writer(stream, chunk_size=64 * 1024, compression=CompressionType.NONE)
        writer.start()
        channel_ids = tuple(
            writer.register_channel(topic=topic, message_encoding="json", schema_id=0)
            for topic in topics
        )
        for log_time, (channel_id, payload) in enumerate(
            zip(channel_ids, payloads, strict=True), start=1
        ):
            writer.add_message(
                channel_id=channel_id, log_time=log_time, publish_time=log_time, data=payload
            )
        writer.finish()
    first_channel_id, second_channel_id = channel_ids
    return first_channel_id, second_channel_id


@pytest.mark.parametrize(
    ("topics", "payloads", "select_by_channel_id"),
    [
        pytest.param(
            ("/target", "/camera"),
            (b'{"value": "target"}', b'{"value": "camera"}'),
            False,
            id="by-topic-excludes-other-topics",
        ),
        # Two channels share a topic, so filtering on the topic alone would
        # return both messages; only the channel id tells them apart.
        pytest.param(
            ("/target", "/target"),
            (b'{"channel": 1}', b'{"channel": 2}'),
            True,
            id="by-channel-id-excludes-a-duplicate-topic",
        ),
    ],
)
def test_episode_channel_reads_only_the_selected_channel(
    tmp_path: Path,
    topics: tuple[str, str],
    payloads: tuple[bytes, bytes],
    select_by_channel_id: bool,
) -> None:
    mcap_path = tmp_path / "episode.mcap"
    selected_channel_id, _ = _write_two_json_channels(mcap_path, topics, payloads)

    with Episode(mcap_path) as episode:
        channel = episode.channel(selected_channel_id if select_by_channel_id else topics[0])

        assert channel.topic == "/target"
        assert channel.channel_id == selected_channel_id
        assert len(channel) == 1
        assert channel.raw == [payloads[0]]


def test_concurrent_frame_extractions_share_workdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from hflow.testing import SyntheticEpisodeSpec, synthesize_episode
    from hflow.transform import write_canonical_episode

    source_mcap = tmp_path / "source.mcap"
    canonical_mcap = tmp_path / "canonical.mcap"
    workdir = tmp_path / "workdir"
    synthesize_episode(
        source_mcap,
        SyntheticEpisodeSpec(duration_s=1.0, cameras=("cam",), image_hz=5.0),
    )
    write_canonical_episode(source_mcap, canonical_mcap)

    def extract_frames() -> list[str]:
        with Episode(canonical_mcap, workdir=workdir) as ep:
            return [str(frame.path) for frame in ep.frames(fps=2.0)]

    def extract_indices() -> list[str]:
        with Episode(canonical_mcap, workdir=workdir) as ep:
            return [str(frame.path) for frame in ep.frames_at_indices(frame_indices=[0, 2])]

    # Synchronize callers so both pass the cache-miss check and stage concurrently
    # before either caller publishes.
    barrier_frames = threading.Barrier(2)
    original_run = subprocess.run

    def barrier_run_frames(*args: Any, **kwargs: Any) -> Any:
        barrier_frames.wait(timeout=5.0)
        return original_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", barrier_run_frames)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(extract_frames) for _ in range(2)]
        results = [f.result() for f in futures]
    assert results[0] == results[1]
    assert len(results[0]) == 2
    assert all(Path(p).is_file() for p in results[0])

    barrier_indices = threading.Barrier(2)

    def barrier_run_indices(*args: Any, **kwargs: Any) -> Any:
        barrier_indices.wait(timeout=5.0)
        return original_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", barrier_run_indices)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(extract_indices) for _ in range(2)]
        results = [f.result() for f in futures]
    assert results[0] == results[1]
    assert len(results[0]) == 2
    assert all(Path(p).is_file() for p in results[0])

    assert not [d for d in workdir.iterdir() if d.is_dir() and d.name.endswith(".tmp")]


def test_frames_at_indices_rejects_incomplete_cache(tmp_path: Path) -> None:
    from hflow.testing import SyntheticEpisodeSpec, synthesize_episode
    from hflow.transform import write_canonical_episode

    source_mcap = tmp_path / "source.mcap"
    canonical_mcap = tmp_path / "canonical.mcap"
    workdir = tmp_path / "workdir_incomplete"
    synthesize_episode(
        source_mcap,
        SyntheticEpisodeSpec(duration_s=1.0, cameras=("cam",), image_hz=5.0),
    )
    write_canonical_episode(source_mcap, canonical_mcap)

    with Episode(canonical_mcap, workdir=workdir) as ep:
        frames = ep.frames_at_indices(frame_indices=[0, 2])
        assert len(frames) == 2

    # Simulate an incomplete cache by deleting one frame
    frames[0].path.unlink()

    with (
        Episode(canonical_mcap, workdir=workdir) as ep,
        pytest.raises(RuntimeError, match="Incomplete frame cache"),
    ):
        ep.frames_at_indices(frame_indices=[0, 2])


@pytest.mark.parametrize("method", ["frames", "frames_at_indices"])
def test_rename_failure_propagates_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    from hflow.testing import SyntheticEpisodeSpec, synthesize_episode
    from hflow.transform import write_canonical_episode

    source_mcap = tmp_path / "source.mcap"
    canonical_mcap = tmp_path / "canonical.mcap"
    workdir = tmp_path / f"workdir_err_{method}"
    synthesize_episode(
        source_mcap,
        SyntheticEpisodeSpec(duration_s=1.0, cameras=("cam",), image_hz=5.0),
    )
    write_canonical_episode(source_mcap, canonical_mcap)

    original_rename = Path.rename

    def fail_rename(self: Path, target: Path) -> Path:
        if not target.exists():
            raise PermissionError("simulated permission denied")
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", fail_rename)

    with (
        Episode(canonical_mcap, workdir=workdir) as ep,
        pytest.raises(PermissionError, match="simulated permission denied"),
    ):
        if method == "frames":
            ep.frames(fps=2.0)
        else:
            ep.frames_at_indices(frame_indices=[0, 2])

    assert not [d for d in workdir.iterdir() if d.is_dir() and d.name.endswith(".tmp")]


@pytest.mark.parametrize("method", ["frames", "frames_at_indices"])
def test_failed_ffmpeg_extraction_cleans_up_and_allows_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    import subprocess

    from hflow.testing import SyntheticEpisodeSpec, synthesize_episode
    from hflow.transform import write_canonical_episode

    source_mcap = tmp_path / "source.mcap"
    canonical_mcap = tmp_path / "canonical.mcap"
    workdir = tmp_path / f"workdir_fail_{method}"
    synthesize_episode(
        source_mcap,
        SyntheticEpisodeSpec(duration_s=1.0, cameras=("cam",), image_hz=5.0),
    )
    write_canonical_episode(source_mcap, canonical_mcap)

    def failing_run(*args: Any, **kwargs: Any) -> Any:
        return subprocess.CompletedProcess(
            args=["ffmpeg"], returncode=1, stdout="", stderr="ffmpeg crashed"
        )

    monkeypatch.setattr(subprocess, "run", failing_run)

    expected_error = (
        "ffmpeg frame extraction produced no frames"
        if method == "frames"
        else "ffmpeg extracted 0 of 2 selected frames"
    )

    with (
        Episode(canonical_mcap, workdir=workdir) as ep,
        pytest.raises(RuntimeError, match=expected_error),
    ):
        if method == "frames":
            ep.frames(fps=2.0)
        else:
            ep.frames_at_indices(frame_indices=[0, 2])

    assert not [d for d in workdir.iterdir() if d.is_dir()]

    monkeypatch.undo()
    with Episode(canonical_mcap, workdir=workdir) as ep:
        if method == "frames":
            frames = ep.frames(fps=2.0)
        else:
            frames = ep.frames_at_indices(frame_indices=[0, 2])
        assert len(frames) == 2
        assert all(f.path.is_file() for f in frames)


def test_empty_channel_to_arrow(tmp_path: Path) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    mcap_path = tmp_path / "empty_channel.mcap"
    with mcap_path.open("wb") as stream:
        writer = Writer(stream, chunk_size=64 * 1024, compression=CompressionType.NONE)
        writer.start()
        writer.register_channel(topic="/empty_json", message_encoding="json", schema_id=0)
        writer.register_channel(
            topic="/empty_unsupported", message_encoding="unsupported_custom", schema_id=0
        )
        writer.finish()

    with Episode(mcap_path) as episode:
        for topic in ("/empty_json", "/empty_unsupported"):
            channel = episode.channel(topic)
            assert len(channel) == 0
            table = channel.to_arrow()
            assert isinstance(table, pyarrow.Table)
            assert table.num_rows == 0
            assert table.column_names == ["log_time_ns"]
            assert table.schema.field("log_time_ns").type == pyarrow.int64()


def test_channel_to_arrow_numeric_list_with_null_elements() -> None:
    import json

    import numpy as np

    from hflow.episode import ChannelData
    from hflow.reader import TopicInfo

    info = TopicInfo(
        topic="/arm/joint_state",
        channel_id=1,
        schema_name="JointState",
        schema_encoding="jsonschema",
        message_encoding="json",
        message_count=4,
        schema_data=b"{}",
    )
    cd = ChannelData(
        topic="/arm/joint_state",
        channel_id=1,
        info=info,
        log_times=np.array([1000, 2000, 3000, 4000], dtype=np.int64),
        publish_times=np.array([1000, 2000, 3000, 4000], dtype=np.int64),
        raw=[
            b'{"position": [1.0, 2.0, null], "untyped": [null, null]}',
            b'{"position": [null, 2.0, 3.0], "untyped": [null, null]}',
            b'{"position": [null, null, null], "untyped": [null, null]}',
            b'{"position": [1.0, 2.0, 3.0], "untyped": [null, null]}',
        ],
        decoder=lambda b: json.loads(b.decode()),
    )
    table = cd.to_arrow()
    assert "position" in table.column_names
    # Untyped all-null list column carries no element type and is omitted like an empty list
    assert "untyped" not in table.column_names
    assert table["position"].to_pylist() == [
        [1.0, 2.0, None],
        [None, 2.0, 3.0],
        [None, None, None],
        [1.0, 2.0, 3.0],
    ]


def test_frame_selection_expression_keeps_large_index_windows_one_frame_wide() -> None:
    """Six-digit time windows must not swallow the neighbouring picture.

    ``%g`` prints six significant digits, so index 250001 at 25 fps collapsed to
    ``between(t,10000,10000.1)`` -- a window covering indices 250000-250002. The
    boundaries must carry round-trip precision instead (#730 review).
    """
    fps = 25.0
    frame_indices = [250001, 250010]
    expression = _frame_selection_expression(frame_indices, fps=fps)

    windows = re.findall(r"between\(t\\,([^,\\]+)\\,([^)\\]+)\)", expression)
    assert len(windows) == len(frame_indices)
    for frame_index, (start, end) in zip(frame_indices, windows, strict=True):
        start_time, end_time = float(start), float(end)
        assert start_time < frame_index / fps < end_time
        assert end_time - start_time == pytest.approx(1 / fps)
