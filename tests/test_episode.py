from pathlib import Path
from typing import Any

import pytest
from mcap.writer import CompressionType, Writer

from hflow.episode import Episode


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
