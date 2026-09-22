"""Outcome-focused coverage for first-class LeRobot Dataset v3 import.

These tests exercise the metadata-driven discovery and fail-loud behavior
with a synthetic v3-style corpus, without asserting third-party
implementation details or contacting external services.
"""

import hashlib
import json
import logging
import shutil
import subprocess
import threading
from collections.abc import Sequence
from dataclasses import replace
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlsplit

import pytest
from lerobot_test_helpers import (
    CorpusEpisodeRow,
    publish_staged_episode,
    split_episode_metadata_into_shards,
    stub_hub_download,
    stub_hub_repo_info,
    stub_single_shard_hub_corpus,
    two_camera_v3_info,
    write_v3_corpus,
    write_v3_data_parquet,
)

import hflow.importers.lerobot as prep
from hflow.cli import main as cli_main
from hflow.reader import open_reader
from hflow.storage import LocalStorageRoot, StorageRoot

_DERIVE = prep._derive_numeric_schema
_ENCODE = prep._encode_cdr_float32_array
_SIX_DIMENSION_NUMERIC_SCHEMAS = {
    "observation.state": prep._NumericSchema(name="observation.state", dim=6),
    "action": prep._NumericSchema(name="action", dim=6),
}


def _numeric_feature(name: str, dtype: str, shape: list) -> prep._NumericFeature:
    return prep._NumericFeature(name=name, dtype=dtype, shape=tuple(shape))


def _source_archive(
    dataset_source: prep.DatasetSource,
    cache_dir: Path,
    *,
    info: dict,
    episodes: Sequence[prep._EpisodeRow] = (),
    video_keys: Sequence[str] = (),
) -> prep._SourceArchive:
    """A source archive whose metadata went through the real info.json parser.

    Building the fakes this way keeps them honest: a test corpus that the
    parser would refuse cannot reach the conversion code under test.
    """
    return prep._SourceArchive(
        dataset_information=prep._parse_dataset_information(info),
        episodes=tuple(episodes),
        video_keys=tuple(video_keys),
        cache_dir=cache_dir,
        dataset=dataset_source,
    )


def _single_camera_episode_row(
    episode_index: int,
    *,
    camera_key: str,
    task: str | None = None,
    length: int = 1,
    data_from: int = 0,
    video_file_index: str = "000",
    from_timestamp: float = 0.0,
    to_timestamp: float = 0.0,
) -> prep._EpisodeRow:
    """An episode in data chunk and file ``000`` with one camera's video window."""
    return prep._EpisodeRow(
        episode_index=episode_index,
        task=f"task-{episode_index}" if task is None else task,
        length=length,
        data_chunk="000",
        data_file="000",
        data_from=data_from,
        data_to=data_from + length,
        video_windows={
            camera_key: prep._VideoWindow(
                chunk_index="000",
                file_index=video_file_index,
                from_timestamp=from_timestamp,
                to_timestamp=to_timestamp,
            )
        },
    )


_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")

# Only the remux and frame-slicing tests below shell out. Scoping this to the
# module would skip the metadata tests too, and none of those touch ffmpeg.
_requires_system_ffmpeg = pytest.mark.skipif(
    _FFMPEG is None or _FFPROBE is None,
    reason="system ffmpeg/ffprobe required to construct and inspect the test video",
)


def _build_fake_corpus(corpus_root: Path) -> dict:
    """Synthetic v3 metadata: 4 episodes, 2 cameras, 6-dim state/action."""
    info = two_camera_v3_info()
    write_v3_corpus(
        corpus_root,
        info=info,
        episode_rows=[
            CorpusEpisodeRow(
                episode_index=episode_index,
                length=60 + episode_index * 5,
                dataset_from_index=episode_index * 200,
                video_to_timestamp=2.0 + episode_index * 0.2,
                tasks=(f"task-{episode_index}",),
                data_chunk_index="chunk-000",
                data_file_index="file-000",
                video_chunk_index="chunk-000",
                video_file_index="file-000",
            )
            for episode_index in range(4)
        ],
        timestamps_as_double=True,
    )
    return {"info": info, "cache_dir": corpus_root}


def test_derive_numeric_schema_float32_vector() -> None:
    schema = _DERIVE(_numeric_feature("observation.state", "float32", [6]))
    assert schema.name == "observation.state"
    assert schema.dim == 6


def test_derive_numeric_schema_rejects_unsupported() -> None:
    with pytest.raises(ValueError, match="unsupported feature"):
        _DERIVE(_numeric_feature("action", "float64", [6]))
    with pytest.raises(ValueError, match="unsupported feature"):
        _DERIVE(_numeric_feature("observation.state", "float32", [2, 3]))
    with pytest.raises(ValueError, match="unsupported feature"):
        _DERIVE(_numeric_feature("observation.state", "float32", []))
    for shape in ([True], [False]):
        with pytest.raises(
            ValueError,
            match=rf"unsupported feature action: dtype=float32, shape=\[{shape[0]}\]",
        ):
            _DERIVE(_numeric_feature("action", "float32", shape))


def test_import_rejects_required_boolean_dimension_without_dataset_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "out"
    dataset_source = prep.DatasetSource(repo_id="fake/repo", revision="abc", license="apache-2.0")
    stub_hub_repo_info(monkeypatch, resolved_sha="abc")
    monkeypatch.setattr(
        prep,
        "_ensure_source_archive",
        lambda source, cache_dir: _source_archive(
            dataset_source,
            cache_dir,
            info={
                "fps": 30,
                "data_path": "data/{chunk_index}/{file_index}.parquet",
                "video_path": "videos/{camera_key}/{chunk_index}/{file_index}.mp4",
                "features": {
                    prep.DEFAULT_CAMERA_KEY: {
                        "dtype": "video",
                        "shape": [480, 640, 3],
                        "info": {"is_depth_map": False},
                    },
                    "action": {"dtype": "float32", "shape": [True]},
                    "observation.state": {"dtype": "float32", "shape": [6]},
                },
            },
            video_keys=[prep.DEFAULT_CAMERA_KEY],
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"unsupported feature action: dtype=float32, shape=\[True\]",
    ):
        prep.import_lerobot_dataset(dataset_repo="fake/repo", output_dir=output_dir)

    assert not (output_dir / "landing").exists()
    assert not (output_dir / "prepared-manifest.json").exists()


def test_cdr_float32_array_n_byte_compatible() -> None:
    out = _ENCODE([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    # CDR XCDR1 little-endian encapsulation header (same bytes hflow decodes)
    assert out[:4] == b"\x00\x01\x00\x00"
    assert len(out) == 4 + 4 * 6


def test_index_discovery_multi_camera_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _build_fake_corpus(tmp_path)
    stub_hub_repo_info(monkeypatch)
    stub_single_shard_hub_corpus(monkeypatch, tmp_path, corpus["info"])

    ds = prep.DatasetSource(repo_id="fake/repo", revision="abc", license="apache-2.0")
    found = prep._ensure_source_archive(ds, tmp_path)
    assert len(found.episodes) == 4
    assert found.episodes[0].length == 60
    assert found.episodes[1].length == 65
    assert found.episodes[0].data_from == 0
    assert found.episodes[0].data_to == 60
    assert set(found.video_keys) == {"observation.images.up", "observation.images.side"}
    # The control for the fps refusals: a positive fps loads.
    assert found.fps == 30
    assert found.episodes[0].video_windows["observation.images.up"].to_timestamp == pytest.approx(
        2.0
    )


@pytest.mark.parametrize(
    "second_shard_path",
    [
        # Distinct basenames in one chunk directory: how lerobot/droid_1.0.1
        # ships its seven metadata shards.
        "meta/episodes/chunk-000/file-001.parquet",
        # The same basename in the next chunk directory, which a cache keyed
        # by basename alone would collapse onto the first shard.
        "meta/episodes/chunk-001/file-000.parquet",
    ],
)
def test_index_discovery_reads_every_metadata_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, second_shard_path: str
) -> None:
    """Episodes and video windows come from every ``meta/episodes`` shard (#293).

    The corpus is split the way Dataset v3 shards it: a different pair of
    episodes in each file and a distinct video window per episode.
    """
    corpus = _build_fake_corpus(tmp_path)
    shard_paths = ("meta/episodes/chunk-000/file-000.parquet", second_shard_path)
    split_episode_metadata_into_shards(
        tmp_path, dict(zip(shard_paths, ((0, 1), (2, 3)), strict=True))
    )

    stub_hub_repo_info(monkeypatch)
    monkeypatch.setattr(prep, "_fetch_info_json", lambda repo, rev, cache: corpus["info"])
    monkeypatch.setattr(
        prep,
        "_hf_episode_metadata_files",
        lambda repo, rev: list(shard_paths),
    )

    def supply_metadata_file(filename: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(tmp_path / filename, destination)

    stub_hub_download(monkeypatch, supply_metadata_file)

    ds = prep.DatasetSource(repo_id="fake/repo", revision="abc", license="apache-2.0")
    cache_dir = tmp_path / "cache"
    found = prep._ensure_source_archive(ds, cache_dir)
    assert [episode.episode_index for episode in found.episodes] == [0, 1, 2, 3]
    assert set(found.video_keys) == {"observation.images.up", "observation.images.side"}
    for episode in found.episodes:
        episode_index = episode.episode_index
        assert episode.length == 60 + episode_index * 5
        for camera_key in found.video_keys:
            window = episode.video_windows[camera_key]
            assert window.chunk_index == "chunk-000"
            assert window.to_timestamp == pytest.approx(2.0 + episode_index * 0.2)


def test_conversion_selects_video_by_file_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _build_fake_corpus(tmp_path)
    camera_key = "observation.images.up"
    dataset_source = prep.DatasetSource(repo_id="fake/repo", revision="abc", license="apache-2.0")

    source_archive = _source_archive(
        dataset_source,
        corpus["cache_dir"],
        info=corpus["info"],
        episodes=[
            _single_camera_episode_row(0, camera_key=camera_key, video_file_index="000"),
            _single_camera_episode_row(1, camera_key=camera_key, video_file_index="001"),
        ],
        video_keys=[camera_key],
    )
    converted_sources: list[bytes] = []

    def fake_download(filename: str, destination_path: Path, **_kwargs: object) -> None:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if filename.startswith("videos/"):
            destination_path.write_bytes(filename.encode())
            return
        shutil.copy(tmp_path / "data" / "chunk-000" / "file-000.parquet", destination_path)

    def fake_transcode(mp4_path: Path, *_args: object, **_kwargs: object) -> list[bytes]:
        converted_sources.append(mp4_path.read_bytes())
        return [b"access-unit"]

    stub_hub_download(monkeypatch, fake_download)
    monkeypatch.setattr(prep, "_transcode_mp4_to_h264", fake_transcode)
    monkeypatch.setattr(prep, "_get_video_pts_times", lambda path: [0])
    monkeypatch.setattr(prep, "ffmpeg_version", lambda: "test-ffmpeg")
    monkeypatch.setattr(
        prep,
        "write_canonical_episode",
        lambda source_path, output_path, *_args, **_kwargs: shutil.copy(source_path, output_path),
    )

    published_uris: list[str] = []
    receipts: list[prep._PublishedEpisode] = []
    for episode_index in (0, 1, 0):
        receipt = prep._convert_single_episode(
            source_archive=source_archive,
            dataset_source=dataset_source,
            storage=LocalStorageRoot(tmp_path / "output"),
            episode_index=episode_index,
            camera_keys=(camera_key,),
            numeric_schemas=_SIX_DIMENSION_NUMERIC_SCHEMAS,
            frames_per_second=30,
        )
        published_uris.append(receipt["uri"])
        receipts.append(receipt)

    assert published_uris[0].endswith("landing/lerobot_episode_0001.mcap")
    assert published_uris[1].endswith("landing/lerobot_episode_0002.mcap")

    # The manifest tests build their receipts in the convert stub, so this is
    # the only place the real function's receipt is checked against the object
    # it published rather than against a value the test wrote itself.
    for receipt in receipts:
        landed_path = Path(receipt["uri"])
        assert receipt["content_id"] == prep.content_episode_id(landed_path)
        assert receipt["size_bytes"] == landed_path.stat().st_size

    video_filenames = [
        "videos/observation.images.up/chunk-000/file-000.mp4",
        "videos/observation.images.up/chunk-000/file-001.mp4",
    ]
    assert converted_sources == [
        video_filenames[0].encode(),
        video_filenames[1].encode(),
        video_filenames[0].encode(),
    ]


def test_camera_selection_validates_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus = _build_fake_corpus(tmp_path)
    stub_hub_repo_info(monkeypatch)
    stub_single_shard_hub_corpus(monkeypatch, tmp_path, corpus["info"])

    with pytest.raises(ValueError, match="not found"):
        prep.import_lerobot_dataset(
            dataset_repo="fake/repo",
            revision="abc",
            output_dir=tmp_path / "out",
            camera_keys="observation.nope",
        )


def test_import_namespaces_source_cache_by_resolved_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Realistic valid shas the production validator at
    # src/hflow/importers/lerobot.py would accept: 40-character hexadecimal,
    # visibly distinct at the start so a reader can tell them apart at a
    # glance. ``branch-a`` and ``tag-a`` resolve to the same sha on purpose:
    # they are the two-revisions-one-cache leg of the contract.
    sha_a = "a1b2c3d4e5f60718293a4b5c6d7e8f9001020304"
    sha_b = "f0e1d2c3b4a5968778695a4b3c2d1e0f00112233"
    resolved_shas = {
        "branch-a": sha_a,
        "branch-b": sha_b,
        "tag-a": sha_a,
    }
    cache_observations: list[tuple[str, Path, str]] = []

    monkeypatch.setattr(
        prep,
        "_hf_repo_info",
        lambda repo, revision: {"sha": resolved_shas[revision], "license": "apache-2.0"},
    )

    def fake_ensure_source_archive(
        dataset_source: prep.DatasetSource, cache_dir: Path
    ) -> prep._SourceArchive:
        cache_dir.mkdir(parents=True, exist_ok=True)
        source_marker = cache_dir / "source-marker.txt"
        if not source_marker.exists():
            source_marker.write_text(dataset_source.revision)
        cache_observations.append((dataset_source.revision, cache_dir, source_marker.read_text()))
        return _source_archive(
            dataset_source,
            cache_dir,
            info={
                "fps": 30,
                "data_path": "data/{chunk_index}/{file_index}.parquet",
                "video_path": "videos/{camera_key}/{chunk_index}/{file_index}.mp4",
                "features": {
                    "action": {"dtype": "float32", "shape": [1]},
                    "observation.state": {"dtype": "float32", "shape": [1]},
                },
            },
            video_keys=[prep.DEFAULT_CAMERA_KEY],
        )

    monkeypatch.setattr(prep, "_ensure_source_archive", fake_ensure_source_archive)

    for revision in ("branch-a", "branch-b", "tag-a"):
        prep.import_lerobot_dataset(
            dataset_repo="fake/repo", revision=revision, output_dir=tmp_path
        )

    assert cache_observations == [
        (sha_a, tmp_path / "_lerobot_cache" / sha_a, sha_a),
        (sha_b, tmp_path / "_lerobot_cache" / sha_b, sha_b),
        (sha_a, tmp_path / "_lerobot_cache" / sha_a, sha_a),
    ]
    assert sorted(path.name for path in (tmp_path / "_lerobot_cache").iterdir()) == [
        sha_a,
        sha_b,
    ]


@pytest.mark.parametrize("resolved_sha", ["../../evil", "/tmp/probe-328-absolute"])
def test_hf_repo_info_rejects_malformed_commit_sha(
    resolved_sha: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        prep,
        "HfApi",
        lambda: SimpleNamespace(
            dataset_info=lambda *_args, **_kwargs: SimpleNamespace(sha=resolved_sha)
        ),
    )

    with pytest.raises(ValueError, match="malformed commit sha"):
        prep._hf_repo_info("fake/repo", "main")


def test_import_refuses_invalid_arguments_before_network(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dataset_repo must not be empty"):
        prep.import_lerobot_dataset(dataset_repo="", output_dir=tmp_path)
    with pytest.raises(ValueError, match="revision must not be empty"):
        prep.import_lerobot_dataset(dataset_repo="lerobot/pusht", revision=" ", output_dir=tmp_path)
    with pytest.raises(ValueError, match="episode_index must be zero or greater"):
        prep.import_lerobot_dataset(
            dataset_repo="lerobot/pusht", episode_index=-1, output_dir=tmp_path
        )
    with pytest.raises(ValueError, match="camera_keys must name at least one"):
        prep.import_lerobot_dataset(
            dataset_repo="lerobot/pusht", camera_keys=(), output_dir=tmp_path
        )
    with pytest.raises(ValueError, match="camera_keys must not contain duplicates"):
        prep.import_lerobot_dataset(
            dataset_repo="lerobot/pusht",
            camera_keys=("observation.image", "observation.image"),
            output_dir=tmp_path,
        )


def test_cli_routes_lerobot_import_refusals_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli_main(["import", "lerobot", "--repo", "", "--output-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err == "import lerobot: dataset_repo must not be empty\n"


@_requires_system_ffmpeg
def test_converter_output_remuxes_without_tail_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The converter's stream must decode in full through the hflow remux.

    B-frame H.264 loses its reorder-buffer tail when remuxed from raw Annex
    B to MP4, so decoded_frame_count undercounts healthy episodes (#250).
    The converter encodes with bframes=0; prove the full chain decodes
    every source frame.
    """

    assert _FFMPEG is not None and _FFPROBE is not None
    system_ffmpeg_path = Path(_FFMPEG)
    system_ffprobe_path = Path(_FFPROBE)
    monkeypatch.setattr(prep, "ffmpeg_path", lambda: system_ffmpeg_path)
    monkeypatch.setattr(prep, "ffprobe_path", lambda: system_ffprobe_path)

    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            _FFMPEG,
            "-hide_banner",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=30:duration=3,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-g",
            "30",
            "-keyint_min",
            "30",
            "-sc_threshold",
            "0",
            str(source),
        ],
        capture_output=True,
        check=True,
    )

    from hflow.video import write_access_units_to_mp4

    units = prep._transcode_mp4_to_h264(source, gop_seconds=1.0, frames_per_second=30.0)
    muxed = write_access_units_to_mp4(units, fps=30.0, output=tmp_path / "remux.mp4")

    probe = subprocess.run(
        [
            _FFPROBE,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=noprint_wrappers=1",
            str(muxed),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "nb_read_frames=90" in probe.stdout, probe.stdout


def _decoded_gray_frame_stdin(access_units: "list[bytes]") -> bytes:
    """Decode the first frame of an Annex B H.264 stream fed on stdin."""
    completed = subprocess.run(
        [
            str(_FFMPEG),
            "-v",
            "error",
            "-f",
            "h264",
            "-i",
            "pipe:0",
            "-frames:v",
            "1",
            "-pix_fmt",
            "gray",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        input=b"".join(access_units),
        capture_output=True,
        check=True,
    )
    return completed.stdout


def _source_gray_frame(video: Path, index: int) -> bytes:
    """Decode one frame of the source video by its absolute frame index."""
    completed = subprocess.run(
        [
            str(_FFMPEG),
            "-v",
            "error",
            "-i",
            str(video),
            "-vf",
            f"select=eq(n\\,{index})",
            "-frames:v",
            "1",
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "gray",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    return completed.stdout


def _mean_absolute_difference(left: bytes, right: bytes) -> float:
    assert left and len(left) == len(right)
    return sum(abs(a - b) for a, b in zip(left, right, strict=True)) / len(left)


def _episode_video_access_units(mcap: Path, camera_key: str) -> "list[bytes]":
    """The episode's stored H.264 access units, in log order."""
    from foxglove_schemas_protobuf.CompressedVideo_pb2 import CompressedVideo

    reader = open_reader(mcap)
    units: list[bytes] = []
    try:
        for batch in reader.iter_batches(topics=[f"/{camera_key}"]):
            for message in batch.data:
                video_message = CompressedVideo()
                video_message.ParseFromString(message)
                units.append(video_message.data)
    finally:
        reader.close()
    return units


@_requires_system_ffmpeg
def test_converter_slices_exactly_the_declared_frame_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A window end landing exactly on a frame boundary must not pull the
    next episode's first frame into the slice.

    Source end timestamps are exclusive ([from, to) like the data rows), but
    an end that lands exactly on the frame grid rounds UP to the next frame's
    timestamp when ffmpeg receives it, so the -to cut includes one frame too
    many and the converter refuses the episode. The slice is cut by the
    declared frame count instead, so the parquet and the video cannot
    disagree by one. The first episode of the pinned svla corpus reproduces
    this on real recordings (length 226 vs 227 access units).
    """
    assert _FFMPEG is not None and _FFPROBE is not None
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            _FFMPEG,
            "-hide_banner",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=30:duration=3,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-bf",
            "0",
            "-g",
            "30",
            "-keyint_min",
            "30",
            "-sc_threshold",
            "0",
            str(source),
        ],
        capture_output=True,
        check=True,
    )
    system_ffmpeg_path = Path(_FFMPEG)
    system_ffprobe_path = Path(_FFPROBE)
    monkeypatch.setattr(prep, "ffmpeg_path", lambda: system_ffmpeg_path)
    monkeypatch.setattr(prep, "ffprobe_path", lambda: system_ffprobe_path)
    monkeypatch.setattr(prep, "ffmpeg_version", lambda: "test-ffmpeg")

    camera_key = "observation.images.up"
    corpus = _build_fake_corpus(tmp_path)

    # Two episodes sharing one 90-frame video. Episode 0 declares 62 frames
    # and its window ends at 62/30 s -- exactly the boundary where a
    # timestamp cut includes frame 62 (the first frame of episode 1).
    boundary = 62 / 30.0

    data_path = tmp_path / "data" / "chunk-000" / "file-000.parquet"
    write_v3_data_parquet(
        data_path, episode_lengths=[62, 28], frames_per_second=30, timestamps_as_double=True
    )

    dataset_source = prep.DatasetSource(repo_id="fake/repo", revision="abc", license="apache-2.0")
    source_archive = prep._SourceArchive(
        dataset_information=prep._parse_dataset_information(corpus["info"]),
        episodes=(
            _single_camera_episode_row(
                0, camera_key=camera_key, length=62, data_from=0, to_timestamp=boundary
            ),
            _single_camera_episode_row(
                1,
                camera_key=camera_key,
                length=28,
                data_from=62,
                from_timestamp=boundary,
                to_timestamp=3.0,
            ),
        ),
        video_keys=(camera_key,),
        cache_dir=corpus["cache_dir"],
        dataset=dataset_source,
    )

    def fake_download(filename: str, destination_path: Path, **_kwargs: object) -> None:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if filename.startswith("videos/"):
            shutil.copy(source, destination_path)
            return
        shutil.copy(data_path, destination_path)

    stub_hub_download(monkeypatch, fake_download)

    storage = LocalStorageRoot(tmp_path / "output")
    landed_by_index: dict[int, Path] = {}
    for index in (0, 1):
        receipt = prep._convert_single_episode(
            source_archive=source_archive,
            dataset_source=dataset_source,
            storage=storage,
            episode_index=index,
            camera_keys=(camera_key,),
            numeric_schemas=_SIX_DIMENSION_NUMERIC_SCHEMAS,
            frames_per_second=30,
        )
        assert Path(receipt["uri"]).name == f"lerobot_episode_{index + 1:04d}.mcap"
        landed_path = Path(receipt["uri"])
        landed_by_index[index] = landed_path
        assert receipt["content_id"] == prep.content_episode_id(landed_path)
        assert receipt["size_bytes"] == landed_path.stat().st_size

    # The canonical landing records the source episode it was converted from.
    episode_metadata = open_reader(
        storage.path / "landing" / "lerobot_episode_0001.mcap"
    ).metadata()
    assert episode_metadata["episode/v1"]["source_episode_index"] == "0"

    # The two episodes share one source video, so counts and receipts alone
    # cannot tell their windows apart: a dropped input seek gives episode 1
    # episode 0's frames with a correct count (silent wrong footage). Pin each
    # episode's first decoded frame to its own window's source frame instead.
    episode_openers = {
        index: _decoded_gray_frame_stdin(_episode_video_access_units(path, camera_key))
        for index, path in landed_by_index.items()
    }
    source_frame_0 = _source_gray_frame(source, 0)
    source_frame_62 = _source_gray_frame(source, 62)

    for index, window_start_frame in ((0, 0), (1, 62)):
        opener = episode_openers[index]
        own = source_frame_0 if window_start_frame == 0 else source_frame_62
        other = source_frame_62 if window_start_frame == 0 else source_frame_0
        assert _mean_absolute_difference(opener, own) < _mean_absolute_difference(opener, other), (
            f"episode {index} does not open on source frame {window_start_frame}"
        )
    assert _mean_absolute_difference(episode_openers[0], episode_openers[1]) > 5.0, (
        "the two episodes' opening frames are not distinguishable"
    )


def test_window_times_are_count_anchored_not_boundary_filtered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Quantized container timestamps at the window edges must not change the
    frame count: the episode's extent is its data-row count, not a half-open
    time filter that float32 rounding can tip by one frame in either
    direction. The pinned svla corpus reproduces both directions (231
    packets in a 230-row window; 214 in a 215-row window) on current main.
    """
    import struct

    def f32(value: float) -> float:
        return struct.unpack("f", struct.pack("f", value))[0]

    fps = 30.0
    start_seconds = 10.0
    frame_count = 230
    end_seconds = start_seconds + frame_count / fps
    # Two hundred thirty data rows (indexes 300..529); the chunk carries
    # frames 299..535, and the next episode's first frame (index 530, at
    # exactly ``end_seconds``) quantizes to a hair BELOW the float64 end --
    # inside a half-open [start, end) filter, outside a count-anchored one.
    packets = [f32(index / fps) for index in range(299, 536)]

    monkeypatch.setattr(prep, "_get_video_pts_times", lambda _path: packets)
    rebased = prep._relative_video_pts_times(
        Path("chunk.mp4"), start_seconds, end_seconds, frame_count, fps
    )
    assert len(rebased) == frame_count
    assert rebased == [f32(index / fps) - start_seconds for index in range(300, 530)]

    # A video with too few packets after the window start is refused loudly
    # rather than silently producing a short episode.
    monkeypatch.setattr(prep, "_get_video_pts_times", lambda _path: packets[: frame_count - 1])
    with pytest.raises(ValueError, match=r"video has \d+ packets"):
        prep._relative_video_pts_times(
            Path("chunk.mp4"), start_seconds, end_seconds, frame_count, fps
        )


@pytest.mark.parametrize(
    "bad_fps",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        None,
        "30",
        0,
        -30,
    ],
    ids=[
        "nan",
        "positive-infinity",
        "negative-infinity",
        "boolean",
        "missing",
        "nonnumeric",
        "zero",
        "negative",
    ],
)
def test_info_json_refuses_non_finite_or_non_positive_fps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_fps: object
) -> None:
    """Invalid fps metadata must be refused before any episode discovery runs."""
    corpus = _build_fake_corpus(tmp_path)
    info = dict(corpus["info"])
    if bad_fps is None:
        info.pop("fps")
    else:
        info["fps"] = bad_fps
    expected_value = info.get("fps")
    stub_hub_repo_info(monkeypatch)
    monkeypatch.setattr(prep, "_fetch_info_json", lambda repo, rev, cache: info)

    def fail_discovery(repo: str, rev: str) -> list[str]:
        raise AssertionError("episode metadata discovery must not run after invalid fps")

    monkeypatch.setattr(prep, "_hf_episode_metadata_files", fail_discovery)

    dataset_source = prep.DatasetSource(repo_id="fake/repo", revision="abc", license="apache-2.0")
    cache_dir = tmp_path / "cache"
    with pytest.raises(ValueError, match="fps") as excinfo:
        prep._ensure_source_archive(dataset_source, cache_dir)

    message = str(excinfo.value)
    assert "FPS must be finite and positive" in message
    assert repr(expected_value) in message
    # Refusal happens before episode metadata discovery: no downloads, no output.
    assert not cache_dir.exists() or not (cache_dir / "meta" / "episodes").exists()


def _stub_single_episode_info(camera_metadata: dict | None = None) -> dict:
    """One RGB camera and the two required numeric features.

    ``camera_metadata`` replaces the camera feature outright, which is how the
    depth-refusal tests mark the same camera as depth.
    """
    return {
        "robot_type": "pusht",
        "fps": 30,
        "data_path": "data/{chunk_index}/{file_index}.parquet",
        "video_path": "videos/{camera_key}/{chunk_index}/{file_index}.mp4",
        "features": {
            prep.DEFAULT_CAMERA_KEY: camera_metadata
            or {
                "dtype": "video",
                "shape": [480, 640, 3],
                "info": {"is_depth_map": False},
            },
            "action": {"dtype": "float32", "shape": [1]},
            "observation.state": {"dtype": "float32", "shape": [1]},
        },
    }


def _stub_single_episode_source_archive(
    dataset_source: prep.DatasetSource, cache_dir: Path
) -> prep._SourceArchive:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return _source_archive(
        dataset_source,
        cache_dir,
        info=_stub_single_episode_info(),
        episodes=[
            prep._EpisodeRow(
                episode_index=0,
                task="push",
                length=1,
                data_chunk="000",
                data_file="000",
                data_from=0,
                data_to=1,
            )
        ],
        video_keys=[prep.DEFAULT_CAMERA_KEY],
    )


def _install_publish_through_convert(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    """Exercise StorageRoot.publish without running the video converter."""

    published_keys: list[str] = []

    def fake_convert(
        *,
        source_archive: object,
        dataset_source: object,
        storage: StorageRoot,
        episode_index: int,
        camera_keys: object,
        numeric_schemas: object,
        frames_per_second: object,
    ) -> prep._PublishedEpisode:
        del source_archive, dataset_source, camera_keys, numeric_schemas, frames_per_second
        staged = tmp_path / f"staged-{episode_index}.mcap"
        staged.write_bytes(f"episode-{episode_index}".encode())
        published_keys.append(prep._landing_relative_key(episode_index))
        return publish_staged_episode(storage, staged, episode_index)

    monkeypatch.setattr(prep, "_convert_single_episode", fake_convert)
    return published_keys


@pytest.mark.parametrize(
    "depth_metadata",
    [
        {"info": {"is_depth_map": True}},
        {"info": {"video.is_depth_map": True}},
        {"video_info": {"video.is_depth_map": True}},
        # LeRobot's own is_depth_map() is truthy, not an identity test, so a
        # corpus marked with a string or an int is depth to LeRobot and must
        # not be RGB to us. An `is True` check here would send exactly these
        # down the H.264 path.
        {"info": {"is_depth_map": "true"}},
        {"info": {"is_depth_map": 1}},
        {"info": {"video.is_depth_map": "yes"}},
        {"video_info": {"video.is_depth_map": 1}},
    ],
)
def test_import_refuses_a_depth_video_before_publishing_dataset_output(
    depth_metadata: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "out"

    def ensure_depth_archive(
        dataset_source: prep.DatasetSource, cache_dir: Path
    ) -> prep._SourceArchive:
        archive = _stub_single_episode_source_archive(dataset_source, cache_dir)
        return replace(
            archive,
            dataset_information=prep._parse_dataset_information(
                _stub_single_episode_info(
                    {"dtype": "video", "shape": [24, 32, 1], **depth_metadata}
                )
            ),
        )

    stub_hub_repo_info(monkeypatch, resolved_sha="abc")
    monkeypatch.setattr(prep, "_ensure_source_archive", ensure_depth_archive)
    _install_publish_through_convert(monkeypatch, tmp_path)

    with pytest.raises(
        ValueError,
        match=r"observation\.image.*depth-map video.*cannot preserve depth values",
    ):
        prep.import_lerobot_dataset(
            dataset_repo="fake/repo",
            revision="main",
            output_dir=output_dir,
            episode_index=0,
        )

    assert not (output_dir / "landing").exists()
    assert not (output_dir / "prepared-manifest.json").exists()


def test_import_returns_local_uris_and_keeps_cache_beside_landing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manifest lists every delivered episode with its content id and size.

    A recipient of a prepared corpus gets a receipt that can be checked
    against the landing directory without re-running the import; the
    content id is the same ``content_episode_id`` the catalog dedupes on.
    """
    output_dir = tmp_path / "out"
    stub_hub_repo_info(monkeypatch, resolved_sha="abc")
    monkeypatch.setattr(prep, "_ensure_source_archive", _stub_single_episode_source_archive)
    _install_publish_through_convert(monkeypatch, tmp_path)

    episode_uris = prep.import_lerobot_dataset(
        dataset_repo="fake/repo",
        revision="main",
        output_dir=output_dir,
        episode_index=0,
    )

    assert episode_uris == [str((output_dir / "landing" / "lerobot_episode_0001.mcap").resolve())]
    assert Path(episode_uris[0]).is_file()
    assert (output_dir / "prepared-manifest.json").is_file()
    manifest_payload = json.loads((output_dir / "prepared-manifest.json").read_text())
    landed_episode_path = output_dir / "landing" / "lerobot_episode_0001.mcap"
    assert manifest_payload == {
        "schema_version": 3,
        "dataset": {
            "repo_id": "fake/repo",
            "revision": "abc",
            "license": "apache-2.0",
        },
        "camera_keys": [prep.DEFAULT_CAMERA_KEY],
        "episodes_converted": 1,
        "episodes": [
            {
                "uri": episode_uris[0],
                "content_id": prep.content_episode_id(landed_episode_path),
                "size_bytes": landed_episode_path.stat().st_size,
            }
        ],
        "converter_version": prep.CONVERTER_VERSION,
    }
    assert (output_dir / "_lerobot_cache" / "abc").is_dir()


def test_converter_version_reaches_the_canonical_episode_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What the importer writes into the episode it publishes, not what survives
    canonicalization.

    ``write_canonical_episode`` is stubbed to a copy, the idiom of the sibling
    converter tests, because synthetic access units are not parseable video. So
    the two records are asserted as written rather than as carried through: the
    real transform reaches them by different routes, copying an unrecognized
    record verbatim (``transform.py:886``) but skipping ``episode/v1`` there and
    rewriting it from the source dict (``transform.py:889-890``). Neither route
    is exercised here.
    """
    corpus = _build_fake_corpus(tmp_path)
    camera_key = "observation.images.up"
    dataset_source = prep.DatasetSource(repo_id="fake/repo", revision="abc", license="apache-2.0")
    source_archive = _source_archive(
        dataset_source,
        corpus["cache_dir"],
        info=corpus["info"],
        episodes=[_single_camera_episode_row(0, camera_key=camera_key, task="pick-and-place")],
        video_keys=[camera_key],
    )

    def fake_download(filename: str, destination_path: Path, **_kwargs: object) -> None:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if filename.startswith("videos/"):
            destination_path.write_bytes(filename.encode())
            return
        shutil.copy(tmp_path / "data" / "chunk-000" / "file-000.parquet", destination_path)

    stub_hub_download(monkeypatch, fake_download)
    monkeypatch.setattr(prep, "_transcode_mp4_to_h264", lambda *args, **kwargs: [b"access-unit"])
    monkeypatch.setattr(prep, "_get_video_pts_times", lambda path: [0])
    monkeypatch.setattr(prep, "ffmpeg_version", lambda: "test-ffmpeg")
    monkeypatch.setattr(
        prep,
        "write_canonical_episode",
        lambda source_path, output_path, *args, **kwargs: shutil.copy(source_path, output_path),
    )

    receipt = prep._convert_single_episode(
        source_archive=source_archive,
        dataset_source=dataset_source,
        storage=LocalStorageRoot(tmp_path / "output"),
        episode_index=0,
        camera_keys=(camera_key,),
        numeric_schemas=_SIX_DIMENSION_NUMERIC_SCHEMAS,
        frames_per_second=30,
    )

    episode_metadata = open_reader(receipt["uri"]).metadata()
    assert episode_metadata["episode/v1"]["converter_version"] == prep.CONVERTER_VERSION
    assert episode_metadata["source-provenance/v1"]["converter_version"] == prep.CONVERTER_VERSION


def test_import_publishes_into_a_bucket_data_root_without_uploading_cache(
    tmp_path: Path,
    bucket_over_tmp: tuple[object, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hflow.storage import BucketStorageRoot

    data_root, remote_dir = bucket_over_tmp
    assert isinstance(data_root, BucketStorageRoot)
    stub_hub_repo_info(monkeypatch, resolved_sha="abc")
    monkeypatch.setattr(prep, "_ensure_source_archive", _stub_single_episode_source_archive)
    _install_publish_through_convert(monkeypatch, tmp_path)

    episode_uris = prep.import_lerobot_dataset(
        dataset_repo="fake/repo",
        revision="main",
        output_dir=data_root,
        episode_index=0,
    )

    assert episode_uris == [f"{data_root.url}/landing/lerobot_episode_0001.mcap"]
    assert all(isinstance(uri, str) for uri in episode_uris)
    assert not isinstance(episode_uris[0], Path)
    assert (remote_dir / "landing" / "lerobot_episode_0001.mcap").is_file()
    assert (remote_dir / "prepared-manifest.json").is_file()
    assert data_root.list_names() == [
        "landing/lerobot_episode_0001.mcap",
        "prepared-manifest.json",
    ]
    assert not any(name.startswith("_lerobot_cache") for name in data_root.list_names())
    assert (data_root.mirror / "_lerobot_cache" / "abc").is_dir()


def test_manifest_content_id_detects_a_truncated_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The #379 controlled result as a test: truncating one episode to zero
    bytes is detectable from the delivery by re-hashing against the manifest."""
    from hflow.importers.lerobot_verify import verify_lerobot_import
    from hflow.verification import (
        REASON_CONTENT_ID_MISMATCH,
        REASON_SIZE_MISMATCH,
        VerificationStatus,
    )

    output_dir = tmp_path / "out"
    stub_hub_repo_info(monkeypatch, resolved_sha="abc")
    monkeypatch.setattr(prep, "_ensure_source_archive", _stub_single_episode_source_archive)
    _install_publish_through_convert(monkeypatch, tmp_path)

    prep.import_lerobot_dataset(
        dataset_repo="fake/repo",
        revision="main",
        output_dir=output_dir,
        episode_index=0,
    )

    manifest = json.loads((output_dir / "prepared-manifest.json").read_text())
    entry = manifest["episodes"][0]
    episode_path = output_dir / "landing" / "lerobot_episode_0001.mcap"
    original_size = episode_path.stat().st_size
    assert original_size == entry["size_bytes"]
    assert prep.content_episode_id(episode_path) == entry["content_id"]

    episode_path.write_bytes(b"")
    assert episode_path.stat().st_size != entry["size_bytes"]
    assert prep.content_episode_id(episode_path) != entry["content_id"]

    report = verify_lerobot_import(output_dir)
    assert report.status is VerificationStatus.DAMAGED
    assert {finding.reason for finding in report.findings} == {
        REASON_SIZE_MISMATCH,
        REASON_CONTENT_ID_MISMATCH,
    }


def test_import_skips_bucket_manifest_when_an_episode_publish_fails(
    tmp_path: Path,
    bucket_over_tmp: tuple[object, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hflow.storage import BucketStorageRoot

    data_root, remote_dir = bucket_over_tmp
    assert isinstance(data_root, BucketStorageRoot)

    convert_calls = 0

    def fail_on_second_episode(
        *,
        source_archive: object,
        dataset_source: object,
        storage: StorageRoot,
        episode_index: int,
        camera_keys: object,
        numeric_schemas: object,
        frames_per_second: object,
    ) -> prep._PublishedEpisode:
        nonlocal convert_calls
        del source_archive, dataset_source, camera_keys, numeric_schemas, frames_per_second
        convert_calls += 1
        if episode_index == 1:
            raise RuntimeError("forced publish failure")
        staged = tmp_path / f"staged-{episode_index}.mcap"
        staged.write_bytes(b"first")
        return publish_staged_episode(storage, staged, episode_index)

    stub_hub_repo_info(monkeypatch, resolved_sha="abc")
    monkeypatch.setattr(prep, "_ensure_source_archive", _ensure_two_episode_archive)
    monkeypatch.setattr(prep, "_convert_single_episode", fail_on_second_episode)

    with pytest.raises(RuntimeError, match="forced publish failure"):
        prep.import_lerobot_dataset(
            dataset_repo="fake/repo",
            revision="main",
            output_dir=data_root,
        )

    assert convert_calls == 2
    assert (remote_dir / "landing" / "lerobot_episode_0001.mcap").is_file()
    assert not (remote_dir / "prepared-manifest.json").exists()
    assert "prepared-manifest.json" not in data_root.list_names()


def _write_identity_matching_landing_mcap(
    destination: Path,
    *,
    dataset_source: prep.DatasetSource,
    episode_index: int,
    camera_keys: tuple[str, ...],
    marker: str,
    episode_record_overrides: dict[str, str] | None = None,
    source_provenance_overrides: dict[str, str] | None = None,
    provenance_overrides: dict[str, str] | None = None,
) -> None:
    """Write a landing MCAP whose metadata satisfies import resume identity.

    The three override hooks exist so a caller can break exactly one identity
    field and leave the rest matching, which is what separates the individual
    comparisons in ``_episode_identity_matches`` from each other.
    """
    from mcap.writer import CompressionType, Writer

    from hflow.format import METADATA_RECORD_EPISODE, METADATA_RECORD_PROVENANCE

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        # Uncompressed chunks: the payload-corruption test flips bytes inside
        # the chunk records region, which is only addressable in place when
        # the records are plaintext.
        writer = Writer(stream, compression=CompressionType.NONE)
        writer.start(profile="", library="test-lerobot-resume")
        schema_id = writer.register_schema(
            name="test.Pointer", encoding="ros2msg", data=b"int32 x\n"
        )
        channel_id = writer.register_channel(
            topic="/pointer", message_encoding="ros2msg", schema_id=schema_id
        )
        writer.add_message(channel_id, log_time=10**9, data=b"\x01\x00\x00\x00", publish_time=10**9)
        writer.add_metadata(
            METADATA_RECORD_EPISODE,
            {
                "task": f"task-{episode_index}",
                "operator": "lerobot_converter",
                "success": "true",
                "embodiment": "unknown",
                "source_dataset": dataset_source.repo_id,
                "source_revision": dataset_source.revision,
                "source_episode_index": str(episode_index),
                "converter_version": prep.CONVERTER_VERSION,
                "camera_keys": prep._encode_camera_keys(camera_keys),
                "gop_seconds": f"{prep.IMPORT_GOP_SECONDS:g}",
                **(episode_record_overrides or {}),
            },
        )
        writer.add_metadata(
            "source-provenance/v1",
            {
                "converter_version": prep.CONVERTER_VERSION,
                "ffmpeg_version": "test-ffmpeg",
                "source_uri": (f"hf://datasets/{dataset_source.repo_id}@{dataset_source.revision}"),
                **(source_provenance_overrides or {}),
            },
        )
        writer.add_metadata(
            METADATA_RECORD_PROVENANCE,
            {
                "schema_version": "1",
                "pipeline_version": "test",
                "ffmpeg_version": "test-ffmpeg",
                "gop_preset": "custom",
                "gop_seconds": f"{prep.IMPORT_GOP_SECONDS:g}",
                "marker": marker,
                **(provenance_overrides or {}),
            },
        )
        writer.finish()


_MATCHING_CAMERA_KEYS = (prep.DEFAULT_CAMERA_KEY,)
_MATCHING_SOURCE = prep.DatasetSource("fake/repo", "abc", "apache-2.0")


@pytest.mark.parametrize(
    ("episode_record_overrides", "source_provenance_overrides", "provenance_overrides"),
    [
        pytest.param({"source_dataset": "other/repo"}, None, None, id="source-dataset"),
        pytest.param({"source_revision": "deadbeef"}, None, None, id="source-revision"),
        pytest.param({"source_episode_index": "7"}, None, None, id="source-episode-index"),
        pytest.param(
            {"camera_keys": prep._encode_camera_keys(("observation.images.other",))},
            None,
            None,
            id="camera-selection",
        ),
        pytest.param({"camera_keys": "not-json"}, None, None, id="unparseable-camera-keys"),
        pytest.param({"camera_keys": '{"a": 1}'}, None, None, id="camera-keys-not-a-list"),
        pytest.param({"camera_keys": "[1, 2]"}, None, None, id="camera-keys-not-strings"),
        pytest.param(
            {"converter_version": "lerobot-converter-v5"}, None, None, id="episode-converter"
        ),
        pytest.param(
            None, {"converter_version": "lerobot-converter-v5"}, None, id="provenance-converter"
        ),
        pytest.param({"gop_seconds": "2"}, None, None, id="episode-gop"),
        pytest.param(None, None, {"gop_seconds": "2"}, id="transform-gop"),
    ],
)
def test_reuse_refuses_a_landing_episode_differing_in_one_identity_field(
    tmp_path: Path,
    episode_record_overrides: dict[str, str] | None,
    source_provenance_overrides: dict[str, str] | None,
    provenance_overrides: dict[str, str] | None,
) -> None:
    """Each identity comparison, on its own.

    The import-level mismatch test differs in two fields at once, so any one
    comparison still catches it and the other five carry no weight. Reuse is
    the direction where trusting too much is dangerous: a landing file from
    another revision or another camera selection served as completed work is
    wrong data delivered silently, so each field earns its own case.
    """
    data_root = LocalStorageRoot(tmp_path / "out")
    landing = tmp_path / "out" / "landing" / "lerobot_episode_0001.mcap"
    _write_identity_matching_landing_mcap(
        landing,
        dataset_source=_MATCHING_SOURCE,
        episode_index=0,
        camera_keys=_MATCHING_CAMERA_KEYS,
        marker="one-field-off",
        episode_record_overrides=episode_record_overrides,
        source_provenance_overrides=source_provenance_overrides,
        provenance_overrides=provenance_overrides,
    )

    assert (
        prep._try_reuse_completed_episode(
            data_root,
            dataset_source=_MATCHING_SOURCE,
            episode_index=0,
            camera_keys=_MATCHING_CAMERA_KEYS,
        )
        is None
    )


def test_reuse_accepts_the_landing_episode_the_overrides_are_measured_against(
    tmp_path: Path,
) -> None:
    """The control: without an override the same fixture is reused.

    Without this, every case above could pass because the fixture never
    matches at all rather than because the one changed field was compared.
    """
    data_root = LocalStorageRoot(tmp_path / "out")
    landing = tmp_path / "out" / "landing" / "lerobot_episode_0001.mcap"
    _write_identity_matching_landing_mcap(
        landing,
        dataset_source=_MATCHING_SOURCE,
        episode_index=0,
        camera_keys=_MATCHING_CAMERA_KEYS,
        marker="all-fields-matching",
    )

    reused = prep._try_reuse_completed_episode(
        data_root,
        dataset_source=_MATCHING_SOURCE,
        episode_index=0,
        camera_keys=_MATCHING_CAMERA_KEYS,
    )

    assert reused is not None
    assert reused["uri"] == data_root.uri_for("landing/lerobot_episode_0001.mcap")
    assert reused["content_id"] == prep.content_episode_id(landing)
    assert reused["size_bytes"] == landing.stat().st_size


def test_reuse_refuses_an_empty_landing_episode(tmp_path: Path) -> None:
    """A zero-byte landing file is an interrupted publish, not completed work.

    Removing both ``< 1`` size checks leaves this passing: an empty file is
    not a readable MCAP, so the reader refusal already covers it. The size
    check before ``storage.fetch`` still earns its place by not downloading a
    zero-byte object to learn that, but it is not what this test holds.
    """
    data_root = LocalStorageRoot(tmp_path / "out")
    landing = tmp_path / "out" / "landing" / "lerobot_episode_0001.mcap"
    landing.parent.mkdir(parents=True, exist_ok=True)
    landing.write_bytes(b"")

    assert (
        prep._try_reuse_completed_episode(
            data_root,
            dataset_source=_MATCHING_SOURCE,
            episode_index=0,
            camera_keys=_MATCHING_CAMERA_KEYS,
        )
        is None
    )


def _ensure_two_episode_archive(
    dataset_source: prep.DatasetSource, cache_dir: Path
) -> prep._SourceArchive:
    archive = _stub_single_episode_source_archive(dataset_source, cache_dir)
    return replace(
        archive,
        episodes=(
            archive.episodes[0],
            replace(archive.episodes[0], episode_index=1, task="second"),
        ),
    )


def test_import_resumes_after_mid_batch_failure_without_rewriting_completed_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "out"
    camera_keys = (prep.DEFAULT_CAMERA_KEY,)
    convert_calls: list[int] = []

    def convert_or_fail(
        *,
        source_archive: object,
        dataset_source: prep.DatasetSource,
        storage: StorageRoot,
        episode_index: int,
        camera_keys: tuple[str, ...],
        numeric_schemas: object,
        frames_per_second: object,
    ) -> prep._PublishedEpisode:
        del source_archive, numeric_schemas, frames_per_second
        convert_calls.append(episode_index)
        if episode_index == 1 and convert_calls.count(1) == 1:
            raise RuntimeError("forced mid-batch failure")
        staged = tmp_path / f"staged-{episode_index}-{len(convert_calls)}.mcap"
        _write_identity_matching_landing_mcap(
            staged,
            dataset_source=dataset_source,
            episode_index=episode_index,
            camera_keys=camera_keys,
            marker=f"episode-{episode_index}-bytes",
        )
        return publish_staged_episode(storage, staged, episode_index)

    stub_hub_repo_info(monkeypatch, resolved_sha="abc")
    monkeypatch.setattr(prep, "_ensure_source_archive", _ensure_two_episode_archive)
    monkeypatch.setattr(prep, "_convert_single_episode", convert_or_fail)

    with pytest.raises(RuntimeError, match="forced mid-batch failure"):
        prep.import_lerobot_dataset(
            dataset_repo="fake/repo",
            revision="main",
            output_dir=output_dir,
            camera_keys=camera_keys,
        )

    first_episode_path = output_dir / "landing" / "lerobot_episode_0001.mcap"
    assert first_episode_path.is_file()
    assert not (output_dir / "prepared-manifest.json").exists()
    first_episode_bytes = first_episode_path.read_bytes()
    assert convert_calls == [0, 1]

    episode_uris = prep.import_lerobot_dataset(
        dataset_repo="fake/repo",
        revision="main",
        output_dir=output_dir,
        camera_keys=camera_keys,
    )

    assert convert_calls == [0, 1, 1]
    assert first_episode_path.read_bytes() == first_episode_bytes
    assert episode_uris == [
        str(first_episode_path.resolve()),
        str((output_dir / "landing" / "lerobot_episode_0002.mcap").resolve()),
    ]
    manifest = json.loads((output_dir / "prepared-manifest.json").read_text())
    assert manifest["schema_version"] == 3
    assert manifest["episodes_converted"] == 1
    assert len(manifest["episodes"]) == 2
    assert [entry["uri"] for entry in manifest["episodes"]] == episode_uris
    assert all(len(entry["content_id"]) == 16 for entry in manifest["episodes"])
    assert all(entry["size_bytes"] > 0 for entry in manifest["episodes"])


def test_import_does_not_reuse_identity_mismatched_landing_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "out"
    landing = output_dir / "landing" / "lerobot_episode_0001.mcap"
    mismatched_source = prep.DatasetSource("other/repo", "deadbeef", "apache-2.0")
    _write_identity_matching_landing_mcap(
        landing,
        dataset_source=mismatched_source,
        episode_index=0,
        camera_keys=(prep.DEFAULT_CAMERA_KEY,),
        marker="wrong-identity",
    )
    original_bytes = landing.read_bytes()
    convert_calls = 0

    def convert_replacement(
        *,
        source_archive: object,
        dataset_source: prep.DatasetSource,
        storage: StorageRoot,
        episode_index: int,
        camera_keys: tuple[str, ...],
        numeric_schemas: object,
        frames_per_second: object,
    ) -> prep._PublishedEpisode:
        nonlocal convert_calls
        del source_archive, numeric_schemas, frames_per_second
        convert_calls += 1
        staged = tmp_path / f"replacement-{episode_index}.mcap"
        _write_identity_matching_landing_mcap(
            staged,
            dataset_source=dataset_source,
            episode_index=episode_index,
            camera_keys=camera_keys,
            marker="replacement",
        )
        return publish_staged_episode(storage, staged, episode_index)

    stub_hub_repo_info(monkeypatch, resolved_sha="abc")
    monkeypatch.setattr(prep, "_ensure_source_archive", _stub_single_episode_source_archive)
    monkeypatch.setattr(prep, "_convert_single_episode", convert_replacement)

    episode_uris = prep.import_lerobot_dataset(
        dataset_repo="fake/repo",
        revision="main",
        output_dir=output_dir,
        episode_index=0,
    )

    assert convert_calls == 1
    assert landing.read_bytes() != original_bytes
    assert episode_uris == [str(landing.resolve())]
    manifest = json.loads((output_dir / "prepared-manifest.json").read_text())
    assert manifest["episodes_converted"] == 1


def test_import_full_reuse_reports_zero_episodes_converted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "out"
    camera_keys = (prep.DEFAULT_CAMERA_KEY,)
    dataset_source = prep.DatasetSource("fake/repo", "abc", "apache-2.0")
    for episode_index in (0, 1):
        _write_identity_matching_landing_mcap(
            output_dir / "landing" / f"lerobot_episode_{episode_index + 1:04d}.mcap",
            dataset_source=dataset_source,
            episode_index=episode_index,
            camera_keys=camera_keys,
            marker=f"already-{episode_index}",
        )

    convert_calls = 0

    def should_not_convert(**_kwargs: object) -> prep._PublishedEpisode:
        nonlocal convert_calls
        convert_calls += 1
        raise AssertionError("matching landing episodes must be reused")

    stub_hub_repo_info(monkeypatch, resolved_sha="abc")
    monkeypatch.setattr(prep, "_ensure_source_archive", _ensure_two_episode_archive)
    monkeypatch.setattr(prep, "_convert_single_episode", should_not_convert)

    episode_uris = prep.import_lerobot_dataset(
        dataset_repo="fake/repo",
        revision="main",
        output_dir=output_dir,
        camera_keys=camera_keys,
    )

    assert convert_calls == 0
    assert len(episode_uris) == 2
    manifest = json.loads((output_dir / "prepared-manifest.json").read_text())
    assert manifest["episodes_converted"] == 0
    assert len(manifest["episodes"]) == 2


# --- success label: read the collector's outcome, never invent it (#395) -----


def _build_success_label_corpus(
    root: Path, outcome_mode: str, frames_per_second: int | float = 30
) -> dict:
    """One two-frame episode.

    outcome_mode: 'transition', 'all-false', 'empty-aggregate', or 'none'.
    """
    has_outcome = outcome_mode != "none"
    info = {
        "fps": frames_per_second,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [1]},
            "observation.state": {"dtype": "float32", "shape": [1]},
            "observation.images.up": {"dtype": "video", "shape": [480, 640, 3]},
            "timestamp": {"dtype": "float32", "shape": [1]},
        },
        "robot_type": "so101",
    }
    if has_outcome:
        info["features"]["next.success"] = {"dtype": "bool", "shape": [1]}
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(json.dumps(info))

    import duckdb

    conn = duckdb.connect()
    ep_cols = [
        "episode_index",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
        "videos/observation.images.up/chunk_index",
        "videos/observation.images.up/file_index",
        "videos/observation.images.up/from_timestamp",
        "videos/observation.images.up/to_timestamp",
        "tasks",
    ]
    row: list[object] = [0, 2, "000", "000", 0, 2, "000", "000", 0.0, 0.0, ["push the block"]]
    if has_outcome:
        if outcome_mode == "transition":
            stats_min, stats_max = [False], [True]
        elif outcome_mode == "empty-aggregate":
            # The column exists but carries no value for this episode, which
            # is a declared feature with nothing recorded rather than a label.
            stats_min, stats_max = [], []
        else:
            stats_min, stats_max = [False], [False]
        ep_cols += ["stats/next.success/min", "stats/next.success/max"]
        row += [stats_min, stats_max]
    ep_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ep_path.parent.mkdir(parents=True, exist_ok=True)
    vals = (
        "("
        + ",".join(
            "[" + ",".join(str(bool(item)) for item in value) + "]"
            if isinstance(value, list) and value and all(isinstance(item, bool) for item in value)
            else "[" + ",".join(f"'{item}'" for item in value) + "]"
            if isinstance(value, list)
            else f"'{value}'"
            if isinstance(value, str)
            else str(value)
            for value in row
        )
        + ")"
    )
    conn.execute(
        f"COPY (SELECT * FROM (VALUES {vals}) AS t({','.join(chr(34) + c + chr(34) for c in ep_cols)})) "
        f"TO '{str(ep_path).replace(chr(39), chr(39) * 2)}' (FORMAT parquet)"
    )

    frame_outcomes = [False, True] if outcome_mode == "transition" else [False, False]
    data_cols = 'index, episode_index, frame_index, timestamp, "observation.state", action'
    data_rows = [
        f"({index}, 0, {frame_index}, 0.0, [0.0], [0.5]"
        for index, frame_index in enumerate(range(2))
    ]
    if has_outcome:
        data_cols += ', "next.success"'
        data_rows = [
            data_row + f", {str(frame_outcomes[frame_index]).lower()})"
            for frame_index, data_row in enumerate(data_rows)
        ]
    else:
        data_rows = [data_row + ")" for data_row in data_rows]
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    conn.execute(
        f"COPY (SELECT * FROM (VALUES {','.join(data_rows)}) AS t({data_cols})) "
        f"TO '{str(data_path).replace(chr(39), chr(39) * 2)}' (FORMAT parquet)"
    )
    conn.close()
    return {"info": info}


def _import_success_label_corpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome_mode: str,
    frames_per_second: int | float = 30,
    transcode_calls: list[float] | None = None,
) -> Path:
    root = tmp_path / "corpus"
    corpus = _build_success_label_corpus(root, outcome_mode, frames_per_second)
    output_dir = tmp_path / "out"

    stub_hub_repo_info(monkeypatch, resolved_sha="abc1234")
    stub_single_shard_hub_corpus(monkeypatch, root, corpus["info"])

    def fake_transcode(mp4_path: Path, gop: float, fps: float) -> list[bytes]:
        if transcode_calls is not None:
            transcode_calls.append(fps)
        return [
            b"\x00\x00\x00\x01\x09\x10\x00\x00\x00\x01\x67\x42\x00"
            b"\x00\x00\x00\x01\x68\x88\x80\x00\x00\x00\x01\x65\x88"
        ] * 2

    monkeypatch.setattr(prep, "_transcode_mp4_to_h264", fake_transcode)
    monkeypatch.setattr(prep, "_get_video_pts_times", lambda path: [0, 0])
    monkeypatch.setattr(prep, "ffmpeg_version", lambda: "test-ffmpeg")

    prep.import_lerobot_dataset(
        dataset_repo="fake/repo",
        output_dir=output_dir,
        camera_keys=("observation.images.up",),
    )
    return output_dir


def test_success_label_reports_max_over_episode_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MAX over the collector's next.success frames: a False frame followed
    by a True frame makes the episode a success, even though the LAST frame
    is False. The derivation is stamped so the methodology travels."""
    from hflow.episode import Episode

    output_dir = _import_success_label_corpus(tmp_path, monkeypatch, "transition")
    landing = sorted((output_dir / "landing").glob("*.mcap"))
    with Episode(landing[0]) as episode:
        record = episode.metadata_records["episode/v1"]
    assert record["success"] == "true"
    assert record["success_derivation"] == "max(stats/next.success)"


def test_success_label_reports_false_when_source_is_all_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An all-false source label ships as 'false', never as an invented
    'true': the collector's judgment, reported verbatim."""
    from hflow.episode import Episode

    output_dir = _import_success_label_corpus(tmp_path, monkeypatch, "all-false")
    landing = sorted((output_dir / "landing").glob("*.mcap"))
    with Episode(landing[0]) as episode:
        record = episode.metadata_records["episode/v1"]
    assert record["success"] == "false"
    assert record["success_derivation"] == "max(stats/next.success)"


def test_success_label_omitted_when_the_outcome_aggregate_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A declared outcome feature with nothing recorded is not a label.

    Dropping the length check stamps ``success: "false"`` here, because
    ``any([])`` is False. That is the same invention the hardcoded ``"true"``
    was, one value over, so the empty aggregate needs its own case rather than
    riding on the no-feature one.
    """
    from hflow.episode import Episode

    output_dir = _import_success_label_corpus(tmp_path, monkeypatch, "empty-aggregate")
    landing = sorted((output_dir / "landing").glob("*.mcap"))
    with Episode(landing[0]) as episode:
        record = episode.metadata_records["episode/v1"]
    assert "success" not in record
    assert "success_derivation" not in record


def test_success_label_omitted_when_source_has_no_outcome_feature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corpus without the outcome feature is normal, not malformed: the key
    is omitted (never substituted), the import succeeds, and the catalog
    promotion renders the omitted key as SQL NULL (catalog.py:886)."""
    import duckdb

    from hflow.catalog import Catalog
    from hflow.episode import Episode
    from hflow.transform import stamps_from_provenance

    output_dir = _import_success_label_corpus(tmp_path, monkeypatch, "none")
    landing = sorted((output_dir / "landing").glob("*.mcap"))
    with Episode(landing[0]) as episode:
        record = episode.metadata_records["episode/v1"]
        assert "success" not in record
        assert "success_derivation" not in record

        catalog_root = tmp_path / "catalog"
        catalog = Catalog(catalog_root)
        catalog.append_episode(
            canonical_path=landing[0],
            stamps=stamps_from_provenance(episode.metadata),
            episode_metadata=dict(episode.metadata),
            check_rows=[],
        )

    rows = duckdb.sql(
        f"SELECT success FROM read_parquet('{catalog_root / 'episodes' / '*.parquet'}')"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] is None


def test_converter_version_bumped_with_the_label_support() -> None:
    """The converter version moves with any change to the published bytes.

    The label changed episode/v1, which content_episode_id hashes. Reading a
    fractional fps as declared moves every message log time. The frame-exact
    window extraction also changes canonical bytes for windowed corpora.
    Reuse keys on this stamp, so a version that lags a byte change makes
    stale output look like completed work.
    """
    assert prep.CONVERTER_VERSION == "lerobot-converter-v9"


def test_reuse_refuses_a_landing_episode_with_damaged_payload(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Payload damage that leaves the MCAP structure and metadata intact is
    still damaged work: the reuse path CRC-validates before stamping the
    receipt, and refuses rather than laundering the bytes (#426)."""
    from reuse_test_helpers import flip_chunk_payload_bytes

    data_root = LocalStorageRoot(tmp_path / "out")
    landing = tmp_path / "out" / "landing" / "lerobot_episode_0001.mcap"
    _write_identity_matching_landing_mcap(
        landing,
        dataset_source=_MATCHING_SOURCE,
        episode_index=0,
        camera_keys=_MATCHING_CAMERA_KEYS,
        marker="payload-damaged",
    )
    receipt_before = prep._try_reuse_completed_episode(
        data_root,
        dataset_source=_MATCHING_SOURCE,
        episode_index=0,
        camera_keys=_MATCHING_CAMERA_KEYS,
    )
    assert receipt_before is not None
    intact_content_id = receipt_before["content_id"]

    flip_chunk_payload_bytes(landing)

    with caplog.at_level(logging.WARNING, logger="hflow.importers.lerobot"):
        assert (
            prep._try_reuse_completed_episode(
                data_root,
                dataset_source=_MATCHING_SOURCE,
                episode_index=0,
                camera_keys=_MATCHING_CAMERA_KEYS,
            )
            is None
        )
    # Silent re-conversion would repair the damage and hide it. The operator
    # gets told which file failed, so bit rot in a landing tree is visible
    # rather than absorbed by the next successful import.
    warning_messages = [record.getMessage() for record in caplog.records]
    assert any("failed CRC validation" in message for message in warning_messages), warning_messages
    assert any(str(landing) in message for message in warning_messages), warning_messages
    # The damaged bytes must not be laundered: the damaged file's hash
    # differs from the receipt the intact file earned.
    from reuse_test_helpers import content_id_differs_from_delivery_receipt

    assert content_id_differs_from_delivery_receipt(landing, intact_content_id)


def test_fractional_fps_sets_the_log_times_from_the_declared_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """29.97 is the rate NTSC-derived capture writes, and it must reach the
    time axis unfloored.

    Frame n sits at n / fps seconds. At 29.97 the second frame is 33.3667 ms
    in; read as 29 it lands at 34.4828 ms, and the error grows with the frame
    index: about a second by frame 900, six seconds by frame 5400.
    """
    from hflow.episode import Episode

    output_dir = _import_success_label_corpus(
        tmp_path, monkeypatch, "none", frames_per_second=29.97
    )
    landing = sorted((output_dir / "landing").glob("*.mcap"))
    with Episode(landing[0]) as episode:
        log_times = list(episode.channel("/action").timestamps)

    expected_second_frame = prep.EPISODE_START_TIME_NS + round(prep.NANOSECONDS_PER_SECOND / 29.97)
    floored = prep.EPISODE_START_TIME_NS + round(prep.NANOSECONDS_PER_SECOND / 29)
    assert log_times[0] == prep.EPISODE_START_TIME_NS
    assert log_times[1] == expected_second_frame
    assert log_times[1] != floored


def test_fractional_fps_reaches_the_transcoder_for_the_keyframe_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The GOP is the half that makes provenance false rather than imprecise.

    _transcode_mp4_to_h264 sets the keyframe interval to
    ``round(gop_seconds * frames_per_second)``. At the declared 29.97 that is
    30, at the floored 29 it is 29, so the file carried a 29 frame GOP while
    provenance/v1 stamped gop_seconds as 1. #376 is open because that field is
    recorded as actually used and never checked; this was one way it could
    already be wrong.
    """
    transcode_calls: list[float] = []
    _import_success_label_corpus(
        tmp_path,
        monkeypatch,
        "none",
        frames_per_second=29.97,
        transcode_calls=transcode_calls,
    )

    assert transcode_calls, "the transcoder was never called"
    assert transcode_calls[0] == 29.97
    keyframe_interval = max(1, round(prep.IMPORT_GOP_SECONDS * transcode_calls[0]))
    assert keyframe_interval == 30
    assert keyframe_interval != max(1, round(prep.IMPORT_GOP_SECONDS * 29))


def test_hub_source_metadata_paginates_pinned_tree_and_reuses_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "repository"
    _build_fake_corpus(source_root)
    first_shard = source_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    second_shard = source_root / "meta" / "episodes" / "chunk-001" / "file-000.parquet"
    split_episode_metadata_into_shards(
        source_root,
        {
            "meta/episodes/chunk-000/file-000.parquet": (0, 1),
            "meta/episodes/chunk-001/file-000.parquet": (2, 3),
        },
    )

    resolved_revision = "a" * 40
    downloaded_filenames: list[str] = []
    metadata_tree_requests: list[str] = []

    class HubHandler(BaseHTTPRequestHandler):
        def do_HEAD(self) -> None:
            self.respond(include_body=False)

        def do_GET(self) -> None:
            self.respond(include_body=True)

        def log_message(self, format: str, *args: object) -> None:
            pass

        def respond(self, *, include_body: bool) -> None:
            request_url = urlsplit(self.path)
            request_path = unquote(request_url.path)
            metadata_tree_path = f"/api/datasets/fake/repo/tree/{resolved_revision}/meta/episodes"
            if request_path == metadata_tree_path:
                metadata_tree_requests.append(self.path)
                second_page = "cursor=second" in request_url.query
                if second_page:
                    entries = [
                        {
                            "type": "file",
                            "path": "meta/episodes/chunk-000/file-000.parquet",
                            "size": first_shard.stat().st_size,
                            "oid": "1" * 40,
                        },
                        {
                            "type": "file",
                            "path": "meta/episodes/chunk-001/file-000.parquet",
                            "size": second_shard.stat().st_size,
                            "oid": "2" * 40,
                        },
                    ]
                else:
                    entries = [
                        {
                            "type": "file",
                            "path": "meta/episodes/chunk-001/file-000.parquet",
                            "size": second_shard.stat().st_size,
                            "oid": "2" * 40,
                        },
                        {
                            "type": "directory",
                            "path": "meta/episodes/chunk-001",
                            "oid": "3" * 40,
                        },
                        {
                            "type": "file",
                            "path": "meta/episodes/README.txt",
                            "size": 0,
                            "oid": "4" * 40,
                        },
                    ]
                payload = json.dumps(entries).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                if not second_page:
                    self.send_header(
                        "Link",
                        f'<http://{self.headers["Host"]}{metadata_tree_path}?cursor=second>; rel="next"',
                    )
                self.end_headers()
                if include_body:
                    self.wfile.write(payload)
                return
            if request_path in {
                "/api/datasets/fake/repo/revision/main",
                f"/api/datasets/fake/repo/revision/{resolved_revision}",
            }:
                payload = json.dumps(
                    {
                        "id": "fake/repo",
                        "sha": resolved_revision,
                        "cardData": {"license": "apache-2.0"},
                        "siblings": [],
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                if include_body:
                    self.wfile.write(payload)
                return
            pinned_prefix = f"/datasets/fake/repo/resolve/{resolved_revision}/"
            if not request_path.startswith(pinned_prefix):
                self.send_error(404)
                return
            filename = request_path.removeprefix(pinned_prefix)
            source_file = source_root / filename
            if not source_file.is_file():
                self.send_error(404)
                return
            payload = source_file.read_bytes()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-Repo-Commit", resolved_revision)
            self.send_header("ETag", hashlib.sha256(payload).hexdigest())
            self.end_headers()
            if include_body:
                downloaded_filenames.append(filename)
                self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), HubHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setattr(prep, "HfApi", partial(prep.HfApi, endpoint=endpoint, token=False))
    monkeypatch.setattr(
        prep, "hf_hub_download", partial(prep.hf_hub_download, endpoint=endpoint, token=False)
    )
    try:
        repository_info = prep._hf_repo_info("fake/repo", "main")
        source = prep.DatasetSource(
            repo_id="fake/repo", revision=repository_info["sha"], license=repository_info["license"]
        )
        cache_directory = tmp_path / "output" / "_lerobot_cache" / source.revision
        archive = prep._ensure_source_archive(source, cache_directory)
        repeated_archive = prep._ensure_source_archive(source, cache_directory)
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()

    assert source.revision == resolved_revision
    assert source.license == "apache-2.0"
    assert [episode.length for episode in archive.episodes] == [60, 65, 70, 75]
    assert repeated_archive == archive
    assert downloaded_filenames == [
        "meta/info.json",
        "meta/episodes/chunk-000/file-000.parquet",
        "meta/episodes/chunk-001/file-000.parquet",
    ]
    assert len(metadata_tree_requests) == 4
    assert all(
        f"/tree/{resolved_revision}/meta/episodes" in unquote(request)
        for request in metadata_tree_requests
    )
    assert sum("cursor=second" in request for request in metadata_tree_requests) == 2
    assert not (cache_directory / "videos").exists()
    assert json.loads((cache_directory / "meta/info.json").read_text())["fps"] == 30


@pytest.mark.parametrize("invalid_payload", [b"\xff\xfe", b"{broken"])
def test_import_refuses_invalid_downloaded_metadata_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid_payload: bytes
) -> None:
    metadata_file = tmp_path / "downloaded-info.json"
    metadata_file.write_bytes(invalid_payload)
    monkeypatch.setattr(prep, "_hf_repo_info", lambda *_args: {"sha": "a" * 40, "license": "mit"})
    monkeypatch.setattr(prep, "hf_hub_download", lambda *_args, **_kwargs: str(metadata_file))
    output = tmp_path / "output"
    with pytest.raises(ValueError, match=r"meta/info\.json"):
        prep.import_lerobot_dataset(output_dir=output)
    assert not (output / "landing").exists()
    assert not (output / "prepared-manifest.json").exists()


@pytest.mark.parametrize(
    "filename",
    [
        "../outside",
        "/absolute",
        "data/../../outside",
        "meta/episodes/../../outside.parquet",
        "data\\outside",
    ],
)
def test_source_download_refuses_paths_outside_its_cache(tmp_path: Path, filename: str) -> None:
    with pytest.raises(ValueError, match="inside the source cache"):
        prep._download_file("fake/repo", "a" * 40, filename, tmp_path / "cache")


def test_source_download_refuses_symlink_outside_its_cache(tmp_path: Path) -> None:
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()
    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()
    (cache_directory / "data").symlink_to(outside_directory, target_is_directory=True)
    with pytest.raises(ValueError, match="inside the source cache"):
        prep._download_file("fake/repo", "a" * 40, "data/file.parquet", cache_directory)
    assert list(outside_directory.iterdir()) == []
