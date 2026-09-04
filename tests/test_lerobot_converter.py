"""Outcome-focused coverage for first-class LeRobot Dataset v3 import.

These tests exercise the metadata-driven discovery and fail-loud behavior
with a synthetic v3-style corpus, without asserting third-party
implementation details or touching the network.
"""

import io
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import cast

import pytest

import hflow.importers.lerobot as prep
from hflow.cli import main as cli_main
from hflow.reader import open_reader
from hflow.storage import LocalStorageRoot, StorageRoot

_DERIVE = prep._derive_numeric_schema
_ENCODE = prep._encode_cdr_float32_array

_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")

# Only the remux test below shells out. Scoping this to the module would skip
# the five metadata tests too, and none of those touch ffmpeg at all.
_requires_system_ffmpeg = pytest.mark.skipif(
    _FFMPEG is None or _FFPROBE is None,
    reason="system ffmpeg/ffprobe required to construct and inspect the test video",
)


def _build_fake_corpus(tmp_path: Path) -> dict:
    """Synthetic v3 metadata: 4 episodes, 2 cameras, 6-dim state/action."""
    info = {
        "fps": 30,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [6]},
            "observation.state": {"dtype": "float32", "shape": [6]},
            "observation.images.up": {"dtype": "video", "shape": [480, 640, 3]},
            "observation.images.side": {"dtype": "video", "shape": [480, 640, 3]},
            "timestamp": {"dtype": "float32", "shape": [1]},
        },
        "robot_type": "so101",
    }
    (tmp_path / "meta").mkdir(parents=True, exist_ok=True)
    (tmp_path / "meta" / "info.json").write_text(json.dumps(info))

    import duckdb

    conn = duckdb.connect()
    rows = []
    for i in range(4):
        rows.append(
            [
                i,
                60 + i * 5,
                "chunk-000",
                "file-000",
                i * 200,
                i * 200 + 60 + i * 5,
                "chunk-000",
                "file-000",
                0.0,
                2.0 + i * 0.2,
                "chunk-000",
                "file-000",
                0.0,
                2.0 + i * 0.2,
                [f"task-{i}"],
            ]
        )
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
        "videos/observation.images.side/chunk_index",
        "videos/observation.images.side/file_index",
        "videos/observation.images.side/from_timestamp",
        "videos/observation.images.side/to_timestamp",
        "tasks",
    ]
    ep_path = tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ep_path.parent.mkdir(parents=True, exist_ok=True)
    ep_quoted = str(ep_path).replace("'", "''")
    vals_sql = ",".join(
        "("
        + ",".join(
            "[" + ",".join(f"'{x}'" for x in v) + "]"
            if isinstance(v, list)
            else f"'{v!s}'"
            if isinstance(v, str)
            else str(v)
            for v in row
        )
        + ")"
        for row in rows
    )
    ep_cols_q = ",".join(f'"{c}"' for c in ep_cols)
    conn.execute(
        f"COPY (SELECT * FROM (VALUES {vals_sql}) AS t({ep_cols_q})) "
        f"TO '{ep_quoted}' (FORMAT parquet)"
    )

    # data parquet with contiguous `index` column matching episode windows
    data_rows = []
    idx = 0
    for i in range(4):
        length = 60 + i * 5
        for f in range(length):
            state = "[" + ",".join(str(float(f)) for _ in range(6)) + "]"
            action = "[" + ",".join(str(float(f + 0.5)) for _ in range(6)) + "]"
            data_rows.append([idx, i, f, round(f / 30.0, 6), state, action])
            idx += 1
    data_path = tmp_path / "data" / "chunk-000" / "file-000.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_quoted = str(data_path).replace("'", "''")
    data_vals = ",".join("(" + ",".join(str(v) for v in row) + ")" for row in data_rows)
    conn.execute(
        f"COPY (SELECT * FROM (VALUES {data_vals}) AS "
        't(index, episode_index, frame_index, timestamp, "observation.state", action)) '
        f"TO '{data_quoted}' (FORMAT parquet)"
    )
    conn.close()

    return {
        "info": info,
        "fps": 30,
        "data_path": info["data_path"],
        "video_path": info["video_path"],
        "cache_dir": tmp_path,
        "data_chunk": "chunk-000",
        "data_file": "file-000",
    }


def test_derive_numeric_schema_float32_vector() -> None:
    schema = _DERIVE("observation.state", {"dtype": "float32", "shape": [6]})
    assert schema.name == "observation.state"
    assert schema.dim == 6


def test_derive_numeric_schema_rejects_unsupported() -> None:
    with pytest.raises(ValueError, match="unsupported feature"):
        _DERIVE("action", {"dtype": "float64", "shape": [6]})
    with pytest.raises(ValueError, match="unsupported feature"):
        _DERIVE("observation.state", {"dtype": "float32", "shape": [2, 3]})
    with pytest.raises(ValueError, match="unsupported feature"):
        _DERIVE("observation.state", {"dtype": "float32", "shape": []})
    for shape in ([True], [False]):
        with pytest.raises(
            ValueError,
            match=rf"unsupported feature action: dtype=float32, shape=\[{shape[0]}\]",
        ):
            _DERIVE("action", {"dtype": "float32", "shape": shape})


def test_import_rejects_required_boolean_dimension_without_dataset_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "out"
    dataset_source = prep.DatasetSource(repo_id="fake/repo", revision="abc", license="apache-2.0")
    monkeypatch.setattr(
        prep,
        "_hf_repo_info",
        lambda repo, revision: {"sha": "abc", "license": "apache-2.0"},
    )
    monkeypatch.setattr(
        prep,
        "_ensure_source_archive",
        lambda source, cache_dir: {
            "info": {
                "features": {
                    prep.DEFAULT_CAMERA_KEY: {
                        "dtype": "video",
                        "shape": [480, 640, 3],
                        "info": {"is_depth_map": False},
                    }
                }
            },
            "numeric_features": {
                "action": {"dtype": "float32", "shape": [True]},
                "observation.state": {"dtype": "float32", "shape": [6]},
            },
            "video_keys": [prep.DEFAULT_CAMERA_KEY],
            "episodes": [],
            "dataset": dataset_source,
        },
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
    monkeypatch.setattr(
        prep, "_hf_repo_info", lambda repo, rev: {"sha": rev, "license": "apache-2.0"}
    )
    monkeypatch.setattr(prep, "_fetch_info_json", lambda repo, rev, cache: corpus["info"])
    monkeypatch.setattr(
        prep,
        "_hf_tree",
        lambda repo, rev, path: (
            [{"path": "meta/episodes/chunk-000/file-000.parquet", "type": "file"}]
            if "episodes" in path
            else [{"path": "meta/info.json", "type": "file"}]
        ),
    )
    monkeypatch.setattr(prep, "_download_file", lambda url, dest, **kw: None)

    def fake_dl(url: str, dest: Path, **kw: object) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if "meta/episodes" in url:
            import shutil

            shutil.copy(
                str(tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"), dest
            )
        elif url.endswith("info.json"):
            import shutil

            shutil.copy(str(tmp_path / "meta" / "info.json"), dest)
        else:
            import shutil

            shutil.copy(str(tmp_path / "data" / "chunk-000" / "file-000.parquet"), dest)

    monkeypatch.setattr(prep, "_download_file", fake_dl)

    ds = prep.DatasetSource(repo_id="fake/repo", revision="abc", license="apache-2.0")
    found = prep._ensure_source_archive(ds, tmp_path)
    assert len(found["episodes"]) == 4
    assert found["episodes"][0]["length"] == 60
    assert found["episodes"][1]["length"] == 65
    assert found["episodes"][0]["data_from"] == 0
    assert found["episodes"][0]["data_to"] == 60
    assert set(found["video_keys"]) == {"observation.images.up", "observation.images.side"}
    assert found["episodes"][0]["video_windows"]["observation.images.up"][
        "to_timestamp"
    ] == pytest.approx(2.0)


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
    single_shard = tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    shard_paths = ("meta/episodes/chunk-000/file-000.parquet", second_shard_path)
    import duckdb

    conn = duckdb.connect()
    conn.execute(
        "CREATE TABLE all_episodes AS SELECT * FROM read_parquet('"
        + str(single_shard).replace("'", "''")
        + "')"
    )
    for shard_path, episode_indexes in zip(shard_paths, ((0, 1), (2, 3)), strict=True):
        shard_file = tmp_path / shard_path
        shard_file.parent.mkdir(parents=True, exist_ok=True)
        conn.execute(
            f"COPY (SELECT * FROM all_episodes WHERE episode_index IN {episode_indexes}) "
            f"TO '{str(shard_file).replace(chr(39), chr(39) * 2)}' (FORMAT parquet)"
        )
    conn.close()

    monkeypatch.setattr(
        prep, "_hf_repo_info", lambda repo, rev: {"sha": rev, "license": "apache-2.0"}
    )
    monkeypatch.setattr(prep, "_fetch_info_json", lambda repo, rev, cache: corpus["info"])
    monkeypatch.setattr(
        prep,
        "_hf_tree",
        lambda repo, rev, path: (
            [{"path": shard_path, "type": "file"} for shard_path in shard_paths]
            if "episodes" in path
            else [{"path": "meta/info.json", "type": "file"}]
        ),
    )
    downloaded_destinations: dict[str, Path] = {}

    def fake_download(url: str, dest: Path, **kw: object) -> None:
        relative_path = url.split("/resolve/abc/", 1)[1]
        assert relative_path not in downloaded_destinations, f"downloaded twice: {url}"
        downloaded_destinations[relative_path] = dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(tmp_path / relative_path, dest)

    monkeypatch.setattr(prep, "_download_file", fake_download)

    ds = prep.DatasetSource(repo_id="fake/repo", revision="abc", license="apache-2.0")
    cache_dir = tmp_path / "cache"
    found = prep._ensure_source_archive(ds, cache_dir)
    # A second discovery reuses each shard's own cache entry instead of
    # re-downloading or colliding with the other shard.
    prep._ensure_source_archive(ds, cache_dir)

    assert set(downloaded_destinations) == set(shard_paths)
    assert len(set(downloaded_destinations.values())) == len(shard_paths)
    assert [episode["episode_index"] for episode in found["episodes"]] == [0, 1, 2, 3]
    assert set(found["video_keys"]) == {"observation.images.up", "observation.images.side"}
    for episode in found["episodes"]:
        episode_index = episode["episode_index"]
        assert episode["length"] == 60 + episode_index * 5
        for camera_key in found["video_keys"]:
            window = episode["video_windows"][camera_key]
            assert window["chunk_index"] == "chunk-000"
            assert window["to_timestamp"] == pytest.approx(2.0 + episode_index * 0.2)


@pytest.mark.parametrize(
    "tree_entry_path",
    [
        "meta/episodes/../../data/chunk-000/file-000.parquet",
        "/tmp/probe-293-absolute.parquet",
        "meta/episodes",
        "meta/other/file-000.parquet",
    ],
)
def test_episode_metadata_cache_path_refuses_entries_outside_the_metadata_tree(
    tmp_path: Path, tree_entry_path: str
) -> None:
    with pytest.raises(ValueError, match="not a file below meta/episodes/"):
        prep._episode_metadata_cache_path(tmp_path, tree_entry_path, repo_id="fake/repo")


def test_video_cache_distinguishes_file_indices_and_reuses_same_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _build_fake_corpus(tmp_path)
    camera_key = "observation.images.up"
    dataset_source = prep.DatasetSource(repo_id="fake/repo", revision="abc", license="apache-2.0")

    def episode_row(episode_index: int, video_file_index: int) -> dict:
        return {
            "episode_index": episode_index,
            "task": f"task-{episode_index}",
            "length": 1,
            "data_chunk": "000",
            "data_file": "000",
            "data_from": 0,
            "data_to": 1,
            "video_windows": {
                camera_key: {
                    "chunk_index": "000",
                    "file_index": f"{video_file_index:03d}",
                    "from_timestamp": 0.0,
                    "to_timestamp": 0.0,
                }
            },
        }

    source_archive = cast(
        prep._SourceArchive,
        {
            **corpus,
            "episodes": [episode_row(0, 0), episode_row(1, 1)],
            "video_keys": [camera_key],
        },
    )
    numeric_schemas = {
        "observation.state": prep._NumericSchema(name="observation.state", dim=6),
        "action": prep._NumericSchema(name="action", dim=6),
    }
    video_downloaded_urls: set[str] = set()
    cache_path_by_url: dict[str, Path] = {}
    converted_sources: list[bytes] = []

    def fake_download(url: str, destination_path: Path, **_kwargs: object) -> None:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if "/videos/" in url:
            if url in video_downloaded_urls:
                raise AssertionError(f"source was downloaded twice: {url}")
            video_downloaded_urls.add(url)
            cache_path_by_url[url] = destination_path
            destination_path.write_bytes(url.encode())
            return
        shutil.copy(tmp_path / "data" / "chunk-000" / "file-000.parquet", destination_path)

    def fake_transcode(mp4_path: Path, *_args: object, **_kwargs: object) -> list[bytes]:
        converted_sources.append(mp4_path.read_bytes())
        return [b"access-unit"]

    monkeypatch.setattr(prep, "_download_file", fake_download)
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
            numeric_schemas=numeric_schemas,
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

    video_urls = [
        "https://huggingface.co/datasets/fake/repo/resolve/abc/videos/"
        "observation.images.up/chunk-000/file-000.mp4",
        "https://huggingface.co/datasets/fake/repo/resolve/abc/videos/"
        "observation.images.up/chunk-000/file-001.mp4",
    ]
    assert converted_sources == [
        video_urls[0].encode(),
        video_urls[1].encode(),
        video_urls[0].encode(),
    ]
    assert cache_path_by_url[video_urls[0]] != cache_path_by_url[video_urls[1]]


def test_camera_selection_validates_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus = _build_fake_corpus(tmp_path)
    monkeypatch.setattr(
        prep, "_hf_repo_info", lambda repo, rev: {"sha": rev, "license": "apache-2.0"}
    )
    monkeypatch.setattr(prep, "_fetch_info_json", lambda repo, rev, cache: corpus["info"])
    monkeypatch.setattr(
        prep,
        "_hf_tree",
        lambda repo, rev, path: (
            [{"path": "meta/episodes/chunk-000/file-000.parquet", "type": "file"}]
            if "episodes" in path
            else [{"path": "meta/info.json", "type": "file"}]
        ),
    )

    def fake_dl(url: str, dest: Path, **kw: object) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        if "meta/episodes" in url:
            shutil.copy(
                str(tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"), dest
            )
        elif url.endswith("info.json"):
            shutil.copy(str(tmp_path / "meta" / "info.json"), dest)
        else:
            shutil.copy(str(tmp_path / "data" / "chunk-000" / "file-000.parquet"), dest)

    monkeypatch.setattr(prep, "_download_file", fake_dl)

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

    def fake_ensure_source_archive(dataset_source: prep.DatasetSource, cache_dir: Path) -> dict:
        cache_dir.mkdir(parents=True, exist_ok=True)
        source_marker = cache_dir / "source-marker.txt"
        if not source_marker.exists():
            source_marker.write_text(dataset_source.revision)
        cache_observations.append((dataset_source.revision, cache_dir, source_marker.read_text()))
        return {
            "info": {},
            "fps": 30,
            "data_path": "data/{chunk_index}/{file_index}.parquet",
            "video_path": "videos/{camera_key}/{chunk_index}/{file_index}.mp4",
            "episodes": [],
            "video_keys": [prep.DEFAULT_CAMERA_KEY],
            "numeric_features": {
                "action": {"dtype": "float32", "shape": [1]},
                "observation.state": {"dtype": "float32", "shape": [1]},
            },
            "cache_dir": cache_dir,
            "dataset": dataset_source,
        }

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
    response_body = json.dumps({"sha": resolved_sha, "cardData": {"license": "apache-2.0"}})
    monkeypatch.setattr(
        prep.urllib.request,
        "urlopen",
        lambda request, timeout: io.BytesIO(response_body.encode()),
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


# --- Hugging Face JSON boundary contextual errors (#301) ---------------------
#
# The LeRobot importer decodes response bytes at three HF boundaries (repo
# info, tree listings, meta/info.json). Invalid UTF-8 or malformed JSON must
# surface as a contextual ValueError naming the source, with the original
# decoder exception chained as the cause. These tests stub urlopen with local
# bytes; no network request is made.


class _StubResponse:
    """Minimal urlopen response double: a context manager with read()."""

    def __init__(self, body: bytes, headers: dict[str, str] | None = None) -> None:
        self._body = body
        self.headers = headers or {}

    def __enter__(self) -> "_StubResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


class _StubUrlopen:
    """Routes urlopen calls to fixed bytes by a marker in the request URL."""

    def __init__(
        self,
        routes: dict[str, bytes],
        response_headers: dict[str, dict[str, str]] | None = None,
        max_requests: int | None = None,
    ) -> None:
        self._routes = routes
        self._response_headers = response_headers or {}
        self._max_requests = max_requests
        self.requests: list[urllib.request.Request] = []

    def __call__(self, request: urllib.request.Request, timeout: int = 60) -> _StubResponse:
        url = request.full_url
        self.requests.append(request)
        if self._max_requests is not None and len(self.requests) > self._max_requests:
            raise AssertionError(f"urlopen exceeded test request limit: {self._max_requests}")
        for marker, body in self._routes.items():
            if marker in url:
                return _StubResponse(body, self._response_headers.get(marker))
        raise AssertionError(f"unexpected urlopen URL: {url}")


def _stub_urlopen(
    monkeypatch: pytest.MonkeyPatch,
    routes: dict[str, bytes],
    response_headers: dict[str, dict[str, str]] | None = None,
    max_requests: int | None = None,
) -> _StubUrlopen:
    stub = _StubUrlopen(routes, response_headers, max_requests)
    monkeypatch.setattr(urllib.request, "urlopen", stub)
    return stub


_INVALID_UTF8 = b"\xff\xfe\x00"
_MALFORMED_JSON = b"{not json"


def test_hf_repo_info_invalid_utf8_raises_contextual_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_urlopen(monkeypatch, {"/api/datasets/": _INVALID_UTF8})
    with pytest.raises(ValueError) as excinfo:
        prep._hf_repo_info("lerobot/pusht", "main")
    assert "lerobot/pusht@main" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, UnicodeDecodeError)


def test_hf_repo_info_malformed_json_raises_contextual_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_urlopen(monkeypatch, {"/api/datasets/": _MALFORMED_JSON})
    with pytest.raises(ValueError) as excinfo:
        prep._hf_repo_info("lerobot/pusht", "main")
    assert "lerobot/pusht@main" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)


def test_hf_tree_invalid_utf8_raises_contextual_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_urlopen(monkeypatch, {"/tree/": _INVALID_UTF8})
    with pytest.raises(ValueError) as excinfo:
        prep._hf_tree("lerobot/pusht", "main", "meta")
    message = str(excinfo.value)
    assert "lerobot/pusht@main" in message
    assert "'meta'" in message
    assert isinstance(excinfo.value.__cause__, UnicodeDecodeError)


def test_hf_tree_malformed_json_raises_contextual_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_urlopen(monkeypatch, {"/tree/": b"[1, 2"})
    with pytest.raises(ValueError) as excinfo:
        prep._hf_tree("lerobot/pusht", "main", "meta")
    message = str(excinfo.value)
    assert "lerobot/pusht@main" in message
    assert "'meta'" in message
    assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)


def test_fetch_info_json_invalid_utf8_raises_contextual_value_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tree_body = json.dumps([{"path": "meta/info.json", "type": "file"}]).encode()
    _stub_urlopen(
        monkeypatch,
        {"recursive=true": tree_body, "meta/info.json": _INVALID_UTF8},
    )
    with pytest.raises(ValueError) as excinfo:
        prep._fetch_info_json("lerobot/pusht", "main", tmp_path)
    message = str(excinfo.value)
    assert "meta/info.json" in message
    assert "lerobot/pusht@main" in message
    assert isinstance(excinfo.value.__cause__, UnicodeDecodeError)


def test_fetch_info_json_malformed_json_raises_contextual_value_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tree_body = json.dumps([{"path": "meta/info.json", "type": "file"}]).encode()
    _stub_urlopen(
        monkeypatch,
        {"recursive=true": tree_body, "meta/info.json": b"{oops"},
    )
    with pytest.raises(ValueError) as excinfo:
        prep._fetch_info_json("lerobot/pusht", "main", tmp_path)
    message = str(excinfo.value)
    assert "meta/info.json" in message
    assert "lerobot/pusht@main" in message
    assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)


def test_hf_repo_info_valid_json_still_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    # A real resolved commit sha: at least the 7 hex characters the sha
    # validation requires, since a cache directory is named after it.
    body = json.dumps({"sha": "abc1234", "cardData": {"license": "apache-2.0"}}).encode()
    _stub_urlopen(monkeypatch, {"/api/datasets/": body})
    assert prep._hf_repo_info("lerobot/pusht", "main") == {
        "sha": "abc1234",
        "license": "apache-2.0",
    }


def test_hf_tree_valid_json_still_returns_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    body = json.dumps([{"path": "meta/info.json", "type": "file"}]).encode()
    stub = _stub_urlopen(monkeypatch, {"/tree/": body})
    assert prep._hf_tree("lerobot/pusht", "main", "meta") == [
        {"path": "meta/info.json", "type": "file"}
    ]
    assert len(stub.requests) == 1


def test_hf_tree_follows_next_link_preserves_headers_and_deduplicates_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_page = json.dumps([{"path": "meta/info.json", "type": "file"}]).encode()
    second_page = json.dumps(
        [
            {"path": "meta/info.json", "type": "file"},
            {"path": "meta/episodes/file-000.parquet", "type": "file"},
        ]
    ).encode()
    next_url = (
        "https://huggingface.co/api/datasets/lerobot/pusht/tree/main/meta"
        "?recursive=true&cursor=page-2"
    )
    stub = _stub_urlopen(
        monkeypatch,
        {"cursor=page-2": second_page, "/tree/": first_page},
        {"/tree/": {"Link": f'<{next_url}>; rel="next"'}},
    )
    monkeypatch.setenv("HF_TOKEN", "test-token")

    assert prep._hf_tree("lerobot/pusht", "main", "meta") == [
        {"path": "meta/info.json", "type": "file"},
        {"path": "meta/episodes/file-000.parquet", "type": "file"},
    ]
    assert [request.full_url for request in stub.requests] == [
        "https://huggingface.co/api/datasets/lerobot/pusht/tree/main/meta?recursive=true",
        next_url,
    ]
    for request in stub.requests:
        assert request.get_header("User-agent") == "hflow-lerobot"
        assert request.get_header("Authorization") == "Bearer test-token"


def test_hf_tree_rejects_next_link_with_different_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_page = json.dumps([{"path": "meta/info.json", "type": "file"}]).encode()
    unsafe_next_url = "https://attacker.example/tree/secret?cursor=page-2"
    stub = _stub_urlopen(
        monkeypatch,
        {"/tree/": first_page},
        {"/tree/": {"Link": f'<{unsafe_next_url}>; rel="next"'}},
        max_requests=1,
    )
    monkeypatch.setenv("HF_TOKEN", "secret-user-token")

    with pytest.raises(ValueError, match="different scheme and host"):
        prep._hf_tree("lerobot/pusht", "main", "meta")

    assert len(stub.requests) == 1


def test_hf_tree_rejects_a_repeated_next_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_page = json.dumps([{"path": "meta/info.json", "type": "file"}]).encode()
    page_url = "https://huggingface.co/api/datasets/lerobot/pusht/tree/main/meta?recursive=true"
    stub = _stub_urlopen(
        monkeypatch,
        {"/tree/": first_page},
        {"/tree/": {"Link": f'<{page_url}>; rel="next"'}},
        max_requests=1,
    )

    with pytest.raises(ValueError, match="already fetched"):
        prep._hf_tree("lerobot/pusht", "main", "meta")

    assert len(stub.requests) == 1


def test_hf_tree_malformed_later_page_raises_contextual_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_page = json.dumps([{"path": "meta/info.json", "type": "file"}]).encode()
    next_url = (
        "https://huggingface.co/api/datasets/lerobot/pusht/tree/main/meta"
        "?recursive=true&cursor=page-2"
    )
    _stub_urlopen(
        monkeypatch,
        {"cursor=page-2": _MALFORMED_JSON, "/tree/": first_page},
        {"/tree/": {"Link": f'<{next_url}>; rel="next"'}},
    )

    with pytest.raises(ValueError) as excinfo:
        prep._hf_tree("lerobot/pusht", "main", "meta")

    message = str(excinfo.value)
    assert "lerobot/pusht@main" in message
    assert "'meta'" in message
    assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)


def test_hf_repo_info_wrong_shape_refusal_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_urlopen(monkeypatch, {"/api/datasets/": b"[1, 2, 3]"})
    with pytest.raises(ValueError, match="not a JSON object"):
        prep._hf_repo_info("lerobot/pusht", "main")


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
    monkeypatch.setattr(
        prep, "_hf_repo_info", lambda repo, rev: {"sha": rev, "license": "apache-2.0"}
    )
    monkeypatch.setattr(prep, "_fetch_info_json", lambda repo, rev, cache: info)

    def fail_tree(repo: str, rev: str, path: str) -> list[dict]:
        raise AssertionError("episode metadata discovery must not run after invalid fps")

    monkeypatch.setattr(prep, "_hf_tree", fail_tree)

    dataset_source = prep.DatasetSource(repo_id="fake/repo", revision="abc", license="apache-2.0")
    cache_dir = tmp_path / "cache"
    with pytest.raises(ValueError, match="fps") as excinfo:
        prep._ensure_source_archive(dataset_source, cache_dir)

    message = str(excinfo.value)
    assert "FPS must be finite and positive" in message
    assert repr(expected_value) in message
    # Refusal happens before episode metadata discovery: no downloads, no output.
    assert not cache_dir.exists() or not (cache_dir / "meta" / "episodes").exists()


def test_info_json_accepts_normal_positive_fps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _build_fake_corpus(tmp_path)
    corpus["info"]["fps"] = 30
    monkeypatch.setattr(
        prep, "_hf_repo_info", lambda repo, rev: {"sha": rev, "license": "apache-2.0"}
    )
    monkeypatch.setattr(prep, "_fetch_info_json", lambda repo, rev, cache: corpus["info"])
    monkeypatch.setattr(
        prep,
        "_hf_tree",
        lambda repo, rev, path: (
            [{"path": "meta/episodes/chunk-000/file-000.parquet", "type": "file"}]
            if "episodes" in path
            else [{"path": "meta/info.json", "type": "file"}]
        ),
    )

    def fake_dl(url: str, dest: Path, **kw: object) -> None:
        import shutil

        dest.parent.mkdir(parents=True, exist_ok=True)
        if "meta/episodes" in url:
            shutil.copy(
                str(tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"), dest
            )

    monkeypatch.setattr(prep, "_download_file", fake_dl)

    dataset_source = prep.DatasetSource(repo_id="fake/repo", revision="abc", license="apache-2.0")
    source_archive = prep._ensure_source_archive(dataset_source, tmp_path / "cache")
    assert source_archive["fps"] == 30


def _stub_single_episode_source_archive(
    dataset_source: prep.DatasetSource, cache_dir: Path
) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return {
        "info": {
            "robot_type": "pusht",
            "features": {
                prep.DEFAULT_CAMERA_KEY: {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                    "info": {"is_depth_map": False},
                }
            },
        },
        "fps": 30,
        "data_path": "data/{chunk_index}/{file_index}.parquet",
        "video_path": "videos/{camera_key}/{chunk_index}/{file_index}.mp4",
        "episodes": [
            {
                "episode_index": 0,
                "task": "push",
                "length": 1,
                "data_chunk": "000",
                "data_file": "000",
                "data_from": 0,
                "data_to": 1,
            }
        ],
        "video_keys": [prep.DEFAULT_CAMERA_KEY],
        "numeric_features": {
            "action": {"dtype": "float32", "shape": [1]},
            "observation.state": {"dtype": "float32", "shape": [1]},
        },
        "cache_dir": cache_dir,
        "dataset": dataset_source,
    }


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
        relative_key = f"landing/lerobot_episode_{episode_index + 1:04d}.mcap"
        staged = tmp_path / f"staged-{episode_index}.mcap"
        staged.write_bytes(f"episode-{episode_index}".encode())
        published_keys.append(relative_key)
        published_uri = storage.publish(staged, relative_key)
        return {
            "uri": published_uri,
            "content_id": prep.content_episode_id(staged),
            "size_bytes": staged.stat().st_size,
        }

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

    def ensure_depth_archive(dataset_source: prep.DatasetSource, cache_dir: Path) -> dict:
        archive = _stub_single_episode_source_archive(dataset_source, cache_dir)
        archive["info"]["features"] = {
            prep.DEFAULT_CAMERA_KEY: {
                "dtype": "video",
                "shape": [24, 32, 1],
                **depth_metadata,
            }
        }
        return archive

    monkeypatch.setattr(
        prep, "_hf_repo_info", lambda repo, revision: {"sha": "abc", "license": "apache-2.0"}
    )
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
    output_dir = tmp_path / "out"
    monkeypatch.setattr(
        prep, "_hf_repo_info", lambda repo, revision: {"sha": "abc", "license": "apache-2.0"}
    )
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
    source_archive = cast(
        prep._SourceArchive,
        {
            **corpus,
            "episodes": [
                {
                    "episode_index": 0,
                    "task": "pick-and-place",
                    "length": 1,
                    "data_chunk": "000",
                    "data_file": "000",
                    "data_from": 0,
                    "data_to": 1,
                    "video_windows": {
                        camera_key: {
                            "chunk_index": "000",
                            "file_index": "000",
                            "from_timestamp": 0.0,
                            "to_timestamp": 0.0,
                        }
                    },
                }
            ],
            "video_keys": [camera_key],
        },
    )
    numeric_schemas = {
        "observation.state": prep._NumericSchema(name="observation.state", dim=6),
        "action": prep._NumericSchema(name="action", dim=6),
    }

    def fake_download(url: str, destination_path: Path, **_kwargs: object) -> None:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if "/videos/" in url:
            destination_path.write_bytes(url.encode())
            return
        shutil.copy(tmp_path / "data" / "chunk-000" / "file-000.parquet", destination_path)

    monkeypatch.setattr(prep, "_download_file", fake_download)
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
        numeric_schemas=numeric_schemas,
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
    monkeypatch.setattr(
        prep, "_hf_repo_info", lambda repo, revision: {"sha": "abc", "license": "apache-2.0"}
    )
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


def test_manifest_records_per_episode_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manifest lists every delivered episode with its content id and size.

    A recipient of a prepared corpus gets a receipt that can be checked
    against the landing directory without re-running the import; the
    content id is the same ``content_episode_id`` the catalog dedupes on.
    """
    output_dir = tmp_path / "out"
    monkeypatch.setattr(
        prep, "_hf_repo_info", lambda repo, revision: {"sha": "abc", "license": "apache-2.0"}
    )
    monkeypatch.setattr(prep, "_ensure_source_archive", _stub_single_episode_source_archive)
    _install_publish_through_convert(monkeypatch, tmp_path)

    episode_uris = prep.import_lerobot_dataset(
        dataset_repo="fake/repo",
        revision="main",
        output_dir=output_dir,
        episode_index=0,
    )

    manifest = json.loads((output_dir / "prepared-manifest.json").read_text())
    assert manifest["schema_version"] == 3
    # v2 top-level keys survive: readers of the old schema keep working.
    assert manifest["episodes_converted"] == 1
    assert manifest["dataset"] == {
        "repo_id": "fake/repo",
        "revision": "abc",
        "license": "apache-2.0",
    }
    assert manifest["converter_version"] == prep.CONVERTER_VERSION
    assert manifest["camera_keys"] == [prep.DEFAULT_CAMERA_KEY]

    entries = manifest["episodes"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["uri"] == episode_uris[0]
    assert entry["size_bytes"] == len(b"episode-0")
    assert entry["content_id"] == prep.content_episode_id(
        output_dir / "landing" / "lerobot_episode_0001.mcap"
    )
    # The receipt describes the published landing object, not a local
    # staging path: for a bucket root this entry is an object URI that a
    # recipient of the bucket prefix can resolve without our filesystem.
    assert entry["uri"].endswith("landing/lerobot_episode_0001.mcap")
    assert "canonical-" not in entry["uri"]
    assert "staged-" not in entry["uri"]


def test_manifest_content_id_detects_a_truncated_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The #379 controlled result as a test: truncating one episode to zero
    bytes is detectable from the delivery by re-hashing against the manifest."""
    output_dir = tmp_path / "out"
    monkeypatch.setattr(
        prep, "_hf_repo_info", lambda repo, revision: {"sha": "abc", "license": "apache-2.0"}
    )
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


def test_import_skips_bucket_manifest_when_an_episode_publish_fails(
    tmp_path: Path,
    bucket_over_tmp: tuple[object, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hflow.storage import BucketStorageRoot

    data_root, remote_dir = bucket_over_tmp
    assert isinstance(data_root, BucketStorageRoot)

    def ensure_two_episodes(dataset_source: prep.DatasetSource, cache_dir: Path) -> dict:
        archive = _stub_single_episode_source_archive(dataset_source, cache_dir)
        archive["episodes"] = [
            archive["episodes"][0],
            {
                **archive["episodes"][0],
                "episode_index": 1,
                "task": "second",
            },
        ]
        return archive

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
        relative_key = f"landing/lerobot_episode_{episode_index + 1:04d}.mcap"
        staged = tmp_path / f"staged-{episode_index}.mcap"
        staged.write_bytes(b"first")
        published_uri = storage.publish(staged, relative_key)
        return {
            "uri": published_uri,
            "content_id": prep.content_episode_id(staged),
            "size_bytes": staged.stat().st_size,
        }

    monkeypatch.setattr(
        prep, "_hf_repo_info", lambda repo, revision: {"sha": "abc", "license": "apache-2.0"}
    )
    monkeypatch.setattr(prep, "_ensure_source_archive", ensure_two_episodes)
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
    from mcap.writer import Writer

    from hflow.format import METADATA_RECORD_EPISODE, METADATA_RECORD_PROVENANCE

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        writer = Writer(stream)
        writer.start(profile="", library="test-lerobot-resume")
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


def _ensure_two_episode_archive(dataset_source: prep.DatasetSource, cache_dir: Path) -> dict:
    archive = _stub_single_episode_source_archive(dataset_source, cache_dir)
    archive["episodes"] = [
        archive["episodes"][0],
        {
            **archive["episodes"][0],
            "episode_index": 1,
            "task": "second",
        },
    ]
    return archive


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
        relative_key = prep._landing_relative_key(episode_index)
        staged = tmp_path / f"staged-{episode_index}-{len(convert_calls)}.mcap"
        _write_identity_matching_landing_mcap(
            staged,
            dataset_source=dataset_source,
            episode_index=episode_index,
            camera_keys=camera_keys,
            marker=f"episode-{episode_index}-bytes",
        )
        published_uri = storage.publish(staged, relative_key)
        return {
            "uri": published_uri,
            "content_id": prep.content_episode_id(staged),
            "size_bytes": staged.stat().st_size,
        }

    monkeypatch.setattr(
        prep, "_hf_repo_info", lambda repo, revision: {"sha": "abc", "license": "apache-2.0"}
    )
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
        relative_key = prep._landing_relative_key(episode_index)
        staged = tmp_path / f"replacement-{episode_index}.mcap"
        _write_identity_matching_landing_mcap(
            staged,
            dataset_source=dataset_source,
            episode_index=episode_index,
            camera_keys=camera_keys,
            marker="replacement",
        )
        published_uri = storage.publish(staged, relative_key)
        return {
            "uri": published_uri,
            "content_id": prep.content_episode_id(staged),
            "size_bytes": staged.stat().st_size,
        }

    monkeypatch.setattr(
        prep, "_hf_repo_info", lambda repo, revision: {"sha": "abc", "license": "apache-2.0"}
    )
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

    monkeypatch.setattr(
        prep, "_hf_repo_info", lambda repo, revision: {"sha": "abc", "license": "apache-2.0"}
    )
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


def _build_success_label_corpus(root: Path, outcome_mode: str) -> dict:
    """One two-frame episode. outcome_mode: 'transition', 'all-false', 'none'."""
    has_outcome = outcome_mode != "none"
    info = {
        "fps": 30,
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
        stats_min, stats_max = (
            ([False], [True]) if outcome_mode == "transition" else ([False], [False])
        )
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome_mode: str
) -> Path:
    root = tmp_path / "corpus"
    corpus = _build_success_label_corpus(root, outcome_mode)
    output_dir = tmp_path / "out"

    monkeypatch.setattr(
        prep, "_hf_repo_info", lambda repo, revision: {"sha": "abc1234", "license": "apache-2.0"}
    )
    monkeypatch.setattr(prep, "_fetch_info_json", lambda repo, rev, cache: corpus["info"])
    monkeypatch.setattr(
        prep,
        "_hf_tree",
        lambda repo, rev, path: (
            [{"path": "meta/episodes/chunk-000/file-000.parquet", "type": "file"}]
            if "episodes" in path
            else [{"path": "meta/info.json", "type": "file"}]
        ),
    )

    def fake_download(url: str, dest: Path, **_kwargs: object) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if "meta/episodes" in url:
            shutil.copy(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet", dest)
        elif url.endswith("info.json"):
            shutil.copy(root / "meta" / "info.json", dest)
        else:
            shutil.copy(root / "data" / "chunk-000" / "file-000.parquet", dest)

    monkeypatch.setattr(prep, "_download_file", fake_download)
    monkeypatch.setattr(
        prep,
        "_transcode_mp4_to_h264",
        lambda mp4_path, gop, fps: (
            [
                b"\x00\x00\x00\x01\x09\x10\x00\x00\x00\x01\x67\x42\x00"
                b"\x00\x00\x00\x01\x68\x88\x80\x00\x00\x00\x01\x65\x88"
            ]
            * 2
        ),
    )
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
    """The label changes episode/v1 bytes, which content_episode_id hashes:
    the converter version moves with the change, not after it."""
    assert prep.CONVERTER_VERSION == "lerobot-converter-v7"
