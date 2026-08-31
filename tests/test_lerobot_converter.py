"""Outcome-focused coverage for first-class LeRobot Dataset v3 import.

These tests exercise the metadata-driven discovery and fail-loud behavior
with a synthetic v3-style corpus, without asserting third-party
implementation details or touching the network.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

import hflow.importers.lerobot as prep
from hflow.cli import main as cli_main

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
    resolved_shas = {"branch-a": "sha-a", "branch-b": "sha-b", "tag-a": "sha-a"}
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
        ("sha-a", tmp_path / "_lerobot_cache" / "sha-a", "sha-a"),
        ("sha-b", tmp_path / "_lerobot_cache" / "sha-b", "sha-b"),
        ("sha-a", tmp_path / "_lerobot_cache" / "sha-a", "sha-a"),
    ]
    assert sorted(path.name for path in (tmp_path / "_lerobot_cache").iterdir()) == [
        "sha-a",
        "sha-b",
    ]


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
