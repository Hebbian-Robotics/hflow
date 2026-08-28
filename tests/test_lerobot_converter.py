"""Outcome-focused fixture for the LeRobot v3 converter generalization.

These tests exercise the metadata-driven discovery and fail-loud behavior
with a synthetic v3-style corpus, without asserting third-party
implementation details or touching the network.
"""

import importlib.util
import json
from pathlib import Path

import pytest

_PREPARE_PATH = Path(__file__).resolve().parents[1] / "examples" / "lerobot" / "prepare.py"
_spec = importlib.util.spec_from_file_location("hflow_lerobot_prepare", _PREPARE_PATH)
assert _spec and _spec.loader
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)
_DERIVE = prep._derive_numeric_schema
_ENCODE = prep._encode_cdr_float32_array


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
    monkeypatch.setattr(prep, "_require_ffmpeg", lambda: None)

    with pytest.raises(ValueError, match="not found"):
        prep.lerobot_to_mcap(
            dataset_repo="fake/repo",
            revision="abc",
            output_dir=tmp_path / "out",
            camera_key="observation.nope",
        )
