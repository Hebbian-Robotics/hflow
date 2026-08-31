"""Tests for the HFlow LeRobot exporter (examples/lerobot/export.py).

The exporter resolves a curation selection to source LeRobot provenance,
fetches the source chunks, and publishes a local LeRobot Dataset v3
repository containing exactly the selected episodes. The tests use a
synthetic source corpus and a fake manifest, with the public
``hflow.import_lerobot_dataset`` entry point monkeypatched to
materialize the synthetic archive.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import duckdb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hflow
from examples.lerobot import export

CAMS = ("observation.images.up", "observation.images.side")
LENGTHS = [60, 65, 70, 75]  # episode lengths: 60 + i*5
OFFSETS = [0, 60, 125, 195]  # cumulative data offsets


def _fake_corpus(tmp_path: Path) -> dict:
    """Synthetic v3 source: 4 episodes, 2 cameras, 6-dim state/action."""
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

    rows = []
    for i in range(4):
        length = LENGTHS[i]
        rows.append(
            [
                i,
                length,
                0,
                0,
                OFFSETS[i],
                OFFSETS[i] + length,
                0,
                0,
                0.0,
                2.0 + i * 0.2,
                0,
                0,
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
    conn = duckdb.connect()
    vals = ",".join(
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
    quoted = str(ep_path).replace("'", "''")
    qcols = ",".join(f'"{c}"' for c in ep_cols)
    conn.execute(
        f"COPY (SELECT * FROM (VALUES {vals}) AS t({qcols})) TO '{quoted}' (FORMAT parquet)"
    )

    # one contiguous data chunk: all episodes' rows, index = row position
    data_rows = []
    idx = 0
    for i in range(4):
        length = LENGTHS[i]
        for f in range(length):
            state = "[" + ",".join(str(float(f)) for _ in range(6)) + "]"
            action = "[" + ",".join(str(float(f + 0.5)) for _ in range(6)) + "]"
            data_rows.append([idx, i, f, round(f / 30.0, 6), state, action])
            idx += 1
    data_path = tmp_path / "data" / "chunk-000" / "file-000.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    dvals = ",".join("(" + ",".join(str(v) for v in row) + ")" for row in data_rows)
    dquoted = str(data_path).replace("'", "''")
    conn.execute(
        f"COPY (SELECT * FROM (VALUES {dvals}) AS "
        't(index, episode_index, frame_index, timestamp, "observation.state", action)) '
        f"TO '{dquoted}' (FORMAT parquet)"
    )

    # video chunks (fake bytes); exporter fetches them under the converter's
    # local naming: cache/videos/<sanitized-cam>-chunk<N>.mp4
    for cam in ("observation.images.up", "observation.images.side"):
        vname = cam.replace("/", "_").replace(".", "_") + "-chunk0.mp4"
        vdir = tmp_path / "videos"
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / vname).write_bytes(f"fake-mp4-{cam}".encode())

    conn.close()

    episodes = []
    for i in range(4):
        episodes.append(
            {
                "episode_index": i,
                "task": f"task-{i}",
                "length": LENGTHS[i],
                "data_chunk": "0",
                "data_file": "0",
                "data_from": OFFSETS[i],
                "data_to": OFFSETS[i] + LENGTHS[i],
                "video_windows": {
                    "observation.images.up": {
                        "chunk_index": "0",
                        "file_index": "0",
                        "from_timestamp": 0.0,
                        "to_timestamp": 2.0 + i * 0.2,
                    },
                    "observation.images.side": {
                        "chunk_index": "0",
                        "file_index": "0",
                        "from_timestamp": 0.0,
                        "to_timestamp": 2.0 + i * 0.2,
                    },
                },
            }
        )

    return {
        "info": info,
        "fps": 30,
        "data_path": info["data_path"],
        "video_path": info["video_path"],
        "cache_dir": tmp_path,
        "data_chunk": 0,
        "data_file": 0,
        "video_keys": ("observation.images.up", "observation.images.side"),
        "episodes": episodes,
    }


@pytest.fixture()
def fake_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    corpus = _fake_corpus(tmp_path)
    corpus["_import_calls"] = []

    def _fake_import(
        dataset_repo: str,
        revision: str,
        output_dir: Path,
        episode_index: object = None,
        camera_keys: tuple[str, ...] = (),
    ) -> list[Path]:
        corpus["_import_calls"].append(
            {
                "dataset_repo": dataset_repo,
                "revision": revision,
                "episode_index": episode_index,
                "camera_keys": camera_keys,
            }
        )
        _make_data_local(corpus)
        cache = Path(output_dir) / "_lerobot_cache"
        if not cache.exists():
            shutil.copytree(tmp_path, cache)
        return []

    monkeypatch.setattr(hflow, "import_lerobot_dataset", _fake_import)
    return corpus


def _fake_manifest(tmp_path: Path, rows: list[dict]) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    mpath = tmp_path / "manifest.parquet"
    table = pa.Table.from_pylist(rows)
    if "episode_id" not in table.column_names:
        table = table.append_column("episode_id", pa.array([f"ep_{i}" for i in range(len(rows))]))
    pq.write_table(table, mpath)
    return mpath


def _provenance_meta(ep: int, task: str = "task-0") -> str:
    return json.dumps(
        {
            "source_dataset": "lerobot/fake",
            "source_revision": "a" * 40,
            "source_episode_index": ep,
            "task": task,
            "embodiment": "so101",
        }
    )


def _make_data_local(corpus: dict) -> None:
    """Place the data chunk where the exporter fetches it."""
    src = Path(corpus["cache_dir"]) / "data" / "chunk-000" / "file-000.parquet"
    dst = Path(corpus["cache_dir"]) / "data" / "chunk-000000-file-000000.parquet"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)


def test_export_noncontiguous_selection(fake_corpus: dict, tmp_path: Path) -> None:
    """Outcome: exactly episodes 0 and 2, in selection order, loadable layout."""
    _make_data_local(fake_corpus)
    manifest = _fake_manifest(
        tmp_path,
        [
            {"metadata_json": _provenance_meta(0)},
            {"metadata_json": _provenance_meta(2, task="task-2")},
        ],
    )
    dest = tmp_path / "out"
    export.export(dest, manifest=manifest, camera_keys=CAMS)

    info = json.loads((dest / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 60 + 70  # episodes 0 and 2
    assert info["splits"]["train"] == ["episode_000000", "episode_000001"]
    assert "observation.images.up" in info["features"]
    assert "observation.images.side" in info["features"]
    assert info["features"]["action"]["dtype"] == "float32"

    conn = duckdb.connect()
    rows = conn.execute(
        'SELECT episode_index, length, "data/file_index", dataset_from_index, dataset_to_index, '
        "tasks FROM read_parquet('"
        + str(dest / "meta/episodes/chunk-000/file-000.parquet").replace("'", "''")
        + "')"
    ).fetchall()
    conn.close()
    assert [r[0] for r in rows] == [0, 1]
    assert rows[0][1] == 60
    assert rows[1][1] == 70
    assert (rows[0][3], rows[0][4]) == (0, 60)
    assert (rows[1][3], rows[1][4]) == (60, 130)
    assert rows[1][5] == ["task-2"]

    conn = duckdb.connect()
    dcount = conn.execute(
        "SELECT count(*), min(episode_index), max(episode_index) FROM read_parquet('"
        + str(dest / "data/chunk-000/file-000.parquet").replace("'", "''")
        + "')"
    ).fetchall()[0]
    # frame indexes preserve the source per-episode numbering
    frames = conn.execute(
        "SELECT frame_index FROM read_parquet('"
        + str(dest / "data/chunk-000/file-000.parquet").replace("'", "''")
        + "')"
    ).fetchall()
    conn.close()
    assert dcount[0] == 130
    assert dcount[1] == 0
    assert dcount[2] == 1
    assert [f[0] for f in frames[:3]] == [0, 1, 2]  # ep0 keeps 0..59
    assert frames[60][0] == 0  # ep2 restarts at 0

    for cam in ("observation.images.up", "observation.images.side"):
        v = dest / "videos" / cam / "chunk-000" / "file-000.mp4"
        assert v.exists()
        assert v.read_bytes() == f"fake-mp4-{cam}".encode()

    assert (dest / "README.md").exists()
    prov = json.loads((dest / "export-provenance.json").read_text())
    assert prov["source_episode_indexes"] == [0, 2]
    assert prov["source_commit"] == "a" * 40


def test_export_mixed_repositories_fail(fake_corpus: dict, tmp_path: Path) -> None:
    rows = [
        {"metadata_json": _provenance_meta(0)},
        {
            "metadata_json": json.dumps(
                {
                    "source_dataset": "lerobot/other",
                    "source_revision": "b" * 40,
                    "source_episode_index": 1,
                    "task": "task-1",
                    "embodiment": "so101",
                }
            )
        },
    ]
    manifest = _fake_manifest(tmp_path, rows)
    dest = tmp_path / "out"
    with pytest.raises(ValueError, match="mixes source repositories"):
        export.export(dest, manifest=manifest, camera_keys=CAMS)
    assert not dest.exists()


def test_export_nonimmutable_revision_fails(fake_corpus: dict, tmp_path: Path) -> None:
    rows = [
        {
            "metadata_json": json.dumps(
                {
                    "source_dataset": "lerobot/fake",
                    "source_revision": "main",
                    "source_episode_index": 0,
                    "task": "task-0",
                    "embodiment": "so101",
                }
            )
        }
    ]
    manifest = _fake_manifest(tmp_path, rows)
    dest = tmp_path / "out"
    with pytest.raises(ValueError, match="not an immutable commit sha"):
        export.export(dest, manifest=manifest, camera_keys=CAMS)
    assert not dest.exists()


def test_export_missing_provenance_fails(fake_corpus: dict, tmp_path: Path) -> None:
    manifest = _fake_manifest(tmp_path, [{"metadata_json": None}])
    dest = tmp_path / "out"
    with pytest.raises(ValueError, match="lacks LeRobot provenance"):
        export.export(dest, manifest=manifest, camera_keys=CAMS)
    assert not dest.exists()


def test_export_missing_source_episode_fails(fake_corpus: dict, tmp_path: Path) -> None:
    manifest = _fake_manifest(tmp_path, [{"metadata_json": _provenance_meta(9)}])
    dest = tmp_path / "out"
    with pytest.raises(ValueError, match="source episodes not present"):
        export.export(dest, manifest=manifest, camera_keys=CAMS)
    assert not dest.exists()


def test_export_duplicate_episode_fails(fake_corpus: dict, tmp_path: Path) -> None:
    manifest = _fake_manifest(
        tmp_path,
        [
            {"metadata_json": _provenance_meta(0)},
            {"metadata_json": _provenance_meta(0)},
        ],
    )
    dest = tmp_path / "out"
    with pytest.raises(ValueError, match="duplicate source episode indexes"):
        export.export(dest, manifest=manifest, camera_keys=CAMS)
    assert not dest.exists()


def test_export_sql_selection(fake_corpus: dict, tmp_path: Path) -> None:
    """SQL selection path: same outcome via a duckdb query string."""
    _make_data_local(fake_corpus)
    dest = tmp_path / "out"
    catalog = tmp_path / "catalog.parquet"
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(
        [
            {"episode_id": "ep_0", "metadata_json": _provenance_meta(0)},
            {"episode_id": "ep_3", "metadata_json": _provenance_meta(3, task="task-3")},
        ]
    )
    pq.write_table(table, catalog)
    export.export(
        dest,
        sql=f"SELECT episode_id, metadata_json FROM read_parquet('{catalog}')",
        camera_keys=CAMS,
    )
    info = json.loads((dest / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 60 + 75  # episodes 0 and 3


def test_export_drives_public_importer(fake_corpus: dict, tmp_path: Path) -> None:
    """Materialization goes through hflow.import_lerobot_dataset.

    Regression for the review finding that export imported private helpers
    from examples/lerobot/prepare.py; the source archive must be
    materialized through the exported entry point, all episodes, with the
    camera keys the user asked to export.
    """
    manifest = _fake_manifest(tmp_path, [{"metadata_json": _provenance_meta(0)}])
    export.export(
        tmp_path / "out",
        manifest=manifest,
        camera_keys=("observation.images.up", "observation.images.side"),
    )
    assert fake_corpus["_import_calls"], "export must drive the public importer"
    call = fake_corpus["_import_calls"][-1]
    assert call["dataset_repo"] == "lerobot/fake"
    assert call["revision"] == "a" * 40
    assert call["episode_index"] is None
    assert call["camera_keys"] == ("observation.images.up", "observation.images.side")
