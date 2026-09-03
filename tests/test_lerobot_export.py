"""Tests for the HFlow LeRobot exporter (examples/lerobot/export.py).

The exporter resolves a curation selection to source LeRobot provenance,
fetches the source chunks, and publishes a local LeRobot Dataset v3
repository containing exactly the selected episodes. The tests use a
synthetic source corpus and a fake manifest, with the public
``hflow.import_lerobot_dataset`` entry point monkeypatched to
materialize the synthetic archive.
"""

from __future__ import annotations

import hashlib
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


def _fake_corpus(tmp_path: Path, *, chunk1_eps: tuple[int, ...] = ()) -> dict:
    """Synthetic v3 source: 4 episodes, 2 cameras, 6-dim state/action.

    Episodes named in ``chunk1_eps`` reference video chunk 1 (files present
    with distinguishable bytes), so a selection can span two source chunks.
    """
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
        vchunk = 1 if i in chunk1_eps else 0
        rows.append(
            [
                i,
                length,
                0,
                0,
                OFFSETS[i],
                OFFSETS[i] + length,
                vchunk,
                0,
                0.0,
                2.0 + i * 0.2,
                vchunk,
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
    # local naming: cache/videos/<sanitized-cam>-chunk<N>-file<M>.mp4
    for chunk in (0, 1) if chunk1_eps else (0,):
        for cam in ("observation.images.up", "observation.images.side"):
            vname = cam.replace("/", "_").replace(".", "_") + f"-chunk{chunk}-file0.mp4"
            vdir = tmp_path / "videos"
            vdir.mkdir(parents=True, exist_ok=True)
            marker = "" if chunk == 0 else "CHUNK-ONE-"
            (vdir / vname).write_bytes(f"fake-mp4-{marker}{cam}".encode())

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
                        "chunk_index": str(vchunk),
                        "file_index": "0",
                        "from_timestamp": 0.0,
                        "to_timestamp": 2.0 + i * 0.2,
                    },
                    "observation.images.side": {
                        "chunk_index": str(vchunk),
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


def _install_fake_import(corpus: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
        cache = Path(output_dir) / "_lerobot_cache" / str(revision)
        if not cache.exists():
            shutil.copytree(tmp_path, cache)
        return []

    monkeypatch.setattr(hflow, "import_lerobot_dataset", _fake_import)


@pytest.fixture()
def fake_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    corpus = _fake_corpus(tmp_path)
    _install_fake_import(corpus, tmp_path, monkeypatch)
    return corpus


@pytest.fixture()
def multi_chunk_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    corpus = _fake_corpus(tmp_path, chunk1_eps=(2,))
    _install_fake_import(corpus, tmp_path, monkeypatch)
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
    assert [r[2] for r in rows] == [0, 1]  # plain indexes, one data parquet per episode
    assert (rows[0][3], rows[0][4]) == (0, 60)
    assert (rows[1][3], rows[1][4]) == (60, 130)
    assert rows[1][5] == ["task-2"]

    ep1_data = dest / "data" / "chunk-000" / "file-001.parquet"
    assert ep1_data.exists()  # the per-episode file episode 1 references
    conn = duckdb.connect()
    dcount = conn.execute(
        "SELECT count(*) FROM read_parquet('"
        + str(dest / "data/chunk-000/file-000.parquet").replace("'", "''")
        + "')"
    ).fetchall()[0]
    dcount2 = conn.execute(
        "SELECT count(*), max(frame_index) FROM read_parquet('"
        + str(ep1_data).replace("'", "''")
        + "')"
    ).fetchall()[0]
    frames = conn.execute(
        "SELECT frame_index FROM read_parquet('"
        + str(dest / "data/chunk-000/file-000.parquet").replace("'", "''")
        + "')"
    ).fetchall()
    conn.close()
    assert dcount[0] == 60  # episode 0 alone in its own parquet
    assert dcount2 == (70, 69)  # episode 1: full length, per-episode restart
    assert [f[0] for f in frames[:3]] == [0, 1, 2]

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


def test_export_missing_index_column_fails(fake_corpus: dict, tmp_path: Path) -> None:
    """A source data parquet without an 'index' column is refused by message."""
    import pyarrow.parquet as pq

    src = Path(fake_corpus["cache_dir"]) / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(src).drop_columns(["index"])
    pq.write_table(table, src)

    manifest = _fake_manifest(tmp_path, [{"metadata_json": _provenance_meta(0)}])
    dest = tmp_path / "out"
    with pytest.raises(ValueError, match=r"has no 'index' column"):
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


def _exported_dataset(fake_corpus: dict, tmp_path: Path) -> Path:
    """Export a valid single-episode dataset for the corruption tests."""
    manifest = _fake_manifest(tmp_path, [{"metadata_json": _provenance_meta(0)}])
    dest = tmp_path / "out"
    export.export(dest, manifest=manifest, camera_keys=CAMS)
    return dest


def test_export_provenance_digests_match_bytes(fake_corpus: dict, tmp_path: Path) -> None:
    """Provenance digests correspond to the bytes on disk.

    Regression for the review finding that _sha256 could return a constant
    without any test noticing: each digest must equal the sha256 of the
    file it names, and distinct files must produce distinct digests.
    """

    def _sha(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    dest = _exported_dataset(fake_corpus, tmp_path)
    prov = json.loads((dest / "export-provenance.json").read_text())

    up = dest / "videos" / "observation.images.up" / "chunk-000" / "file-000.mp4"
    side = dest / "videos" / "observation.images.side" / "chunk-000" / "file-000.mp4"
    data = dest / "data" / "chunk-000" / "file-000.parquet"
    assert set(prov["video_sha256"]) == {
        "observation.images.up",
        "observation.images.side",
    }
    assert prov["video_sha256"]["observation.images.up"] == {
        "videos/observation.images.up/chunk-000/file-000.mp4": _sha(up)
    }
    assert prov["video_sha256"]["observation.images.side"] == {
        "videos/observation.images.side/chunk-000/file-000.mp4": _sha(side)
    }
    assert prov["data_parquet_sha256"] == {"data/chunk-000/file-000.parquet": _sha(data)}
    # a constant would fail here: different files have different digests
    video_digests = [
        digest for per_cam in prov["video_sha256"].values() for digest in per_cam.values()
    ]
    assert (
        prov["video_sha256"]["observation.images.up"][
            "videos/observation.images.up/chunk-000/file-000.mp4"
        ]
        != prov["video_sha256"]["observation.images.side"][
            "videos/observation.images.side/chunk-000/file-000.mp4"
        ]
    )
    assert all(digest not in video_digests for digest in prov["data_parquet_sha256"].values())


def test_export_multi_chunk_videos(multi_chunk_corpus: dict, tmp_path: Path) -> None:
    """A selection spanning two source video chunks ships both byte-exact.

    Regression for the review finding that every camera landed in
    chunk-000/file-000.mp4 and the second source chunk never reached the
    output: distinct source (chunk, file) identities must land in distinct
    destination files, and each episode row must reference the file its
    frames actually come from.
    """
    manifest = _fake_manifest(
        tmp_path,
        [
            {"metadata_json": _provenance_meta(0)},
            {"metadata_json": _provenance_meta(1)},
            {"metadata_json": _provenance_meta(2, task="task-2")},
        ],
    )
    dest = tmp_path / "out"
    export.export(dest, manifest=manifest, camera_keys=CAMS)

    up0 = dest / "videos" / "observation.images.up" / "chunk-000" / "file-000.mp4"
    up1 = dest / "videos" / "observation.images.up" / "chunk-001" / "file-000.mp4"
    side1 = dest / "videos" / "observation.images.side" / "chunk-001" / "file-000.mp4"
    assert up0.read_bytes() == b"fake-mp4-observation.images.up"
    assert up1.read_bytes() == b"fake-mp4-CHUNK-ONE-observation.images.up"
    assert side1.read_bytes() == b"fake-mp4-CHUNK-ONE-observation.images.side"
    assert not (dest / "videos" / "observation.images.up" / "chunk-000" / "file-001.mp4").exists()

    conn = duckdb.connect()
    rows = conn.execute(
        'SELECT episode_index, "videos/observation.images.up/chunk_index" FROM read_parquet(\''
        + str(dest / "meta/episodes/chunk-000/file-000.parquet").replace("'", "''")
        + "')"
    ).fetchall()
    conn.close()
    assert rows == [(0, 0), (1, 0), (2, 1)]

    info = json.loads((dest / "meta" / "info.json").read_text())
    assert info["total_videos"] == 4  # up/side x chunk0/chunk1, deduplicated

    prov = json.loads((dest / "export-provenance.json").read_text())
    up_digests = prov["video_sha256"]["observation.images.up"]
    assert (
        up_digests["videos/observation.images.up/chunk-000/file-000.mp4"]
        != up_digests["videos/observation.images.up/chunk-001/file-000.mp4"]
    )
    assert set(prov["video_sha256"]["observation.images.side"]) == {
        "videos/observation.images.side/chunk-000/file-000.mp4",
        "videos/observation.images.side/chunk-001/file-000.mp4",
    }


def test_export_output_readback(fake_corpus: dict, tmp_path: Path) -> None:
    """The exported episode parquet parses with the exporter's own reader.

    Regression for the review finding that data/chunk_index was written as
    'chunk-000' and broke int() conversion on re-read: exported indexes are
    plain integers and every referenced data file exists on disk.
    """
    dest = _exported_dataset(fake_corpus, tmp_path)
    corpus = export._read_corpus_from_cache(dest)
    assert len(corpus["episodes"]) == 1
    ep = corpus["episodes"][0]
    assert int(ep["data_chunk"]) == 0
    assert int(ep["data_file"]) == 0
    assert ep["video_windows"]["observation.images.up"]["chunk_index"] == "0"
    info = json.loads((dest / "meta" / "info.json").read_text())
    drel = info["data_path"].format(
        chunk_index=int(ep["data_chunk"]), file_index=int(ep["data_file"])
    )
    assert (dest / drel).exists()


def test_export_validation_failure_leaves_no_destination(
    fake_corpus: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A staging validation refusal propagates and never publishes.

    Regression for the review finding that the _validate_v3 call could be
    deleted without any test noticing: a validator refusal must abort the
    export, leave no destination behind, and leave a pre-existing
    destination untouched.
    """

    def _reject(_stage: Path) -> None:
        raise ValueError("staged dataset rejected for the test")

    monkeypatch.setattr(export, "_validate_v3", _reject)

    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "sentinel.txt").write_text("keep")
    manifest = _fake_manifest(tmp_path, [{"metadata_json": _provenance_meta(0)}])
    with pytest.raises(ValueError, match="rejected for the test"):
        export.export(dest, manifest=manifest, camera_keys=CAMS)
    assert (dest / "sentinel.txt").read_text() == "keep"

    dest2 = tmp_path / "out2"
    with pytest.raises(ValueError, match="rejected for the test"):
        export.export(dest2, manifest=manifest, camera_keys=CAMS)
    assert not dest2.exists()


def test_validate_v3_rejects_non_integer_data_refs(fake_corpus: dict, tmp_path: Path) -> None:
    """Index fields must be values data_path can format: 'chunk-000' => refusal.

    Regression for the review finding that the exported data refs were
    written as 'chunk-000' strings: the loader formats these with :03d, so a
    non-integer reference makes the dataset unloadable and must be refused.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    dest = _exported_dataset(fake_corpus, tmp_path)
    ep_pq = dest / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(str(ep_pq))
    idx = table.schema.get_field_index("data/chunk_index")
    table = table.set_column(idx, "data/chunk_index", pa.array(["chunk-000"], pa.string()))
    pq.write_table(table, str(ep_pq))
    with pytest.raises(ValueError, match="cannot format references"):
        export._validate_v3(dest)


def test_validate_v3_rejects_incoherent_info(fake_corpus: dict, tmp_path: Path) -> None:
    """info.json coherence is a hard validation: wrong code => refusal."""
    dest = _exported_dataset(fake_corpus, tmp_path)
    meta = dest / "meta" / "info.json"
    info = json.loads(meta.read_text())
    info["code"] = "LeRobotDataset/v2"
    meta.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="not LeRobotDataset/v3"):
        export._validate_v3(dest)


def test_validate_v3_rejects_window_length_mismatch(fake_corpus: dict, tmp_path: Path) -> None:
    """Per-episode window must equal length: corrupt length => refusal."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    dest = _exported_dataset(fake_corpus, tmp_path)
    ep_pq = dest / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(str(ep_pq))
    lengths = table.column("length").to_pylist()
    lengths[0] = int(lengths[0]) + 1
    table = table.set_column(
        table.schema.get_field_index("length"), "length", pa.array(lengths, pa.int64())
    )
    pq.write_table(table, str(ep_pq))
    with pytest.raises(ValueError, match="!= length"):
        export._validate_v3(dest)


def test_validate_v3_rejects_missing_video(fake_corpus: dict, tmp_path: Path) -> None:
    """Every referenced camera's video file must exist: missing => refusal."""
    dest = _exported_dataset(fake_corpus, tmp_path)
    video = dest / "videos" / "observation.images.side" / "chunk-000" / "file-000.mp4"
    video.unlink()
    with pytest.raises(ValueError, match="references missing video"):
        export._validate_v3(dest)
