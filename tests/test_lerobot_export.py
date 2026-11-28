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
from lerobot_test_helpers import (
    TWO_CAMERA_KEYS,
    CorpusEpisodeRow,
    two_camera_v3_info,
    write_v3_corpus,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hflow
from examples.lerobot import export

CAMS = TWO_CAMERA_KEYS
LENGTHS = [60, 65, 70, 75]  # episode lengths: 60 + i*5
OFFSETS = [0, 60, 125, 195]  # cumulative data offsets


def _fake_corpus(corpus_root: Path, *, chunk1_eps: tuple[int, ...] = ()) -> dict:
    """Synthetic v3 source: 4 episodes, 2 cameras, 6-dim state/action.

    Episodes named in ``chunk1_eps`` reference video chunk 1 (files present
    with distinguishable bytes), so a selection can span two source chunks.
    """
    info = two_camera_v3_info()
    write_v3_corpus(
        corpus_root,
        info=info,
        episode_rows=[
            CorpusEpisodeRow(
                episode_index=episode_index,
                length=LENGTHS[episode_index],
                dataset_from_index=OFFSETS[episode_index],
                video_to_timestamp=2.0 + episode_index * 0.2,
                tasks=(f"task-{episode_index}",),
                video_chunk_index=1 if episode_index in chunk1_eps else 0,
            )
            for episode_index in range(4)
        ],
    )

    # Source video chunks retain their repository paths in the SDK cache.
    for chunk in (0, 1) if chunk1_eps else (0,):
        for cam in CAMS:
            vdir = corpus_root / "videos" / cam / f"chunk-{chunk:03d}"
            vdir.mkdir(parents=True, exist_ok=True)
            marker = "" if chunk == 0 else "CHUNK-ONE-"
            (vdir / "file-000.mp4").write_bytes(f"fake-mp4-{marker}{cam}".encode())

    return {"info": info, "cache_dir": corpus_root}


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
        cache = Path(output_dir) / "_lerobot_cache" / str(revision)
        if not cache.exists():
            shutil.copytree(tmp_path, cache)
        failed_indexes = corpus.get("_fail_import_indexes", set())
        if episode_index is None and failed_indexes:
            raise ValueError(f"synthetic conversion failure for episode {min(failed_indexes)}")
        if episode_index in failed_indexes:
            raise ValueError(f"synthetic conversion failure for episode {episode_index}")
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


@pytest.mark.parametrize("explicit_video_path", [True, False])
def test_export_noncontiguous_selection(
    fake_corpus: dict, tmp_path: Path, explicit_video_path: bool
) -> None:
    """Outcome: exactly episodes 0 and 2, in selection order, loadable layout."""
    if not explicit_video_path:
        source_root = Path(fake_corpus["cache_dir"])
        for camera_key in CAMS:
            video_directory = source_root / "videos" / camera_key
            default_video_path = video_directory / "000000" / "000000.mp4"
            default_video_path.parent.mkdir(parents=True)
            (video_directory / "chunk-000" / "file-000.mp4").rename(default_video_path)
        del fake_corpus["info"]["video_path"]
        (source_root / "meta" / "info.json").write_text(json.dumps(fake_corpus["info"]))
    manifest = _fake_manifest(
        tmp_path,
        [
            {"metadata_json": _provenance_meta(0)},
            {"metadata_json": _provenance_meta(2, task="task-2")},
        ],
    )
    # Episode 1 stands in for a corrupt source episode outside this selection.
    # Export must not invoke conversion for it.
    fake_corpus["_fail_import_indexes"] = {1}
    dest = tmp_path / "out"
    export.export(dest, manifest=manifest, camera_keys=CAMS)
    assert [call["episode_index"] for call in fake_corpus["_import_calls"]] == [0, 2]

    info = json.loads((dest / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 60 + 70  # episodes 0 and 2
    assert info["splits"]["train"] == "0:2"
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


def _provenance_json(**overrides: object) -> str:
    return json.dumps({**json.loads(_provenance_meta(0)), **overrides})


def test_export_registry_matches_written_features(fake_corpus: dict, tmp_path: Path) -> None:
    """All written non-video features retain their exact source declarations."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    source_path = Path(fake_corpus["cache_dir"]) / "data/chunk-000/file-000.parquet"
    source = pq.read_table(source_path)
    for name in ("index", "episode_index", "frame_index"):
        source = source.set_column(
            source.schema.get_field_index(name), name, source[name].cast(pa.int64())
        )
    source = source.append_column("task_index", pa.array([0] * source.num_rows, pa.int64()))
    source = source.append_column(
        "next.done", pa.array([i % 2 == 0 for i in range(source.num_rows)], pa.bool_())
    )
    pq.write_table(source, source_path)
    features = fake_corpus["info"]["features"]
    for name in ("index", "episode_index", "frame_index", "task_index", "next.done"):
        features[name] = {
            "dtype": "bool" if name == "next.done" else "int64",
            "shape": [1],
            "names": [name],
        }
    features["unwritten.float"] = {"dtype": "float32", "shape": [1], "names": None}
    features["unwritten.integer"] = {"dtype": "int64", "shape": [1], "names": None}
    (tmp_path / "meta/info.json").write_text(json.dumps(fake_corpus["info"]))
    manifest = _fake_manifest(
        tmp_path,
        [{"metadata_json": _provenance_meta(0)}, {"metadata_json": _provenance_meta(2)}],
    )
    destination = tmp_path / "out"

    export.export(destination, manifest=manifest, camera_keys=CAMS[:1])

    registry = json.loads((destination / "meta/info.json").read_text())["features"]
    assert {key for key, value in registry.items() if value["dtype"] == "video"} == {CAMS[0]}
    data_features = {key: value for key, value in registry.items() if value["dtype"] != "video"}
    for exported_file in sorted((destination / "data").rglob("*.parquet")):
        exported = pq.read_table(exported_file)
        assert data_features == {name: features[name] for name in exported.column_names}
        assert set(registry) == set(exported.column_names) | {CAMS[0]}
        for name in ("index", "episode_index", "frame_index", "task_index"):
            assert exported.schema.field(name).type == pa.int64()
        assert exported.schema.field("next.done").type == pa.bool_()


@pytest.mark.parametrize(
    ("mismatch", "differing_field"),
    [
        ("type", "index"),
        ("nullability", "index"),
        ("field_metadata", "index"),
        ("schema_metadata", "schema metadata"),
        ("missing_field", "action"),
        ("extra_field", "extra"),
        ("field_order", "index"),
    ],
)
def test_export_refuses_written_schema_mismatch(
    fake_corpus: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
    differing_field: str,
) -> None:
    """A changed schema in the second data file aborts publication with a useful error."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    manifest = _fake_manifest(
        tmp_path,
        [{"metadata_json": _provenance_meta(0)}, {"metadata_json": _provenance_meta(2)}],
    )
    write_table = pq.write_table

    def write_changed_schema(table: pa.Table, where: Path) -> None:
        if where.parent.name == "chunk-000" and where.name == "file-001.parquet":
            first_field = table.schema.field(0)
            match mismatch:
                case "type":
                    table = table.set_column(
                        0, first_field.name, table.column(0).cast(pa.float64())
                    )
                case "nullability":
                    table = table.cast(table.schema.set(0, first_field.with_nullable(False)))
                case "field_metadata":
                    table = table.cast(
                        table.schema.set(0, first_field.with_metadata({b"changed": b"true"}))
                    )
                case "schema_metadata":
                    table = table.replace_schema_metadata({b"changed": b"true"})
                case "missing_field":
                    table = table.drop_columns(["action"])
                case "extra_field":
                    table = table.append_column("extra", pa.array([0] * table.num_rows))
                case "field_order":
                    table = table.select(list(reversed(table.column_names)))
        write_table(table, where)

    monkeypatch.setattr(pq, "write_table", write_changed_schema)
    destination = tmp_path / "out"
    destination.mkdir()
    (destination / "sentinel.txt").write_text("previous export")
    with pytest.raises(ValueError, match="schema differs from source") as error:
        export.export(destination, manifest=manifest, camera_keys=CAMS)
    message = str(error.value)
    assert "data/chunk-000/file-001.parquet" in message
    assert "schema differs from source" in message
    assert differing_field in message
    assert (destination / "sentinel.txt").read_text() == "previous export"
    assert sorted(path.name for path in destination.iterdir()) == ["sentinel.txt"]


def test_export_preserves_source_data_schema(fake_corpus: dict, tmp_path: Path) -> None:
    """Renumbered rows retain source float widths, array shapes, and metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    source_path = Path(fake_corpus["cache_dir"]) / "data/chunk-000/file-000.parquet"
    source_schema = pa.schema(
        [
            pa.field("index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("frame_index", pa.int64()),
            pa.field("timestamp", pa.float32()),
            pa.field(
                "observation.state",
                pa.list_(pa.float32(), 6),
                metadata={b"source": b"synthetic state feature"},
            ),
            pa.field("action", pa.list_(pa.float32(), 6)),
        ],
        metadata={b"source": b"synthetic LeRobot float32 features"},
    )
    source = pq.read_table(source_path).cast(source_schema)
    pq.write_table(source, source_path)
    manifest = _fake_manifest(
        tmp_path,
        [
            {
                "metadata_json": _provenance_json(
                    source_dataset=str(tmp_path), source_episode_index=2, task="task-2"
                )
            }
        ],
    )
    destination = tmp_path / "out"

    export.export(destination, manifest=manifest, camera_keys=CAMS)

    exported = pq.read_table(destination / "data/chunk-000/file-000.parquet")
    assert exported.schema.equals(pq.read_schema(source_path), check_metadata=True)
    assert exported["episode_index"].to_pylist() == [0] * LENGTHS[2]
    assert exported["index"].to_pylist() == list(range(LENGTHS[2]))
    for feature in ("timestamp", "observation.state", "action"):
        assert exported[feature] == source[feature].slice(OFFSETS[2], LENGTHS[2])


@pytest.mark.parametrize(
    ("manifest_rows", "expected_message"),
    [
        pytest.param(
            [
                {"metadata_json": _provenance_meta(0)},
                {
                    "metadata_json": _provenance_json(
                        source_dataset="lerobot/other",
                        source_revision="b" * 40,
                        source_episode_index=1,
                        task="task-1",
                    )
                },
            ],
            "mixes source repositories",
            id="mixed-repositories",
        ),
        pytest.param(
            [{"metadata_json": _provenance_json(source_revision="main")}],
            "not an immutable commit sha",
            id="nonimmutable-revision",
        ),
        pytest.param(
            [{"metadata_json": None}], "lacks LeRobot provenance", id="missing-provenance"
        ),
        pytest.param(
            [{"metadata_json": _provenance_meta(9)}],
            "source episodes not present",
            id="missing-source-episode",
        ),
        pytest.param(
            [{"metadata_json": _provenance_meta(0)}, {"metadata_json": _provenance_meta(0)}],
            "duplicate source episode indexes",
            id="duplicate-episode",
        ),
    ],
)
def test_export_refuses_an_unexportable_manifest(
    fake_corpus: dict, tmp_path: Path, manifest_rows: list[dict], expected_message: str
) -> None:
    manifest = _fake_manifest(tmp_path, manifest_rows)
    dest = tmp_path / "out"
    with pytest.raises(ValueError, match=expected_message):
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

    Export materializes the selected source episode through the public entry
    point and passes the camera keys the user asked to export.
    """
    manifest = _fake_manifest(tmp_path, [{"metadata_json": _provenance_meta(0)}])
    export.export(
        tmp_path / "out",
        manifest=manifest,
        camera_keys=("observation.images.up", "observation.images.side"),
    )
    assert fake_corpus["_import_calls"], "export must drive the public importer"
    call = fake_corpus["_import_calls"][0]
    assert call["dataset_repo"] == "lerobot/fake"
    assert call["revision"] == "a" * 40
    assert call["episode_index"] == 0
    assert call["camera_keys"] == ("observation.images.up", "observation.images.side")


def test_export_refuses_selected_broken_episode_without_replacing_destination(
    fake_corpus: dict, tmp_path: Path
) -> None:
    """A selected episode failure leaves the previously published export intact."""
    fake_corpus["_fail_import_indexes"] = {1}
    manifest = _fake_manifest(tmp_path, [{"metadata_json": _provenance_meta(1)}])
    dest = tmp_path / "out"
    dest.mkdir()
    marker = dest / "existing.txt"
    marker.write_text("previous export")

    with pytest.raises(ValueError, match="synthetic conversion failure for episode 1"):
        export.export(dest, manifest=manifest, camera_keys=CAMS)

    assert marker.read_text() == "previous export"
    assert sorted(path.name for path in dest.iterdir()) == ["existing.txt"]


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


def test_export_refuses_frame_referencing_unpublished_task(
    fake_corpus: dict, tmp_path: Path
) -> None:
    """#529: frame task_index referencing an unpublished task must be refused.

    When an episode's data frames carry a task_index referencing a task that the
    single-task output does not publish, export must fail loudly before writing
    destination files, naming episode, frame, and index.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Add task_index column to source data: frame 0 has task_index=0, frame 1 has task_index=1
    src_pq = Path(fake_corpus["cache_dir"]) / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(str(src_pq))
    task_indices = [0] * table.num_rows
    task_indices[1] = 1  # frame 1 points to task 1 (unpublished in single-task episode)
    table = table.append_column("task_index", pa.array(task_indices, pa.int64()))
    pq.write_table(table, str(src_pq))

    manifest = _fake_manifest(
        tmp_path,
        [{"metadata_json": _provenance_meta(0, task="pick cup")}],
    )
    dest = tmp_path / "out_refuse"
    with pytest.raises(
        ValueError,
        match=r"source episode 0 frame 1: task_index 1 references an unpublished task",
    ):
        export.export(dest, manifest=manifest, camera_keys=CAMS)
    assert not dest.exists()


def test_export_preserves_valid_task_index(fake_corpus: dict, tmp_path: Path) -> None:
    """A task_index matching the published task (index 0) exports cleanly."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    src_pq = Path(fake_corpus["cache_dir"]) / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(str(src_pq))
    task_indices = [0] * table.num_rows
    table = table.append_column("task_index", pa.array(task_indices, pa.int64()))
    pq.write_table(table, str(src_pq))

    manifest = _fake_manifest(
        tmp_path,
        [{"metadata_json": _provenance_meta(0, task="pick cup")}],
    )
    dest = tmp_path / "out_valid_task"
    export.export(dest, manifest=manifest, camera_keys=CAMS)

    out_pq = dest / "data" / "chunk-000" / "file-000.parquet"
    out_table = pq.read_table(str(out_pq))
    assert "task_index" in out_table.column_names
    assert out_table.column("task_index").to_pylist() == [0] * LENGTHS[0]


def test_validate_v3_rejects_corrupted_task_index(fake_corpus: dict, tmp_path: Path) -> None:
    """_validate_v3 refuses staged datasets where task_index resolves out-of-bounds."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    dest = _exported_dataset(fake_corpus, tmp_path)
    data_pq = dest / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(str(data_pq))
    corrupted_indices = [0] * table.num_rows
    corrupted_indices[3] = 99  # unpublished task index
    table = table.append_column("task_index", pa.array(corrupted_indices, pa.int64()))
    pq.write_table(table, str(data_pq))

    with pytest.raises(
        ValueError,
        match=r"episode 0 frame 3: task_index 99 references an unpublished task",
    ):
        export._validate_v3(dest)


def test_export_multitask_episode_preserves_task_indices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#632: multi-task episodes with task_index >= 1 must export cleanly."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    corpus_root = tmp_path / "corpus_multitask"
    info = two_camera_v3_info()
    write_v3_corpus(
        corpus_root,
        info=info,
        episode_rows=[
            CorpusEpisodeRow(
                episode_index=0,
                length=60,
                dataset_from_index=0,
                video_to_timestamp=2.0,
                tasks=("pick cup", "place cup"),
            )
        ],
    )
    for cam in CAMS:
        vdir = corpus_root / "videos" / cam / "chunk-000"
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "file-000.mp4").write_bytes(f"fake-mp4-{cam}".encode())

    # Write task_index containing both 0 and 1
    src_pq = corpus_root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(str(src_pq))
    task_indices = [0 if i < 30 else 1 for i in range(table.num_rows)]
    table = table.append_column("task_index", pa.array(task_indices, pa.int64()))
    pq.write_table(table, str(src_pq))

    corpus = {"info": info, "cache_dir": corpus_root}
    _install_fake_import(corpus, corpus_root, monkeypatch)

    manifest = _fake_manifest(
        tmp_path,
        [{"metadata_json": _provenance_meta(0, task="pick cup")}],
    )
    dest = tmp_path / "out_multitask"
    export.export(dest, manifest=manifest, camera_keys=CAMS)

    # Check exported data parquet preserves task_indices
    out_pq = dest / "data" / "chunk-000" / "file-000.parquet"
    out_table = pq.read_table(str(out_pq))
    assert "task_index" in out_table.column_names
    assert out_table.column("task_index").to_pylist() == task_indices

    # Check exported episode parquet has both tasks
    conn = duckdb.connect()
    ep_rows = conn.execute(
        f"SELECT tasks FROM read_parquet('{dest}/meta/episodes/chunk-000/file-000.parquet')"
    ).fetchall()
    conn.close()
    assert ep_rows[0][0] == ["pick cup", "place cup"]


def test_export_multitask_refuses_out_of_bounds_task_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#632: multi-task episodes with task_index exceeding published tasks must fail before write."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    corpus_root = tmp_path / "corpus_multitask_oob"
    info = two_camera_v3_info()
    write_v3_corpus(
        corpus_root,
        info=info,
        episode_rows=[
            CorpusEpisodeRow(
                episode_index=0,
                length=60,
                dataset_from_index=0,
                video_to_timestamp=2.0,
                tasks=("pick cup", "place cup"),
            )
        ],
    )
    for cam in CAMS:
        vdir = corpus_root / "videos" / cam / "chunk-000"
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "file-000.mp4").write_bytes(f"fake-mp4-{cam}".encode())

    # Frame 10 carries task_index=2 (unpublished since only 2 tasks exist: 0, 1)
    src_pq = corpus_root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(str(src_pq))
    task_indices = [0] * table.num_rows
    task_indices[10] = 2
    table = table.append_column("task_index", pa.array(task_indices, pa.int64()))
    pq.write_table(table, str(src_pq))

    corpus = {"info": info, "cache_dir": corpus_root}
    _install_fake_import(corpus, corpus_root, monkeypatch)

    manifest = _fake_manifest(
        tmp_path,
        [{"metadata_json": _provenance_meta(0, task="pick cup")}],
    )
    dest = tmp_path / "out_multitask_oob"
    with pytest.raises(
        ValueError,
        match=r"source episode 0 frame 10: task_index 2 references an unpublished task",
    ):
        export.export(dest, manifest=manifest, camera_keys=CAMS)
    assert not dest.exists()


def test_export_multi_episode_different_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#632: multi-episode selection with different task indices exports cleanly."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    corpus_root = tmp_path / "corpus_multi_ep"
    info = two_camera_v3_info()
    write_v3_corpus(
        corpus_root,
        info=info,
        episode_rows=[
            CorpusEpisodeRow(
                episode_index=0,
                length=60,
                dataset_from_index=0,
                video_to_timestamp=2.0,
                tasks=("task-0",),
            ),
            CorpusEpisodeRow(
                episode_index=1,
                length=65,
                dataset_from_index=60,
                video_to_timestamp=2.2,
                tasks=("task-1",),
            ),
        ],
    )
    for cam in CAMS:
        vdir = corpus_root / "videos" / cam / "chunk-000"
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "file-000.mp4").write_bytes(f"fake-mp4-{cam}".encode())

    # Episode 0 frames have task_index=0, Episode 1 frames have task_index=1
    src_pq = corpus_root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(str(src_pq))
    task_indices = [0] * 60 + [1] * 65
    table = table.append_column("task_index", pa.array(task_indices, pa.int64()))
    pq.write_table(table, str(src_pq))

    corpus = {"info": info, "cache_dir": corpus_root}
    _install_fake_import(corpus, corpus_root, monkeypatch)

    manifest = _fake_manifest(
        tmp_path,
        [
            {"metadata_json": _provenance_meta(0, task="task-0")},
            {"metadata_json": _provenance_meta(1, task="task-1")},
        ],
    )
    dest = tmp_path / "out_multi_ep"
    export.export(dest, manifest=manifest, camera_keys=CAMS)
    assert dest.exists()
    dest_tasks = dest / "meta" / "tasks.parquet"
    assert dest_tasks.exists()
    t_table = pq.read_table(str(dest_tasks))
    assert t_table.column("task_index").to_pylist() == [0, 1]
    assert t_table.column("task").to_pylist() == ["task-0", "task-1"]


def test_fetch_task_registry_downloads_from_hub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_fetch_task_registry downloads remote meta/tasks.parquet into cache_dir."""
    import huggingface_hub
    import pyarrow as pa
    import pyarrow.parquet as pq

    remote_file = tmp_path / "remote_tasks.parquet"
    t_table = pa.Table.from_arrays(
        [pa.array([0, 1], type=pa.int64()), pa.array(["task_a", "task_b"], type=pa.string())],
        names=["task_index", "task"],
    )
    pq.write_table(t_table, remote_file)

    download_calls = []

    def _fake_hf_hub_download(repo_id: str, filename: str, **kwargs: object) -> str:
        download_calls.append((repo_id, filename))
        if filename == "meta/tasks.parquet":
            dest_target = kwargs.get("local_dir")
            assert isinstance(dest_target, Path)
            target = dest_target / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(remote_file, target)
            return str(target)
        raise FileNotFoundError("not found")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_hf_hub_download)

    cache = tmp_path / "cache"
    cache.mkdir()
    export._fetch_task_registry("lerobot/mock-repo", "a" * 40, cache)
    assert (cache / "meta" / "tasks.parquet").exists()
    cached_table = pq.read_table(str(cache / "meta" / "tasks.parquet"))
    assert cached_table.column("task").to_pylist() == ["task_a", "task_b"]


def test_fetch_task_registry_fails_loudly_on_download_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_fetch_task_registry fails loudly when download encounters an error."""
    import huggingface_hub

    def _failing_download(*args: object, **kwargs: object) -> str:
        raise RuntimeError("network failure downloading task registry")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _failing_download)

    cache = tmp_path / "cache_fail"
    cache.mkdir()
    with pytest.raises(RuntimeError, match="network failure downloading task registry"):
        export._fetch_task_registry("lerobot/mock-repo", "a" * 40, cache)


def test_read_task_registry_helper(tmp_path: Path) -> None:
    """_read_task_registry parses both parquet and jsonl formats."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Empty dir returns empty dict
    meta_empty = tmp_path / "empty"
    meta_empty.mkdir()
    assert export._read_task_registry(meta_empty) == {}

    # Parquet registry
    meta_pq = tmp_path / "pq"
    meta_pq.mkdir()
    pq_table = pa.Table.from_arrays(
        [pa.array([0, 10], type=pa.int64()), pa.array(["task-0", "task-10"], type=pa.string())],
        names=["task_index", "task"],
    )
    pq.write_table(pq_table, meta_pq / "tasks.parquet")
    assert export._read_task_registry(meta_pq) == {0: "task-0", 10: "task-10"}

    # JSONL registry
    meta_jl = tmp_path / "jl"
    meta_jl.mkdir()
    jl_lines = '{"task_index": 1, "task": "task-1"}\n{"task_index": 5, "task": "task-5"}\n'
    (meta_jl / "tasks.jsonl").write_text(jl_lines)
    assert export._read_task_registry(meta_jl) == {1: "task-1", 5: "task-5"}


def test_export_refuses_frame_task_index_missing_from_sparse_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-validation refuses frames with task_index missing from source task registry.

    If source task registry has indices 0 and 10, a frame with task_index 1 must be
    refused even though 1 < len(registry).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    corpus_root = tmp_path / "corpus_sparse_reg_refuse"
    info = two_camera_v3_info()
    write_v3_corpus(
        corpus_root,
        info=info,
        episode_rows=[
            CorpusEpisodeRow(
                episode_index=0,
                length=60,
                dataset_from_index=0,
                video_to_timestamp=2.0,
                tasks=("task-0", "task-10"),
            )
        ],
    )
    for cam in CAMS:
        vdir = corpus_root / "videos" / cam / "chunk-000"
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "file-000.mp4").write_bytes(f"fake-mp4-{cam}".encode())

    # Write source tasks.parquet with sparse indices [0, 10]
    tasks_table = pa.Table.from_arrays(
        [pa.array([0, 10], type=pa.int64()), pa.array(["task-0", "task-10"], type=pa.string())],
        names=["task_index", "task"],
    )
    pq.write_table(tasks_table, corpus_root / "meta" / "tasks.parquet")

    # Frame 5 carries task_index=1 (missing from registry [0, 10])
    src_pq = corpus_root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(str(src_pq))
    task_indices = [0] * table.num_rows
    task_indices[5] = 1
    table = table.append_column("task_index", pa.array(task_indices, pa.int64()))
    pq.write_table(table, str(src_pq))

    corpus = export._read_corpus_from_cache(corpus_root)
    _install_fake_import(corpus, corpus_root, monkeypatch)

    manifest = _fake_manifest(
        tmp_path,
        [{"metadata_json": _provenance_meta(0, task="task-0")}],
    )
    dest = tmp_path / "out_sparse_refuse"
    with pytest.raises(
        ValueError,
        match=r"source episode 0 frame 5: task_index 1 references an unpublished task",
    ):
        export.export(dest, manifest=manifest, camera_keys=CAMS)
    assert not dest.exists()


def test_validate_v3_refuses_frame_task_index_missing_from_sparse_registry(
    tmp_path: Path,
) -> None:
    """_validate_v3 refuses frames when task_index is missing from destination registry."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    dataset_dir = tmp_path / "dataset_sparse_validate"
    dataset_dir.mkdir()
    meta_dir = dataset_dir / "meta"
    meta_dir.mkdir()
    (meta_dir / "info.json").write_text(
        json.dumps(
            {
                "code": "LeRobotDataset/v3",
                "codebase_version": "v3.0",
                "total_episodes": 1,
                "total_tasks": 2,
                "splits": {"train": "0:1"},
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            }
        )
    )

    # Registry with indices [0, 10]
    tasks_table = pa.Table.from_arrays(
        [pa.array([0, 10], type=pa.int64()), pa.array(["task-0", "task-10"], type=pa.string())],
        names=["task_index", "task"],
    )
    pq.write_table(tasks_table, meta_dir / "tasks.parquet")

    # Episode metadata
    ep_dir = meta_dir / "episodes" / "chunk-000"
    ep_dir.mkdir(parents=True)
    ep_table = pa.Table.from_pydict(
        {
            "episode_index": [0],
            "length": [10],
            "dataset_from_index": [0],
            "dataset_to_index": [10],
            "data/chunk_index": [0],
            "data/file_index": [0],
            "tasks": [["task-0", "task-10"]],
        }
    )
    pq.write_table(ep_table, ep_dir / "file-000.parquet")

    # Data file where frame 5 has task_index 1 (not in [0, 10])
    data_dir = dataset_dir / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    data_table = pa.Table.from_pydict(
        {
            "index": list(range(10)),
            "task_index": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        }
    )
    pq.write_table(data_table, data_dir / "file-000.parquet")

    with pytest.raises(
        ValueError,
        match=r"episode 0 frame 5: task_index 1 references an unpublished task",
    ):
        export._validate_v3(dataset_dir)


def test_export_sparse_registry_preserves_valid_indices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Multi-task dataset with sparse registry indices [0, 10] exports cleanly."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    corpus_root = tmp_path / "corpus_sparse_valid"
    info = two_camera_v3_info()
    write_v3_corpus(
        corpus_root,
        info=info,
        episode_rows=[
            CorpusEpisodeRow(
                episode_index=0,
                length=60,
                dataset_from_index=0,
                video_to_timestamp=2.0,
                tasks=("task-0", "task-10"),
            )
        ],
    )
    for cam in CAMS:
        vdir = corpus_root / "videos" / cam / "chunk-000"
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "file-000.mp4").write_bytes(f"fake-mp4-{cam}".encode())

    # Write source tasks.parquet with sparse indices [0, 10]
    tasks_table = pa.Table.from_arrays(
        [pa.array([0, 10], type=pa.int64()), pa.array(["task-0", "task-10"], type=pa.string())],
        names=["task_index", "task"],
    )
    pq.write_table(tasks_table, corpus_root / "meta" / "tasks.parquet")

    # Frames have task_index 0 and 10
    src_pq = corpus_root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(str(src_pq))
    task_indices = [0 if i < 30 else 10 for i in range(table.num_rows)]
    table = table.append_column("task_index", pa.array(task_indices, pa.int64()))
    pq.write_table(table, str(src_pq))

    corpus = export._read_corpus_from_cache(corpus_root)
    _install_fake_import(corpus, corpus_root, monkeypatch)

    manifest = _fake_manifest(
        tmp_path,
        [{"metadata_json": _provenance_meta(0, task="task-0")}],
    )
    dest = tmp_path / "out_sparse_valid"
    export.export(dest, manifest=manifest, camera_keys=CAMS)

    assert dest.exists()
    dest_tasks = dest / "meta" / "tasks.parquet"
    assert dest_tasks.exists()
    t_table = pq.read_table(str(dest_tasks))
    # Official loader uses iloc[task_index], so unused slots are padded; source
    # IDs 0 and 10 are unchanged in both the registry and the data file.
    assert t_table.column("task_index").to_pylist() == list(range(11))
    names = t_table.column("task").to_pylist()
    assert names[0] == "task-0"
    assert names[10] == "task-10"
    assert names[1:10] == [""] * 9
    pandas_meta = json.loads(t_table.schema.metadata[b"pandas"])
    assert pandas_meta["index_columns"] == ["task"]
    out_data = pq.read_table(str(dest / "data" / "chunk-000" / "file-000.parquet"))
    assert out_data.column("task_index").to_pylist() == task_indices


def test_export_info_json_matches_official_lerobot_v3_fields(
    fake_corpus: dict, tmp_path: Path
) -> None:
    """#778: exported info.json carries the fields the official loader reads first."""
    dest = _exported_dataset(fake_corpus, tmp_path)
    info = json.loads((dest / "meta" / "info.json").read_text())
    assert info["code"] == "LeRobotDataset/v3"
    assert info["codebase_version"] == "v3.0"
    assert info["total_episodes"] == 1
    assert info["total_frames"] == LENGTHS[0]
    assert info["total_tasks"] == 1
    assert info["splits"] == {"train": "0:1"}
    assert info["version"] == 1


def test_export_tasks_parquet_uses_official_positional_index(
    fake_corpus: dict, tmp_path: Path
) -> None:
    """#778: official loader takes the task string from the DataFrame index at iloc[task_index]."""
    import pyarrow.parquet as pq

    dest = _exported_dataset(fake_corpus, tmp_path)
    table = pq.read_table(str(dest / "meta" / "tasks.parquet"))
    pandas_meta = json.loads(table.schema.metadata[b"pandas"])
    assert pandas_meta["index_columns"] == ["task"]
    assert table.column("task").to_pylist()[0] == "task-0"
    assert table.column("task_index").to_pylist()[0] == 0


def test_validate_v3_rejects_missing_codebase_version(fake_corpus: dict, tmp_path: Path) -> None:
    """#778: a staged export without codebase_version is not official-loader compatible."""
    dest = _exported_dataset(fake_corpus, tmp_path)
    meta = dest / "meta" / "info.json"
    info = json.loads(meta.read_text())
    del info["codebase_version"]
    meta.write_text(json.dumps(info))
    with pytest.raises(ValueError, match=r"codebase_version is None, expected 'v3\.0'"):
        export._validate_v3(dest)


def test_validate_v3_rejects_list_form_splits(fake_corpus: dict, tmp_path: Path) -> None:
    """#778: official splits are a range string, not a list of episode names."""
    dest = _exported_dataset(fake_corpus, tmp_path)
    meta = dest / "meta" / "info.json"
    info = json.loads(meta.read_text())
    info["splits"] = {"train": ["episode_000000"]}
    meta.write_text(json.dumps(info))
    with pytest.raises(ValueError, match=r"splits.train is \['episode_000000'\], expected '0:1'"):
        export._validate_v3(dest)


@pytest.mark.requires_system_ffmpeg
def test_official_lerobot_loader_opens_typed_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#778 opt-in: export real MP4s and open them with the official loader.

    Skipped unless HFLOW_LEROBOT_LOADER_PYTHON points at an interpreter with
    ``lerobot`` installed (kept out of default CI so the lockfile is untouched).
    Without the official-field fix the subprocess fails with KeyError:
    'codebase_version'.
    """
    import os
    import subprocess

    loader = os.environ.get("HFLOW_LEROBOT_LOADER_PYTHON")
    if not loader:
        pytest.skip("set HFLOW_LEROBOT_LOADER_PYTHON to run the official-loader round-trip")

    import pyarrow as pa
    import pyarrow.parquet as pq

    width, height, fps, length = 64, 48, 30, 8
    corpus_root = tmp_path / "typed-corpus"
    info = {
        "fps": fps,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "robot_type": "so101",
        "features": {
            "action": {"dtype": "float32", "shape": [6], "names": None},
            "observation.state": {"dtype": "float32", "shape": [6], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            CAMS[0]: {"dtype": "video", "shape": [height, width, 3], "names": None},
            CAMS[1]: {"dtype": "video", "shape": [height, width, 3], "names": None},
        },
    }
    write_v3_corpus(
        corpus_root,
        info=info,
        episode_rows=[
            CorpusEpisodeRow(
                episode_index=0,
                length=length,
                dataset_from_index=0,
                video_to_timestamp=length / fps,
                tasks=("task-0", "task-10"),
            )
        ],
    )
    # Replace DuckDB-inferred decimals with the official v3 dtypes.
    state = [[float(i)] * 6 for i in range(length)]
    action = [[float(i) + 0.5] * 6 for i in range(length)]
    data = pa.table(
        {
            "index": pa.array(range(length), type=pa.int64()),
            "episode_index": pa.array([0] * length, type=pa.int64()),
            "frame_index": pa.array(range(length), type=pa.int64()),
            "timestamp": pa.array([i / fps for i in range(length)], type=pa.float32()),
            "task_index": pa.array([0, 10] * (length // 2), type=pa.int64()),
            "observation.state": pa.array(state, type=pa.list_(pa.float32(), 6)),
            "action": pa.array(action, type=pa.list_(pa.float32(), 6)),
        }
    )
    pq.write_table(data, corpus_root / "data" / "chunk-000" / "file-000.parquet")
    pq.write_table(
        pa.table({"task_index": [0, 10], "task": ["task-0", "task-10"]}),
        corpus_root / "meta" / "tasks.parquet",
    )
    (corpus_root / "meta" / "info.json").write_text(json.dumps(info))

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("system ffmpeg is required to encode decodable camera videos")
    for cam, color in zip(CAMS, ("red", "blue"), strict=True):
        vdir = corpus_root / "videos" / cam / "chunk-000"
        vdir.mkdir(parents=True, exist_ok=True)
        encoded = vdir / "file-000.mp4"
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"color=c={color}:s={width}x{height}:r={fps}:d={length / fps}",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(encoded),
            ],
            check=True,
            capture_output=True,
        )

    corpus = export._read_corpus_from_cache(corpus_root)
    _install_fake_import(corpus, corpus_root, monkeypatch)
    dest = tmp_path / "typed-export"
    manifest = _fake_manifest(tmp_path, [{"metadata_json": _provenance_meta(0)}])
    export.export(dest, manifest=manifest, camera_keys=CAMS)

    loader_script = tmp_path / "load_official.py"
    loader_script.write_text(
        "\n".join(
            [
                "import json",
                "import sys",
                "from importlib.metadata import version",
                "from pathlib import Path",
                "from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata",
                'assert version("lerobot") == "0.4.4"',
                "root = Path(sys.argv[1])",
                'meta = LeRobotDatasetMetadata("local/export", root=root)',
                'assert meta.info["codebase_version"] == "v3.0"',
                "assert meta.total_tasks == 2",
                'assert meta.info["splits"]["train"] == "0:1"',
                'assert meta.tasks.iloc[0].name == "task-0"',
                'assert meta.tasks.iloc[10].name == "task-10"',
                'ds = LeRobotDataset("local/export", root=root, video_backend="pyav")',
                "assert len(ds) == 8",
                "for i in range(len(ds)):",
                "    item = ds[i]",
                "    task_index = (0, 10)[i % 2]",
                '    assert int(item["task_index"]) == task_index',
                '    assert item["task"] == f"task-{task_index}"',
                '    for cam in ("observation.images.up", "observation.images.side"):',
                "        tensor = item[cam]",
                "        shape = tuple(tensor.shape[-3:])",
                "        assert shape in {(3, 48, 64), (48, 64, 3)}",
                'print("OFFICIAL_LOADER_OK", len(ds), json.dumps({"task": item["task"]}))',
                "",
            ]
        )
    )
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
    result = subprocess.run(
        [loader, str(loader_script), str(dest)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        raise AssertionError("official loader failed:\n" + result.stdout + "\n" + result.stderr)
    assert "OFFICIAL_LOADER_OK" in result.stdout
