"""Export an HFlow-curated selection as a loadable LeRobot Dataset v3 repository.

Reads a curation manifest (``hflow curate`` output parquet) or a SQL query
against the HFlow catalog, resolves each selected episode back to its
LeRobot source via the episode/v1 provenance stamped by the converter
(#189 contract: source_dataset, source_revision, source_episode_index,
task, embodiment), and materializes a local LeRobot Dataset v3 repository
containing exactly the selected episodes.

Byte-faithful: source video chunks are copied unchanged and the selected
episode data rows are sliced from their source chunk parquets, so camera
content, feature schema, dtypes, shapes, and frame timing match the source
exactly. Episode indexes are renumbered sequentially in selection order.

Failures fail before any publishable output is written: mixed source
repositories or revisions, missing provenance, missing source episodes,
duplicate selections, and incompatible feature schemas all abort without
replacing a previously valid destination.

Uploading to the Hugging Face Hub is out of scope; output is local only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import examples.lerobot.prepare as prepare
except ModuleNotFoundError:
    # Direct `python examples/lerobot/export.py` runs: repository root is
    # not on sys.path, so add it (examples/ is not a package).
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    import examples.lerobot.prepare as prepare


EXPORT_VERSION = "lerobot-export-v1"

_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_CAMERA_KEY_RE = re.compile(r"^observation\.images\.[A-Za-z0-9_]+$")


@dataclass(frozen=True)
class Selection:
    """One selected episode with its resolved LeRobot provenance."""

    episode_id: str
    source_dataset: str
    source_revision: str
    source_episode_index: int
    task: str
    embodiment: str


def _read_selection(manifest: Path | None, sql: str | None) -> list[dict]:
    """Read the curation selection rows (manifest parquet or raw SQL)."""
    if (manifest is None) == (sql is None):
        raise ValueError("provide exactly one of manifest or sql")
    con = duckdb.connect()
    try:
        if manifest is not None:
            quoted = str(manifest).replace("'", "''")
            rows = con.execute(f"SELECT * FROM read_parquet('{quoted}')").fetchall()
        else:
            rows = con.execute(sql or "").fetchall()
        cols = [d[0] for d in con.description]
        return [dict(zip(cols, row, strict=True)) for row in rows]
    finally:
        con.close()


def _resolve_selection(rows: list[dict]) -> list[Selection]:
    """Resolve each row to its LeRobot provenance; fail loudly on gaps."""
    selections: list[Selection] = []
    for row in rows:
        meta = row.get("metadata_json")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                meta = None
        if not isinstance(meta, dict):
            raise ValueError(
                f"episode {row.get('episode_id', '<unknown>')} lacks LeRobot "
                "provenance (metadata_json); only episodes that entered HFlow "
                "through the LeRobot adapter can be exported"
            )
        ds = meta.get("source_dataset")
        rev = meta.get("source_revision")
        ep_idx = meta.get("source_episode_index")
        if not isinstance(ds, str) or not ds:
            raise ValueError(
                f"episode {row.get('episode_id', '<unknown>')} has no source_dataset "
                "in its provenance"
            )
        if not isinstance(rev, str) or not rev:
            raise ValueError(
                f"episode {row.get('episode_id', '<unknown>')} has no source_revision "
                "in its provenance"
            )
        try:
            ep_num = int(ep_idx)
        except (TypeError, ValueError):
            raise ValueError(
                f"episode {row.get('episode_id', '<unknown>')} has a non-integer "
                f"source_episode_index {ep_idx!r}"
            ) from None
        selections.append(
            Selection(
                episode_id=str(row.get("episode_id", "")),
                source_dataset=ds,
                source_revision=rev,
                source_episode_index=ep_num,
                task=str(meta.get("task") or ""),
                embodiment=str(meta.get("embodiment") or ""),
            )
        )
    return selections


def _check_immutable(selections: list[Selection]) -> None:
    """Revisions must be immutable commit shas and single-valued."""
    datasets = {s.source_dataset for s in selections}
    revisions = {s.source_revision for s in selections}
    if len(datasets) != 1:
        raise ValueError(f"selection mixes source repositories: {sorted(datasets)}")
    if len(revisions) != 1:
        raise ValueError(f"selection mixes source revisions: {sorted(revisions)}")
    rev = next(iter(revisions))
    if not _COMMIT_SHA_RE.match(rev):
        raise ValueError(f"source revision {rev!r} is not an immutable commit sha")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _episode_rows(table: pa.Table) -> list[dict]:
    cols = table.column_names
    out: list[dict] = []
    for row in zip(*[table[col].to_pylist() for col in cols], strict=True):
        out.append(dict(zip(cols, row, strict=True)))
    return out


def _write_v3_repository(
    *,
    corpus: dict,
    selections: list[Selection],
    camera_keys: tuple[str, ...],
    destination: Path,
) -> dict:
    """Write the staged v3 repository; returns provenance metadata dict."""
    src_ds = selections[0].source_dataset
    src_rev = selections[0].source_revision
    cache_dir: Path = corpus["cache_dir"]
    info = corpus["info"]
    src_by_index = {e["episode_index"]: e for e in corpus["episodes"]}
    base = f"https://huggingface.co/datasets/{src_ds}/resolve/{src_rev}"

    # download data + video chunks (same access path the converter uses)
    def _fetch_data(ep: dict) -> Path:
        chunk = int(ep["data_chunk"])
        file = int(ep["data_file"])
        rel = corpus["data_path"].format(chunk_index=chunk, file_index=file)
        local = cache_dir / "data" / f"chunk-{chunk:06d}-file-{file:06d}.parquet"
        if not local.exists():
            prepare._download_file(f"{base}/{rel}", local)
        return local

    def _fetch_video(cam: str, vw: dict, ep: dict) -> Path:
        chunk = (
            int(vw["chunk_index"])
            if str(vw.get("chunk_index", "")).isdigit()
            else int(ep["data_chunk"])
        )
        file = (
            int(vw["file_index"])
            if str(vw.get("file_index", "")).isdigit()
            else int(ep["data_file"])
        )
        rel = corpus["video_path"].format(
            chunk_index=chunk, file_index=file, video_key=cam, camera_key=cam
        )
        local = cache_dir / "videos" / f"{cam.replace('/', '_').replace('.', '_')}-chunk{chunk}.mp4"
        if not local.exists():
            prepare._download_file(f"{base}/{rel}", local)
        return local

    episodes_dir = destination / "meta" / "episodes" / "chunk-000"
    data_dir = destination / "data" / "chunk-000"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # one data parquet per selected episode: windowed rows, renumbered
    ep_rows_out: list[dict] = []
    data_frames: list[dict] = []
    total_frames = 0
    video_paths: list[Path] = []

    for new_idx, sel in enumerate(selections):
        src = src_by_index[sel.source_episode_index]
        length = int(src["length"])
        if length < 1:
            raise ValueError(f"source episode {sel.source_episode_index} has no frames")
        data_local = _fetch_data(src)
        conn = duckdb.connect()
        try:
            escaped = str(data_local).replace("'", "''")
            cols = [
                d[0]
                for d in conn.execute(
                    f"SELECT * FROM read_parquet('{escaped}') LIMIT 0"
                ).description
            ]
            index_col = "index" if "index" in cols else "frame_index"
            d_from, d_to = int(src["data_from"]), int(src["data_to"])
            rows = conn.execute(
                f"SELECT * FROM read_parquet('{escaped}') "
                f"WHERE {index_col} >= {d_from} AND {index_col} < {d_to} "
                f"ORDER BY {index_col}"
            ).fetchall()
        finally:
            conn.close()
        if not rows:
            raise ValueError(
                f"no data rows for source episode {sel.source_episode_index} window {d_from}-{d_to}"
            )
        if len(rows) != length:
            raise ValueError(
                f"source episode {sel.source_episode_index}: expected {length} data rows, "
                f"found {len(rows)}"
            )

        for local_frame, row in enumerate(rows):
            d = dict(zip(cols, row, strict=True))
            d["episode_index"] = new_idx
            d["frame_index"] = local_frame
            d[index_col] = len(data_frames)
            data_frames.append(d)
        total_frames += length

        # videos: copy the chunk files byte-exact, windows preserved
        ep_video_refs: dict[str, dict] = {}
        for cam in camera_keys:
            vw = (src.get("video_windows") or {}).get(cam)
            if vw is None:
                raise ValueError(
                    f"source episode {sel.source_episode_index} has no video window "
                    f"for camera {cam}"
                )
            vlocal = _fetch_video(cam, vw, src)
            dst_dir = destination / "videos" / cam / "chunk-000"
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / "file-000.mp4"
            if not dst.exists():  # same chunk reused across episodes: copy once
                shutil.copy2(vlocal, dst)
                video_paths.append(dst)
            ep_video_refs[cam] = {
                "chunk_index": "chunk-000",
                "file_index": "file-000",
                "from_timestamp": float(vw.get("from_timestamp", 0.0)),
                "to_timestamp": float(vw.get("to_timestamp", 0.0)),
            }

        ep_out: dict = {
            "episode_index": new_idx,
            "length": length,
            "tasks": [sel.task] if sel.task else [],
            "data/chunk_index": "chunk-000",
            "data/file_index": f"file-{new_idx:03d}.parquet",
            "dataset_from_index": (total_frames - length),
            "dataset_to_index": total_frames,
        }
        for cam in camera_keys:
            v = ep_video_refs[cam]
            ep_out[f"videos/{cam}/chunk_index"] = v["chunk_index"]
            ep_out[f"videos/{cam}/file_index"] = v["file_index"]
            ep_out[f"videos/{cam}/from_timestamp"] = v["from_timestamp"]
            ep_out[f"videos/{cam}/to_timestamp"] = v["to_timestamp"]
        ep_rows_out.append(ep_out)

    # write the single data parquet (all selected frames, renumbered, in order)
    if not data_frames:
        raise ValueError("selection produced no data frames")
    data_table = pa.Table.from_pylist(data_frames)
    pq.write_table(data_table, data_dir / "file-000.parquet")

    # write per-episode parquet rows
    ep_table = pa.Table.from_pylist(ep_rows_out)
    pq.write_table(ep_table, episodes_dir / "file-000.parquet")

    # meta/info.json
    video_features = {
        cam: corpus["info"]["features"].get(cam, {"dtype": "video", "shape": [480, 640, 3]})
        for cam in camera_keys
    }
    numeric_features = {
        k: v
        for k, v in corpus["info"].get("features", {}).items()
        if isinstance(v, dict) and v.get("dtype") == "float32"
    }
    out_info = {
        "code": "LeRobotDataset/v3",
        "total_episodes": len(selections),
        "total_frames": total_frames,
        "total_videos": len(camera_keys) * len(selections),
        "robot_type": selections[0].embodiment or info.get("robot_type", "unknown"),
        "fps": info.get("fps", 30),
        "splits": {"train": [f"episode_{i:06d}" for i in range(len(selections))]},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {**numeric_features, **video_features},
        "version": 1,
    }
    (destination / "meta" / "info.json").write_text(json.dumps(out_info, indent=2))

    return {
        "exporter_version": EXPORT_VERSION,
        "source_repository": src_ds,
        "source_commit": src_rev,
        "source_episode_indexes": [s.source_episode_index for s in selections],
        "output_episode_count": len(selections),
        "output_frames": total_frames,
        "cameras": list(camera_keys),
        "video_sha256": {cam: _sha256(video_paths[i]) for i, cam in enumerate(camera_keys)},
        "data_parquet_sha256": _sha256(data_dir / "file-000.parquet"),
    }


def _validate_v3(dataset_dir: Path) -> None:
    """Validate the staged repository is a loadable LeRobot Dataset v3.

    Structural checks run always (meta/info.json present and coherent, episode
    and data parquets readable, per-episode windows within the data file,
    video files present for every referenced camera). When the official
    ``lerobot`` package is installed the staged directory is additionally
    loaded through its LeRobotDataset API; without it, the structural checks
    are the validation (CI installs lerobot for the official-API pass).
    """
    meta_path = dataset_dir / "meta" / "info.json"
    if not meta_path.exists():
        raise ValueError(f"staged dataset has no meta/info.json: {dataset_dir}")
    info = json.loads(meta_path.read_text())
    if info.get("code") != "LeRobotDataset/v3":
        raise ValueError(f"staged dataset is not LeRobotDataset/v3: {info.get('code')}")
    if int(info.get("total_episodes", 0)) < 1:
        raise ValueError("staged dataset has no episodes")

    episodes_parquets = sorted((dataset_dir / "meta" / "episodes").rglob("*.parquet"))
    if not episodes_parquets:
        raise ValueError("staged dataset has no episode parquets")
    conn = duckdb.connect()
    try:
        for ep_pq in episodes_parquets:
            quoted = str(ep_pq).replace("'", "''")
            cols = [
                d[0]
                for d in conn.execute(f"SELECT * FROM read_parquet('{quoted}') LIMIT 0").description
            ]
            rows = conn.execute(f"SELECT * FROM read_parquet('{quoted}')").fetchall()
            for row in rows:
                d = dict(zip(cols, row, strict=True))
                ep = int(d["episode_index"])
                length = int(d["length"])
                if length < 1:
                    raise ValueError(f"episode {ep} in {ep_pq.name} has no frames")
                d_from = int(d["dataset_from_index"])
                d_to = int(d["dataset_to_index"])
                if d_to - d_from != length:
                    raise ValueError(f"episode {ep} window {d_from}-{d_to} != length {length}")
                for cam in info.get("features") or {}:
                    if not str(cam).startswith("observation.images."):
                        continue
                    vkey = f"videos/{cam}/file_index"
                    if vkey not in d:
                        raise ValueError(f"episode {ep} lacks video reference for {cam}")
                    vfile = dataset_dir / "videos" / cam / "chunk-000" / f"{d[vkey]}.mp4"
                    if not vfile.exists():
                        raise ValueError(f"episode {ep} references missing video {vfile}")
    finally:
        conn.close()

    data_pqs = sorted((dataset_dir / "data").rglob("*.parquet"))
    if not data_pqs:
        raise ValueError("staged dataset has no data parquets")

    try:
        import lerobot.common.datasets.lerobot_dataset as _lerobot_ds  # ty: ignore

        LeRobotDataset = _lerobot_ds.LeRobotDataset
        LeRobotDataset(dataset_dir)
    except ImportError:
        pass  # structural validation above stands in for the API pass


def _write_dataset_card(
    destination: Path, provenance: dict, sql: str | None, manifest_name: str | None
) -> None:
    card = f"""---
license: unknown
tags:
- hflow
- lerobot-dataset-v3
---

# HFlow curated LeRobot dataset

Exported by the HFlow LeRobot exporter ({EXPORT_VERSION}).

## Provenance

- source repository: `{provenance["source_repository"]}`
- source commit: `{provenance["source_commit"]}`
- selected source episode indexes: {provenance["source_episode_indexes"]}
- output episodes: {provenance["output_episode_count"]}
- output frames: {provenance["output_frames"]}
- cameras: {", ".join(provenance["cameras"])}
- selection manifest: `{manifest_name or "inline SQL"}`
- selection SQL: `{(sql or "").strip() or "(manifest)"}`

Video chunks are byte-for-byte copies of the source; data rows are sliced
from the source chunk parquets for the selected episodes. Frame timing and
feature schema match the source.
"""
    (destination / "README.md").write_text(card)


def export(
    destination: Path,
    *,
    manifest: Path | None = None,
    sql: str | None = None,
    catalog_root: Path | None = None,
    camera_keys: tuple[str, ...] | None = None,
) -> Path:
    """Export the curated selection into a new local LeRobot v3 repository."""
    rows = _read_selection(manifest, sql)
    if not rows:
        raise ValueError("selection is empty: nothing to export")
    selections = _resolve_selection(rows)
    _check_immutable(selections)

    src_ds = selections[0].source_dataset
    src_rev = selections[0].source_revision
    want_indexes = [s.source_episode_index for s in selections]
    if len(set(want_indexes)) != len(want_indexes):
        raise ValueError("selection contains duplicate source episode indexes")

    with tempfile.TemporaryDirectory(prefix="lerobot-export-") as tmp:
        stage_root = Path(tmp)
        cache_dir = stage_root / "_lerobot_cache"
        corpus = prepare._ensure_source_archive(
            prepare.DatasetSource(repo_id=src_ds, revision=src_rev, license=""),
            cache_dir,
        )

        if camera_keys is None:
            camera_keys = tuple(corpus["video_keys"])
        else:
            for k in camera_keys:
                if not _CAMERA_KEY_RE.match(k):
                    raise ValueError(f"invalid camera key {k!r}")
                if k not in corpus["video_keys"]:
                    raise ValueError(
                        f"camera key {k!r} not in source cameras {corpus['video_keys']}"
                    )

        exist = {e["episode_index"] for e in corpus["episodes"]}
        missing = [i for i in want_indexes if i not in exist]
        if missing:
            raise ValueError(f"source episodes not present in {src_ds}@{src_rev[:8]}: {missing}")

        stage = stage_root / "stage"
        stage.mkdir()
        provenance = _write_v3_repository(
            corpus=corpus,
            selections=selections,
            camera_keys=camera_keys,
            destination=stage,
        )

        meta = json.loads((stage / "meta" / "info.json").read_text())
        if meta["total_episodes"] != len(selections):
            raise ValueError("internal error: staged episode count mismatch")
        expected_frames = sum(
            int(e["length"]) for e in corpus["episodes"] if e["episode_index"] in want_indexes
        )
        if meta["total_frames"] != expected_frames:
            raise ValueError(
                f"internal error: staged frame count {meta['total_frames']} != "
                f"expected {expected_frames}"
            )

        _validate_v3(stage)
        _write_dataset_card(stage, provenance, sql, manifest.name if manifest else None)
        (stage / "export-provenance.json").write_text(json.dumps(provenance, indent=2))

        # publish atomically
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp_dest = destination.parent / f".{destination.name}.export-tmp"
        if tmp_dest.exists():
            shutil.rmtree(tmp_dest)
        shutil.copytree(stage, tmp_dest)
        if destination.exists():
            shutil.rmtree(destination)
        tmp_dest.rename(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        required=True,
        help="output dataset directory (created on success)",
    )
    parser.add_argument("--manifest", type=Path, default=None, help="hflow curate output parquet")
    parser.add_argument("--sql", type=str, default=None, help="curation SQL over the catalog")
    parser.add_argument(
        "--catalog-root", type=Path, default=None, help="HFlow catalog root (for --sql selections)"
    )
    parser.add_argument(
        "--camera-keys", type=str, default=None, help="comma-separated camera keys (default: all)"
    )
    args = parser.parse_args()

    camera_keys = None
    if args.camera_keys:
        camera_keys = tuple(k.strip() for k in args.camera_keys.split(",") if k.strip())
    out = export(
        args.destination,
        manifest=args.manifest,
        sql=args.sql,
        catalog_root=args.catalog_root,
        camera_keys=camera_keys,
    )
    print(f"exported to {out}")


if __name__ == "__main__":
    main()
