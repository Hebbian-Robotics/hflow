"""Success-label fixture: what hflow ships when the source carries an outcome.

Builds a v3-style corpus whose data parquet declares a `next.success` feature
(the collector-labeled outcome signal, exactly as real lerobot/pusht declares
it) with every frame False, and whose episodes parquet carries the per-episode
`stats/next.success/min|max` aggregates the real format writes. Runs the REAL
import path and prints the source outcome vs the shipped episode/v1 record.

Run: uv run --locked --all-extras python .zcode/success_fixture.py
"""

import json
import shutil
from pathlib import Path

import duckdb
from mcap.reader import make_reader

import hflow.importers.lerobot as prep

ROOT = Path(".zcode/success_fixture")
EPISODES = 2


def build_corpus(root: Path) -> dict:
    info = {
        "fps": 30,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [1]},
            "observation.state": {"dtype": "float32", "shape": [1]},
            "observation.images.up": {"dtype": "video", "shape": [480, 640, 3]},
            "observation.images.side": {"dtype": "video", "shape": [480, 640, 3]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "next.success": {"dtype": "bool", "shape": [1]},
        },
        "robot_type": "so101",
    }
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(json.dumps(info))

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
        "videos/observation.images.side/chunk_index",
        "videos/observation.images.side/file_index",
        "videos/observation.images.side/from_timestamp",
        "videos/observation.images.side/to_timestamp",
        "tasks",
        "stats/next.success/min",
        "stats/next.success/max",
    ]
    rows = [
        [
            i,
            1,
            "000",
            "000",
            i,
            i + 1,
            "000",
            "000",
            0.0,
            0.0,
            "000",
            "000",
            0.0,
            0.0,
            [f"task-{i}"],
            [False],
            [False],
        ]
        for i in range(EPISODES)
    ]
    ep_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ep_path.parent.mkdir(parents=True, exist_ok=True)
    vals = ",".join(
        "("
        + ",".join(
            "[" + ",".join(f"'{x}'" for x in v) + "]"
            if isinstance(v, list)
            else f"'{v}'"
            if isinstance(v, str)
            else str(v)
            for v in row
        )
        + ")"
        for row in rows
    )
    conn.execute(
        f"COPY (SELECT * FROM (VALUES {vals}) AS t({','.join(chr(34) + c + chr(34) for c in ep_cols)})) "
        f"TO '{str(ep_path).replace(chr(39), chr(39) * 2)}' (FORMAT parquet)"
    )

    data_rows = [[i, i, 0, 0.0, [0.0], [0.5], False] for i in range(EPISODES)]
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    dvals = ",".join(
        "("
        + ",".join(
            str(v)
            if not isinstance(v, (str, list))
            else (
                "'" + str(v) + "'"
                if isinstance(v, str)
                else "[" + ",".join(str(x) for x in v) + "]"
            )
            for v in row
        )
        + ")"
        for row in data_rows
    )
    conn.execute(
        f"COPY (SELECT * FROM (VALUES {dvals}) AS t(index, episode_index, frame_index, "
        'timestamp, "observation.state", action, "next.success")) '
        f"TO '{str(data_path).replace(chr(39), chr(39) * 2)}' (FORMAT parquet)"
    )
    conn.close()
    return {"info": info}


def main() -> None:
    shutil.rmtree(ROOT, ignore_errors=True)
    corpus = build_corpus(ROOT)
    output_dir = ROOT / "prepared"

    stubs = {
        "_hf_repo_info": lambda repo, revision: {"sha": "abc1234", "license": "apache-2.0"},
        "_fetch_info_json": lambda repo, rev, cache: corpus["info"],
        "_hf_tree": lambda repo, rev, path: (
            [{"path": "meta/episodes/chunk-000/file-000.parquet", "type": "file"}]
            if "episodes" in path
            else [{"path": "meta/info.json", "type": "file"}]
        ),
        "_transcode_mp4_to_h264": lambda mp4_path, gop, fps: [
            b"\x00\x00\x00\x01\x09\x10\x00\x00\x00\x01\x67\x42\x00"
            b"\x00\x00\x00\x01\x68\x88\x80\x00\x00\x00\x01\x65\x88"
        ],
        "_get_video_pts_times": lambda path: [0],
        "ffmpeg_version": lambda: "fixture-ffmpeg",
    }
    originals = {name: getattr(prep, name) for name in stubs}

    def fake_download(url: str, dest: Path, **kw: object) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if "meta/episodes" in url:
            import shutil

            shutil.copy(ROOT / "meta" / "episodes" / "chunk-000" / "file-000.parquet", dest)
        elif url.endswith("info.json"):
            import shutil

            shutil.copy(ROOT / "meta" / "info.json", dest)
        else:
            import shutil

            shutil.copy(ROOT / "data" / "chunk-000" / "file-000.parquet", dest)

    stubs["_download_file"] = fake_download
    try:
        for name, fake in stubs.items():
            setattr(prep, name, fake)
        prep.import_lerobot_dataset(
            dataset_repo="fake/repo",
            output_dir=output_dir,
            camera_keys=("observation.images.up", "observation.images.side"),
        )
    finally:
        for name, original in originals.items():
            setattr(prep, name, original)

    print("--- what the source declares ---")
    conn = duckdb.connect()
    src = conn.execute(
        'SELECT episode_index, "next.success" FROM read_parquet('
        f"'{ROOT / 'data' / 'chunk-000' / 'file-000.parquet'}') ORDER BY index"
    ).fetchall()
    for r in src:
        print(f"  data parquet frame (episode {r[0]}): next.success = {r[1]}")
    agg = conn.execute(
        'SELECT episode_index, "stats/next.success/max" FROM read_parquet('
        f"'{ROOT / 'meta' / 'episodes' / 'chunk-000' / 'file-000.parquet'}') ORDER BY episode_index"
    ).fetchall()
    for r in agg:
        print(f"  episodes parquet aggregate (episode {r[0]}): stats/next.success/max = {r[1]}")
    conn.close()

    print("--- what hflow shipped ---")
    for episode_path in sorted((output_dir / "landing").glob("*.mcap")):
        with episode_path.open("rb") as stream:
            for metadata in make_reader(stream).iter_metadata():
                if metadata.name == "episode/v1":
                    print(f"  {episode_path.name}: episode/v1 = {json.dumps(metadata.metadata)}")
                    break

    print("--- consequence ---")
    print("  source collector label: next.success = False (this demo never succeeded)")
    print('  shipped episode/v1:     "success": "true"')
    print("  catalog promotion:      success VARCHAR first-class column (catalog.py:59,68)")
    print("  buyer filter success=true: matches this failed demo")


if __name__ == "__main__":
    main()
