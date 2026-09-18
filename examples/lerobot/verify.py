"""Verify an exported LeRobot Dataset v3 repository in a clean process.

Runs without importing HFlow: reads ``meta/info.json``, opens every
episode and data parquet, resolves every episode's data and camera
references against the files that exist, and -- when the official
``lerobot`` package is importable in this interpreter -- also loads the
repository through ``LeRobotDataset`` and enumerates its episodes. The
exporter renumbers episodes in selection order, so a three-episode
selection is verified as ``--expect 0,1,2``.

    python examples/lerobot/verify.py ./data/lerobot_workflow/v3 --expect 0,1,2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _format_ref(template: str, **values: object) -> Path:
    try:
        return Path(template.format(**values))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"reference template {template!r} cannot format {values!r}: {error}"
        ) from None


def verify(destination: Path, expect: list[int]) -> dict:
    """Return verification facts; raise ValueError on the first defect."""
    meta_path = destination / "meta" / "info.json"
    if not meta_path.exists():
        raise ValueError(f"exported dataset has no meta/info.json: {destination}")
    info = json.loads(meta_path.read_text())
    if info.get("code") != "LeRobotDataset/v3":
        raise ValueError(f"exported dataset is not LeRobotDataset/v3: {info.get('code')}")

    expected = sorted(expect)
    if int(info.get("total_episodes", -1)) != len(expected):
        raise ValueError(
            f"info.json declares {info.get('total_episodes')} episodes, expected {len(expected)}"
        )
    video_features = [
        key
        for key, value in (info.get("features") or {}).items()
        if isinstance(value, dict) and value.get("dtype") == "video"
    ]
    if not video_features:
        raise ValueError("info.json declares no video features")

    episode_parquets = sorted((destination / "meta" / "episodes").rglob("*.parquet"))
    data_parquets = sorted((destination / "data").rglob("*.parquet"))
    if not episode_parquets:
        raise ValueError(f"exported dataset has no episode parquets under {destination}")
    if not data_parquets:
        raise ValueError(f"exported dataset has no data parquets under {destination}")

    import duckdb

    connection = duckdb.connect()
    try:
        episode_indexes: list[int] = []
        total_frames = 0
        for episode_parquet in episode_parquets:
            quoted = str(episode_parquet).replace("'", "''")
            rows = connection.execute(f"SELECT * FROM read_parquet('{quoted}')").fetchall()
            columns = [d[0] for d in connection.description]
            for row in rows:
                record = dict(zip(columns, row, strict=True))
                episode_index = int(record["episode_index"])
                length = int(record["length"])
                if length < 1:
                    raise ValueError(f"episode {episode_index} has no frames")
                data_from = int(record.get("dataset_from_index", -1))
                data_to = int(record.get("dataset_to_index", -1))
                if data_to - data_from != length:
                    raise ValueError(
                        f"episode {episode_index} window {data_from}-{data_to} != length {length}"
                    )
                data_ref = destination / _format_ref(
                    info["data_path"],
                    chunk_index=record.get("data/chunk_index"),
                    file_index=record.get("data/file_index"),
                )
                if not data_ref.exists():
                    raise ValueError(f"episode {episode_index} references missing data {data_ref}")
                for feature in video_features:
                    video_ref = destination / _format_ref(
                        info["video_path"],
                        video_key=feature,
                        chunk_index=record.get(f"videos/{feature}/chunk_index"),
                        file_index=record.get(f"videos/{feature}/file_index"),
                    )
                    if not video_ref.exists():
                        raise ValueError(
                            f"episode {episode_index} references missing video {video_ref}"
                        )
                episode_indexes.append(episode_index)
                total_frames += length

        if sorted(set(episode_indexes)) != expected:
            raise ValueError(
                f"exported episodes {sorted(set(episode_indexes))} != expected {expected}"
            )

        data_rows = 0
        for data_parquet in data_parquets:
            quoted = str(data_parquet).replace("'", "''")
            count_row = connection.execute(
                f"SELECT COUNT(*) FROM read_parquet('{quoted}')"
            ).fetchone()
            if count_row is None:
                raise ValueError(f"count query returned no row for {data_parquet}")
            data_rows += int(count_row[0])
    finally:
        connection.close()

    if data_rows != total_frames:
        raise ValueError(f"data rows {data_rows} != declared episode frames {total_frames}")
    expected_frames = int(info.get("total_frames", -1))
    if expected_frames >= 0 and expected_frames != total_frames:
        raise ValueError(f"info.json total_frames {expected_frames} != counted {total_frames}")

    lerobot_load = _load_with_lerobot(destination, expected)
    return {
        "episodes": expected,
        "frames": total_frames,
        "video_features": video_features,
        "lerobot_load": lerobot_load,
    }


def _load_with_lerobot(destination: Path, expected: list[int]) -> str | None:
    """Load through the official package when importable; None when not installed."""
    try:
        from lerobot import LeRobotDataset  # ty: ignore[unresolved-import]
    except (ImportError, ModuleNotFoundError):
        return None
    dataset = LeRobotDataset(
        dataset_id=destination.name,
        root=str(destination.parent),
        local_files_only=True,
    )
    episodes = list(dataset.episodes)
    indexes = sorted(
        int(getattr(episode, "index", getattr(episode, "episode_index", -1)))
        for episode in episodes
    )
    if indexes != sorted(expected):
        raise ValueError(f"LeRobotDataset loaded episodes {indexes}, expected {sorted(expected)}")
    return f"LeRobotDataset loaded {len(indexes)} episode(s) (indexes {indexes})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="exported v3 repository directory")
    parser.add_argument("--expect", required=True, help="comma-separated episode indexes")
    args = parser.parse_args(argv)

    try:
        expected = sorted(int(part) for part in args.expect.split(",") if part.strip())
        facts = verify(args.destination, expected)
    except ValueError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    print(
        f"verify: {len(facts['episodes'])} episode(s) (indexes "
        f"{','.join(str(index) for index in facts['episodes'])}), "
        f"{facts['frames']} frame(s), {len(facts['video_features'])} camera feature(s)"
    )
    if facts["lerobot_load"] is None:
        print("verify: official LeRobotDataset load skipped (lerobot not importable here)")
    else:
        print(f"verify: {facts['lerobot_load']}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
