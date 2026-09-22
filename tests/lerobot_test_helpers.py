"""Shared fakes for the LeRobot Dataset v3 importer, exporter, and workflow tests.

The synthetic corpus follows the v3 on-disk layout (``meta/info.json``, one
``meta/episodes`` parquet, one data parquet), so each test builds only the
parts it varies: episode windows, frame rate, and which video files exist.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pytest

import hflow.importers.lerobot as lerobot_importer
from hflow.storage import StorageRoot

TWO_CAMERA_KEYS = ("observation.images.up", "observation.images.side")
V3_DATA_PATH_TEMPLATE = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
V3_VIDEO_PATH_TEMPLATE = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
SINGLE_EPISODE_METADATA_SHARD = "meta/episodes/chunk-000/file-000.parquet"
SINGLE_DATA_FILE = "data/chunk-000/file-000.parquet"


def exactly(message: str) -> str:
    """A ``match=`` pattern pinning the whole message, metacharacters and all.

    #405 asks for the metadata refusals to be byte-identical, and several
    contain a ``.`` (``meta/info.json``), which unescaped would also match
    ``meta/infoXjson``.
    """
    return rf"^{re.escape(message)}$"


def two_camera_v3_info(
    *, frames_per_second: int = 30, camera_frame_shape: Sequence[int] = (480, 640, 3)
) -> dict:
    """info.json for two cameras and 6-dim float32 state and action."""
    return {
        "fps": frames_per_second,
        "data_path": V3_DATA_PATH_TEMPLATE,
        "video_path": V3_VIDEO_PATH_TEMPLATE,
        "features": {
            "action": {"dtype": "float32", "shape": [6]},
            "observation.state": {"dtype": "float32", "shape": [6]},
            TWO_CAMERA_KEYS[0]: {"dtype": "video", "shape": list(camera_frame_shape)},
            TWO_CAMERA_KEYS[1]: {"dtype": "video", "shape": list(camera_frame_shape)},
            "timestamp": {"dtype": "float32", "shape": [1]},
        },
        "robot_type": "so101",
    }


@dataclass(frozen=True)
class CorpusEpisodeRow:
    """One ``meta/episodes`` row. Every camera shares the same video window."""

    episode_index: int
    length: int
    dataset_from_index: int
    video_to_timestamp: float
    tasks: tuple[str, ...]
    data_chunk_index: int | str = 0
    data_file_index: int | str = 0
    video_chunk_index: int | str = 0
    video_file_index: int | str = 0


def _sql_literal(value: object) -> str:
    if isinstance(value, list | tuple):
        return "[" + ",".join(_sql_literal(item) for item in value) + "]"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return str(value)


def _quoted_sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def write_v3_data_parquet(
    destination: Path,
    *,
    episode_lengths: Sequence[int],
    frames_per_second: float,
    timestamps_as_double: bool,
) -> None:
    """Contiguous data rows: ``index`` counts across episodes, frames restart per episode.

    Each frame's state is its frame number and its action that number plus
    0.5, six dimensions each. ``timestamps_as_double`` rewrites the
    timestamp column from DuckDB's inferred DECIMAL to DOUBLE.
    """
    data_rows = []
    row_index = 0
    for episode_index, length in enumerate(episode_lengths):
        for frame_index in range(length):
            state = "[" + ",".join(str(float(frame_index)) for _ in range(6)) + "]"
            action = "[" + ",".join(str(float(frame_index + 0.5)) for _ in range(6)) + "]"
            timestamp = round(frame_index / frames_per_second, 6)
            data_rows.append([row_index, episode_index, frame_index, timestamp, state, action])
            row_index += 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    quoted_destination = _quoted_sql_path(destination)
    data_values = ",".join("(" + ",".join(str(value) for value in row) + ")" for row in data_rows)
    connection = duckdb.connect()
    try:
        connection.execute(
            f"COPY (SELECT * FROM (VALUES {data_values}) AS "
            't(index, episode_index, frame_index, timestamp, "observation.state", action)) '
            f"TO '{quoted_destination}' (FORMAT parquet)"
        )
        if timestamps_as_double:
            connection.execute(
                "COPY (SELECT index, episode_index, frame_index, "
                "CAST(timestamp AS DOUBLE) AS timestamp, "
                f"\"observation.state\", action FROM read_parquet('{quoted_destination}')) "
                f"TO '{quoted_destination}' (FORMAT parquet)"
            )
    finally:
        connection.close()


def write_v3_corpus(
    corpus_root: Path,
    *,
    info: dict,
    episode_rows: Sequence[CorpusEpisodeRow],
    camera_keys: Sequence[str] = TWO_CAMERA_KEYS,
    timestamps_as_double: bool = False,
) -> None:
    """Write info.json, the single episode-metadata shard, and the data parquet.

    Video files are left to the caller, since each test stubs them differently.
    """
    (corpus_root / "meta").mkdir(parents=True, exist_ok=True)
    (corpus_root / "meta" / "info.json").write_text(json.dumps(info))

    episode_columns = [
        "episode_index",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
    ]
    for camera_key in camera_keys:
        episode_columns += [
            f"videos/{camera_key}/chunk_index",
            f"videos/{camera_key}/file_index",
            f"videos/{camera_key}/from_timestamp",
            f"videos/{camera_key}/to_timestamp",
        ]
    episode_columns.append("tasks")

    episode_values = []
    for row in episode_rows:
        values: list[object] = [
            row.episode_index,
            row.length,
            row.data_chunk_index,
            row.data_file_index,
            row.dataset_from_index,
            row.dataset_from_index + row.length,
        ]
        for _camera_key in camera_keys:
            values += [row.video_chunk_index, row.video_file_index, 0.0, row.video_to_timestamp]
        values.append(list(row.tasks))
        episode_values.append("(" + ",".join(_sql_literal(value) for value in values) + ")")

    episode_metadata_path = corpus_root / SINGLE_EPISODE_METADATA_SHARD
    episode_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    quoted_columns = ",".join(f'"{column}"' for column in episode_columns)
    connection = duckdb.connect()
    try:
        connection.execute(
            f"COPY (SELECT * FROM (VALUES {','.join(episode_values)}) AS t({quoted_columns})) "
            f"TO '{_quoted_sql_path(episode_metadata_path)}' (FORMAT parquet)"
        )
    finally:
        connection.close()

    write_v3_data_parquet(
        corpus_root / SINGLE_DATA_FILE,
        episode_lengths=[row.length for row in episode_rows],
        frames_per_second=info["fps"],
        timestamps_as_double=timestamps_as_double,
    )


def split_episode_metadata_into_shards(
    corpus_root: Path, episode_indexes_by_shard: Mapping[str, tuple[int, ...]]
) -> None:
    """Rewrite the single episode-metadata parquet as several v3 shards.

    A shard path may name the original file, which is then overwritten with
    its own subset.
    """
    single_shard = corpus_root / SINGLE_EPISODE_METADATA_SHARD
    connection = duckdb.connect()
    try:
        connection.execute(
            "CREATE TABLE all_episodes AS SELECT * FROM read_parquet('"
            + _quoted_sql_path(single_shard)
            + "')"
        )
        for shard_path, episode_indexes in episode_indexes_by_shard.items():
            shard_file = corpus_root / shard_path
            shard_file.parent.mkdir(parents=True, exist_ok=True)
            connection.execute(
                f"COPY (SELECT * FROM all_episodes WHERE episode_index IN {episode_indexes}) "
                f"TO '{_quoted_sql_path(shard_file)}' (FORMAT parquet)"
            )
    finally:
        connection.close()


def stub_hub_repo_info(
    monkeypatch: pytest.MonkeyPatch,
    *,
    resolved_sha: str | None = None,
    license_name: str = "apache-2.0",
) -> None:
    """Resolve every revision to ``resolved_sha``, or to itself when it is None."""
    monkeypatch.setattr(
        lerobot_importer,
        "_hf_repo_info",
        lambda _repo, revision: {
            "sha": revision if resolved_sha is None else resolved_sha,
            "license": license_name,
        },
    )


def stub_hub_download(
    monkeypatch: pytest.MonkeyPatch, supply_file: Callable[[str, Path], None]
) -> None:
    """Supply fixture files at the SDK boundary, using its repository layout."""

    def download(_repo_id: str, filename: str, *, local_dir: Path, **_kwargs: object) -> str:
        destination = local_dir / filename
        if not destination.exists():
            supply_file(filename, destination)
        return str(destination)

    monkeypatch.setattr(lerobot_importer, "hf_hub_download", download)


def stub_single_shard_hub_corpus(
    monkeypatch: pytest.MonkeyPatch, corpus_root: Path, info: dict
) -> None:
    """Serve a corpus written by ``write_v3_corpus`` as the Hub would.

    Any ``meta/episodes`` request gets the one metadata shard, ``info.json``
    gets the info file, and every other file gets the data parquet.
    """
    monkeypatch.setattr(lerobot_importer, "_fetch_info_json", lambda _repo, _rev, _cache: info)
    monkeypatch.setattr(
        lerobot_importer,
        "_hf_episode_metadata_files",
        lambda _repo, _rev: [SINGLE_EPISODE_METADATA_SHARD],
    )

    def supply_corpus_file(filename: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if "meta/episodes" in filename:
            shutil.copy(corpus_root / SINGLE_EPISODE_METADATA_SHARD, destination)
        elif filename.endswith("info.json"):
            shutil.copy(corpus_root / "meta" / "info.json", destination)
        else:
            shutil.copy(corpus_root / SINGLE_DATA_FILE, destination)

    stub_hub_download(monkeypatch, supply_corpus_file)


def publish_staged_episode(
    storage: StorageRoot, staged: Path, episode_index: int
) -> lerobot_importer._PublishedEpisode:
    """Publish ``staged`` at the episode's landing key and return its receipt."""
    published_uri = storage.publish(staged, lerobot_importer._landing_relative_key(episode_index))
    return {
        "uri": published_uri,
        "content_id": lerobot_importer.content_episode_id(staged),
        "size_bytes": staged.stat().st_size,
    }
