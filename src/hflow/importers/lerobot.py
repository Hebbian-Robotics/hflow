"""Import LeRobot Dataset v3 repositories as canonical MCAP episodes.

The converter reads repository metadata (feature schema, fps, episode
boundaries, video paths) instead of encoding dataset-specific assumptions.
Every selected camera is converted into its own foxglove.CompressedVideo
channel; numeric state and action schemas are derived from the declared
dtype and shape, failing loud before conversion when a feature is
unsupported.

The public :func:`import_lerobot_dataset` API and ``hflow import lerobot``
command are the supported entry points. The importer does not install or
import LeRobot itself: it reads Dataset v3 metadata and Parquet files through
HFlow's existing dependencies, downloads selected source media from the
Hugging Face Hub, and uses HFlow's managed FFmpeg build.
"""

import json
import logging
import math
import os
import struct
import subprocess
import tempfile
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, TypedDict

from mcap.writer import Writer as McapWriter

from hflow.ffmpeg import ffmpeg_path, ffmpeg_version, ffprobe_path
from hflow.transform import TransformConfig, write_canonical_episode

logger = logging.getLogger(__name__)

DEFAULT_REPO = "lerobot/pusht"
DEFAULT_REVISION = "main"
DEFAULT_OUTPUT_DIR = Path("./data/lerobot_pusht")
DEFAULT_CAMERA_KEY = "observation.image"

CONVERTER_VERSION = "lerobot-converter-v3"
PRESENTATION_TIMESTAMP_EPSILON_S = 0.050

# Timestamp handling
NANOSECONDS_PER_SECOND = 1_000_000_000
EPISODE_START_TIME_NS = 1_755_000_000_000_000_000
HUGGING_FACE_TOKEN_ENVIRONMENT_VARIABLES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


class _EpisodeRow(TypedDict):
    episode_index: int
    task: str
    length: int
    data_chunk: str
    data_file: str
    data_from: int
    data_to: int
    video_windows: NotRequired[dict[str, "_VideoWindow"]]


class _VideoWindow(TypedDict):
    chunk_index: str
    file_index: str
    from_timestamp: float
    to_timestamp: float


@dataclass(frozen=True)
class DatasetSource:
    repo_id: str
    revision: str
    license: str


class _DatasetRepositoryInformation(TypedDict):
    sha: str
    license: str


class _SourceArchive(TypedDict):
    info: dict
    fps: int | float
    data_path: str
    video_path: str
    episodes: list[_EpisodeRow]
    video_keys: list[str]
    numeric_features: dict[str, dict]
    cache_dir: Path
    dataset: DatasetSource


def _encode_cdr_float32_array(values: list[float] | tuple[float, ...]) -> bytes:
    """Encode a float32[N] array as ROS 2 CDR (XCDR1 little-endian).

    CDR encapsulation header (00 01 00 00) followed by the packed floats;
    byte-compatible with hflow's mcap_ros2 decoder.
    """
    encapsulation = b"\x00\x01\x00\x00"
    payload = struct.pack(f"<{len(values)}f", *(float(value) for value in values))
    return encapsulation + payload


def _transcode_mp4_to_h264(
    mp4_path: Path, gop_seconds: float, frames_per_second: float
) -> list[bytes]:
    """Transcode an mp4 to H.264 access units split on AUD markers."""
    keyframe_interval = max(1, round(gop_seconds * frames_per_second))
    ffmpeg_command = [
        str(ffmpeg_path()),
        "-y",
        "-i",
        str(mp4_path),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-g",
        str(keyframe_interval),
        "-keyint_min",
        str(keyframe_interval),
        "-sc_threshold",
        "0",
        "-x264-params",
        # bframes=0: B-frame streams lose their reorder-buffer tail through
        # the raw Annex B -> MP4 remux, undercounting decoded_frame_count (#250).
        "aud=1:bframes=0",
        "-f",
        "h264",
        "pipe:1",
    ]
    completed_process = subprocess.run(ffmpeg_command, capture_output=True, timeout=600)
    if completed_process.returncode != 0:
        raise RuntimeError(
            "ffmpeg transcode failed: " + completed_process.stderr.decode(errors="ignore")
        )
    return _split_h264_by_aud(completed_process.stdout)


def _split_h264_by_aud(h264_data: bytes) -> list[bytes]:
    """Split H.264 stream on AUD (0x00000109) boundaries."""
    units: list[bytes] = []
    pattern = b"\x00\x00\x00\x01\x09"
    offset = 0
    while True:
        boundary_index = h264_data.find(pattern, offset)
        if boundary_index == -1:
            break
        if boundary_index > offset:
            units.append(h264_data[offset:boundary_index])
        offset = boundary_index + 5
    if offset < len(h264_data):
        units.append(h264_data[offset:])
    return units


def _get_video_pts_times(mp4_path: Path) -> list[float]:
    completed_process = subprocess.run(
        [
            str(ffprobe_path()),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=pts_time",
            "-of",
            "csv=p=0",
            str(mp4_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed_process.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {completed_process.stderr}")
    times: list[float] = []
    for line in completed_process.stdout.splitlines():
        line = line.strip()
        if line and line != "N/A":
            times.append(float(line))
    return times


def _slice_video(
    video_path: Path,
    start_seconds: float,
    end_seconds: float,
    output_path: Path | None = None,
) -> Path:
    """Slice a video to a time window with stream copy (no re-encode)."""
    resolved_output_path = output_path or (
        video_path.parent / f"{video_path.stem}_slice{video_path.suffix}"
    )
    ffmpeg_command = [
        str(ffmpeg_path()),
        "-y",
        "-ss",
        f"{start_seconds:.6f}",
        "-to",
        f"{end_seconds:.6f}",
        "-i",
        str(video_path),
        "-c",
        "copy",
        "-an",
        str(resolved_output_path),
    ]
    completed_process = subprocess.run(ffmpeg_command, capture_output=True, timeout=600)
    if completed_process.returncode != 0:
        raise RuntimeError(
            "ffmpeg slice failed: " + completed_process.stderr.decode(errors="ignore")
        )
    return resolved_output_path


def _hf_repo_info(repo_id: str, revision: str) -> _DatasetRepositoryInformation:
    """Resolve the immutable commit sha and license for a HF dataset repo."""
    url = f"https://huggingface.co/api/datasets/{repo_id}/revision/{revision}"
    with urllib.request.urlopen(_hugging_face_request(url), timeout=60) as response:
        repository_information = json.loads(response.read().decode())
    if not isinstance(repository_information, dict):
        raise ValueError("Hugging Face repository response is not a JSON object")
    resolved_revision = repository_information.get("sha")
    if not isinstance(resolved_revision, str) or not resolved_revision.strip():
        raise ValueError(
            f"Hugging Face did not resolve {repo_id}@{revision} to an immutable commit"
        )
    card_data = repository_information.get("cardData")
    license_name = card_data.get("license") if isinstance(card_data, dict) else None
    return {
        "sha": resolved_revision,
        "license": str(license_name or "unknown"),
    }


def _hf_tree(repo_id: str, revision: str, path: str) -> list[dict]:
    """List files under a HF dataset tree path (recursive)."""
    url = f"https://huggingface.co/api/datasets/{repo_id}/tree/{revision}/{path}?recursive=true"
    with urllib.request.urlopen(_hugging_face_request(url), timeout=60) as response:
        tree_entries = json.loads(response.read().decode())
    if not isinstance(tree_entries, list) or not all(
        isinstance(tree_entry, dict) for tree_entry in tree_entries
    ):
        raise ValueError(f"Hugging Face tree response for {path!r} is not a list of objects")
    return tree_entries


def _hugging_face_request(url: str) -> urllib.request.Request:
    headers = {"User-Agent": "hflow-lerobot"}
    for environment_variable_name in HUGGING_FACE_TOKEN_ENVIRONMENT_VARIABLES:
        token = os.environ.get(environment_variable_name, "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
            break
    return urllib.request.Request(url, headers=headers)


def _fetch_info_json(repo_id: str, revision: str, cache_dir: Path) -> dict:
    meta_entries = _hf_tree(repo_id, revision, "meta")
    info_path = None
    for entry in meta_entries:
        if entry.get("path") == "meta/info.json" and entry.get("type") == "file":
            info_path = True
            break
    if not info_path:
        raise RuntimeError("meta/info.json not found; not a LeRobot v3 repository")
    load_url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/meta/info.json"
    with urllib.request.urlopen(_hugging_face_request(load_url), timeout=120) as response:
        dataset_information = json.loads(response.read().decode())
    if not isinstance(dataset_information, dict):
        raise ValueError("LeRobot meta/info.json is not a JSON object")
    (cache_dir / "meta").mkdir(parents=True, exist_ok=True)
    (cache_dir / "meta" / "info.json").write_text(json.dumps(dataset_information, indent=2))
    return dataset_information


def _download_file(url: str, destination_path: Path, chunk_size: int = 1 << 20) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination_path.name}.",
        suffix=".partial",
        dir=destination_path.parent,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        try:
            with urllib.request.urlopen(_hugging_face_request(url), timeout=300) as response:
                while chunk := response.read(chunk_size):
                    temporary_file.write(chunk)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_path.replace(destination_path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise


def _ensure_source_archive(dataset_source: DatasetSource, cache_dir: Path) -> _SourceArchive:
    """Download the corpus parquets and video chunks needed for the given episodes."""
    import duckdb

    dataset_base_url = (
        "https://huggingface.co/datasets/"
        f"{dataset_source.repo_id}/resolve/{dataset_source.revision}"
    )
    metadata_directory = cache_dir / "meta"
    dataset_information = _fetch_info_json(
        dataset_source.repo_id, dataset_source.revision, cache_dir
    )
    frames_per_second = dataset_information.get("fps")
    if (
        isinstance(frames_per_second, bool)
        or not isinstance(frames_per_second, int | float)
        or not math.isfinite(frames_per_second)
        or frames_per_second <= 0
    ):
        raise ValueError(
            f"LeRobot meta/info.json has invalid fps={frames_per_second!r}; "
            "FPS must be finite and positive"
        )
    data_path_template = dataset_information.get("data_path")
    if not isinstance(data_path_template, str) or not data_path_template.strip():
        raise ValueError("LeRobot meta/info.json must define a non-empty data_path template")
    video_path_template = dataset_information.get(
        "video_path", "videos/{camera_key}/{chunk_index:06d}/{file_index:06d}.mp4"
    )
    if not isinstance(video_path_template, str) or not video_path_template.strip():
        raise ValueError("LeRobot meta/info.json must define a non-empty video_path template")

    # Determine the episodes parquet location (v3 uses meta/episodes/*.parquet)
    episodes_metadata_directory = metadata_directory / "episodes"
    episodes_metadata_directory.mkdir(parents=True, exist_ok=True)
    episode_metadata_files: list[Path] = []
    entries = _hf_tree(dataset_source.repo_id, dataset_source.revision, "meta/episodes")
    for entry in entries:
        if entry.get("type") == "file" and entry["path"].endswith(".parquet"):
            destination_path = episodes_metadata_directory / Path(entry["path"]).name
            if not destination_path.exists():
                _download_file(f"{dataset_base_url}/{entry['path']}", destination_path)
            episode_metadata_files.append(destination_path)

    if not episode_metadata_files:
        raise RuntimeError("no meta/episodes parquet files found")

    # Index of per-episode data windows across chunks
    connection = duckdb.connect()
    try:
        episode_rows: list[_EpisodeRow] = []
        for episode_metadata_file in episode_metadata_files:
            parquet_episode_rows = connection.execute(
                f"""
                SELECT "episode_index", "tasks", "length",
                       "data/chunk_index", "data/file_index",
                       "dataset_from_index", "dataset_to_index"
                FROM read_parquet('{str(episode_metadata_file).replace("'", "''")}')
                ORDER BY "episode_index"
                """
            ).fetchall()
            for parquet_episode_row in parquet_episode_rows:
                tasks = parquet_episode_row[1]
                if isinstance(tasks, list):
                    task = str(tasks[0]) if tasks else ""
                else:
                    task = str(tasks or "")
                episode_rows.append(
                    {
                        "episode_index": int(parquet_episode_row[0]),
                        "task": task,
                        "length": int(parquet_episode_row[2]),
                        "data_chunk": str(parquet_episode_row[3]).split("/")[-1],
                        "data_file": str(parquet_episode_row[4]).split("/")[-1],
                        "data_from": int(parquet_episode_row[5]),
                        "data_to": int(parquet_episode_row[6]),
                    }
                )
        episode_rows.sort(key=lambda episode: episode["episode_index"])

        # Video window columns: videos/<camera>/{chunk_index,file_index,from_timestamp,to_timestamp}
        flattened_column_names = [
            column_description[0]
            for column_description in connection.execute(
                "SELECT * FROM read_parquet('"
                + str(episode_metadata_files[0]).replace("'", "''")
                + "') LIMIT 1"
            ).description
        ]
        video_keys = sorted(
            {
                column_name.split("/")[1]
                for column_name in flattened_column_names
                if column_name.startswith("videos/") and column_name.endswith("/from_timestamp")
            }
        )

        # Gather video windows per episode+camera
        video_windows_by_episode: dict[int, dict[str, _VideoWindow]] = {}
        video_window_selectors: list[str] = []
        for camera_key in video_keys:
            video_window_selectors += [
                f'"videos/{camera_key}/chunk_index" as "vc_{camera_key}",',
                f'"videos/{camera_key}/file_index" as "vf_{camera_key}",',
                f'"videos/{camera_key}/from_timestamp" as "vfrom_{camera_key}",',
                f'"videos/{camera_key}/to_timestamp" as "vto_{camera_key}",',
            ]
        video_window_select_sql = "episode_index, " + " ".join(video_window_selectors).rstrip(",")
        first_episode_metadata_file = str(episode_metadata_files[0]).replace("'", "''")
        video_window_rows = connection.execute(
            f"SELECT {video_window_select_sql} FROM read_parquet('{first_episode_metadata_file}')"
        ).fetchall()
        video_window_column_names = [
            column_description[0] for column_description in connection.description
        ]
        for video_window_row in video_window_rows:
            video_window_by_column = dict(
                zip(video_window_column_names, video_window_row, strict=True)
            )
            episode_index = int(video_window_by_column["episode_index"])
            video_windows_by_episode[episode_index] = {}
            for camera_key in video_keys:
                video_chunk_index = video_window_by_column.get(f"vc_{camera_key}")
                video_file_index = video_window_by_column.get(f"vf_{camera_key}")
                video_windows_by_episode[episode_index][camera_key] = {
                    "chunk_index": (
                        "" if video_chunk_index is None else str(video_chunk_index).split("/")[-1]
                    ),
                    "file_index": (
                        "" if video_file_index is None else str(video_file_index).split("/")[-1]
                    ),
                    "from_timestamp": float(
                        video_window_by_column.get(f"vfrom_{camera_key}") or 0.0
                    ),
                    "to_timestamp": float(video_window_by_column.get(f"vto_{camera_key}") or 0.0),
                }
        for episode_row in episode_rows:
            episode_row["video_windows"] = dict(
                video_windows_by_episode.get(episode_row["episode_index"], {})
            )
    finally:
        connection.close()

    dataset_features = dataset_information.get("features")
    if not isinstance(dataset_features, dict):
        raise ValueError("LeRobot meta/info.json must define a features object")
    video_keys_from_schema = sorted(
        feature_name
        for feature_name, feature_specification in dataset_features.items()
        if isinstance(feature_specification, dict) and feature_specification.get("dtype") == "video"
    )
    if not video_keys:
        video_keys = video_keys_from_schema
    numeric_features = {
        feature_name: feature_specification
        for feature_name, feature_specification in dataset_features.items()
        if isinstance(feature_specification, dict)
        and feature_specification.get("dtype") == "float32"
    }

    return {
        "info": dataset_information,
        "fps": frames_per_second,
        "data_path": data_path_template,
        "video_path": video_path_template,
        "episodes": episode_rows,
        "video_keys": video_keys,
        "numeric_features": numeric_features,
        "cache_dir": cache_dir,
        "dataset": dataset_source,
    }


@dataclass
class _NumericSchema:
    name: str
    dim: int


def _derive_numeric_schema(feature_name: str, feature_specification: dict) -> _NumericSchema:
    declared_dtype = feature_specification.get("dtype")
    declared_shape = feature_specification.get("shape") or []
    if declared_dtype != "float32":
        raise ValueError(
            f"unsupported feature {feature_name}: "
            f"dtype={declared_dtype}, shape={declared_shape} "
            "(only float32 fixed-width numeric vectors are supported)"
        )
    if (
        len(declared_shape) != 1
        or isinstance(declared_shape[0], bool)
        or not isinstance(declared_shape[0], int)
        or declared_shape[0] < 1
    ):
        raise ValueError(
            f"unsupported feature {feature_name}: "
            f"dtype={declared_dtype}, shape={declared_shape} "
            "(only 1-D fixed-width numeric vectors are supported)"
        )
    return _NumericSchema(name=feature_name, dim=int(declared_shape[0]))


def import_lerobot_dataset(
    dataset_repo: str = DEFAULT_REPO,
    revision: str = DEFAULT_REVISION,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    episode_index: int | None = None,
    camera_keys: str | Sequence[str] = (DEFAULT_CAMERA_KEY,),
) -> list[Path]:
    """Import selected LeRobot Dataset v3 episodes as canonical MCAP.

    ``dataset_repo`` is a Hugging Face dataset repository. ``revision`` may
    name a branch, tag, or commit; the importer resolves it to an immutable
    commit before downloading data and records that commit in episode
    metadata and the prepared manifest. ``camera_keys`` accepts a sequence of
    camera features; a comma-separated string is accepted for compatibility.
    When ``episode_index`` is omitted, every episode is imported.

    Dataset v3 video features and one-dimensional, fixed-width float32 state
    and action vectors are supported. Unsupported feature layouts fail before
    any episode is published. The returned paths identify the canonical MCAP
    episodes written under ``output_dir / "landing"``.
    """
    normalized_dataset_repo = dataset_repo.strip()
    normalized_revision = revision.strip()
    if not normalized_dataset_repo:
        raise ValueError("dataset_repo must not be empty")
    if not normalized_revision:
        raise ValueError("revision must not be empty")
    if episode_index is not None and (
        isinstance(episode_index, bool) or not isinstance(episode_index, int) or episode_index < 0
    ):
        raise ValueError("episode_index must be zero or greater")
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"output_dir is not a directory: {output_dir}")

    camera_key_arguments = (camera_keys,) if isinstance(camera_keys, str) else camera_keys
    resolved_camera_keys = tuple(
        camera_key.strip()
        for camera_key_argument in camera_key_arguments
        for camera_key in camera_key_argument.split(",")
        if camera_key.strip()
    )
    if not resolved_camera_keys:
        raise ValueError("camera_keys must name at least one video feature")
    if len(set(resolved_camera_keys)) != len(resolved_camera_keys):
        raise ValueError("camera_keys must not contain duplicates")

    repository_information = _hf_repo_info(normalized_dataset_repo, normalized_revision)
    cache_directory = output_dir / "_lerobot_cache"
    source_archive = _ensure_source_archive(
        DatasetSource(
            repo_id=normalized_dataset_repo,
            revision=repository_information["sha"],
            license=repository_information["license"],
        ),
        cache_directory,
    )

    for camera_key in resolved_camera_keys:
        if camera_key not in source_archive["video_keys"]:
            raise ValueError(
                f"camera key '{camera_key}' not found in dataset. "
                f"Available: {source_archive['video_keys']}"
            )

    # Numeric schemas derived from metadata (fail before any conversion)
    numeric_schemas = {
        feature_name: _derive_numeric_schema(feature_name, feature_specification)
        for feature_name, feature_specification in source_archive["numeric_features"].items()
        if feature_name in ("observation.state", "action")
        or feature_name.startswith("observation.")
    }
    missing_required_features = {"observation.state", "action"} - numeric_schemas.keys()
    if missing_required_features:
        raise ValueError(
            "LeRobot dataset is missing supported required features: "
            + ", ".join(sorted(missing_required_features))
        )

    episode_rows = source_archive["episodes"]
    if episode_index is not None and episode_index >= len(episode_rows):
        raise ValueError(
            f"episode_index {episode_index} is out of range for {len(episode_rows)} episode(s)"
        )
    selected_episode_indexes = (
        [episode_index] if episode_index is not None else list(range(len(episode_rows)))
    )
    canonical_episode_paths: list[Path] = []

    dataset_source = source_archive["dataset"]
    for selected_episode_index in selected_episode_indexes:
        canonical_episode_path = _convert_single_episode(
            source_archive=source_archive,
            dataset_source=dataset_source,
            output_dir=output_dir,
            episode_index=selected_episode_index,
            camera_keys=resolved_camera_keys,
            numeric_schemas=numeric_schemas,
            frames_per_second=int(source_archive["fps"]),
        )
        canonical_episode_paths.append(canonical_episode_path)

    manifest_path = output_dir / "prepared-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_contents = json.dumps(
        {
            "schema_version": 2,
            "dataset": {
                "repo_id": dataset_source.repo_id,
                "revision": dataset_source.revision,
                "license": dataset_source.license,
            },
            "camera_keys": list(resolved_camera_keys),
            "episodes_converted": len(canonical_episode_paths),
            "converter_version": CONVERTER_VERSION,
        },
        indent=2,
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{manifest_path.name}.",
        suffix=".partial",
        dir=manifest_path.parent,
        delete=False,
    ) as temporary_manifest:
        temporary_manifest_path = Path(temporary_manifest.name)
        try:
            temporary_manifest.write(manifest_contents)
            temporary_manifest.flush()
            os.fsync(temporary_manifest.fileno())
            temporary_manifest_path.replace(manifest_path)
        except BaseException:
            temporary_manifest_path.unlink(missing_ok=True)
            raise
    logger.info("wrote LeRobot import manifest %s", manifest_path)
    return canonical_episode_paths


def _convert_single_episode(
    source_archive: _SourceArchive,
    dataset_source: DatasetSource,
    output_dir: Path,
    episode_index: int,
    camera_keys: tuple[str, ...],
    numeric_schemas: dict[str, _NumericSchema],
    frames_per_second: int,
) -> Path:
    """Convert a single episode to canonical MCAP. Returns output path."""
    import duckdb

    episode_row = source_archive["episodes"][episode_index]
    if episode_row["length"] is None or episode_row["length"] < 1:
        raise ValueError(f"episode {episode_index} has no frames")

    dataset_base_url = (
        "https://huggingface.co/datasets/"
        f"{dataset_source.repo_id}/resolve/{dataset_source.revision}"
    )
    cache_directory = source_archive["cache_dir"]

    # Locate the data parquet for this episode
    data_chunk_index = episode_row["data_chunk"]
    data_file_index = episode_row["data_file"]
    data_relative_path = source_archive["data_path"].format(
        chunk_index=int(data_chunk_index), file_index=int(data_file_index)
    )
    local_data_path = (
        cache_directory
        / "data"
        / f"chunk-{int(data_chunk_index):06d}-file-{int(data_file_index):06d}.parquet"
    )
    if not local_data_path.exists():
        _download_file(f"{dataset_base_url}/{data_relative_path}", local_data_path)

    # Episode video windows per camera (v3 flat columns: videos/<cam>/from_timestamp etc.)
    video_time_window_by_camera: dict[str, tuple[float, float]] = {}
    video_metadata_by_camera: dict[str, _VideoWindow] = {}
    for camera_key in camera_keys:
        video_window = episode_row.get("video_windows", {}).get(camera_key)
        if video_window:
            video_time_window_by_camera[camera_key] = (
                video_window["from_timestamp"],
                video_window["to_timestamp"],
            )
            video_metadata_by_camera[camera_key] = video_window

    # Query the data window
    connection = duckdb.connect()
    try:
        # data window query uses `index` (row position in chunk file), not frame_index
        escaped_data_path = str(local_data_path).replace("'", "''")
        index_column_name = (
            "index"
            if "index"
            in [
                column_description[0]
                for column_description in connection.execute(
                    f"SELECT * FROM read_parquet('{escaped_data_path}') LIMIT 0"
                ).description
            ]
            else "frame_index"
        )
        data_start_index = int(episode_row["data_from"])
        data_end_index = int(episode_row["data_to"])
        episode_data_rows = connection.execute(
            f"SELECT * FROM read_parquet('{escaped_data_path}') "
            f"WHERE {index_column_name} >= {data_start_index} "
            f"AND {index_column_name} < {data_end_index} "
            f"ORDER BY {index_column_name}"
        ).fetchall()
        column_names = [column_description[0] for column_description in connection.description]
        if not episode_data_rows:
            raise ValueError(
                f"no data rows for episode {episode_index} window "
                f"{data_start_index}-{data_end_index}"
            )
    finally:
        connection.close()

    # Build feature arrays from the window
    def _feature_rows(feature_name: str) -> list | None:
        if feature_name not in column_names:
            return None
        feature_column_index = column_names.index(feature_name)
        return [episode_data_row[feature_column_index] for episode_data_row in episode_data_rows]

    state_rows = _feature_rows("observation.state")
    action_rows = _feature_rows("action")
    timestamp_rows = _feature_rows("timestamp")
    if state_rows is None or action_rows is None or timestamp_rows is None:
        raise RuntimeError("required features observation.state/action/timestamp missing")

    frame_count = len(state_rows)

    # Per-camera video: download chunk video, slice to episode window, transcode
    video_data_by_camera: dict[str, tuple[list[bytes], list[float]]] = {}
    for camera_key in camera_keys:
        camera_video_metadata = video_metadata_by_camera.get(camera_key, {})
        video_time_window = video_time_window_by_camera.get(camera_key)
        video_start_seconds = video_time_window[0] if video_time_window is not None else 0.0
        video_end_seconds = video_time_window[1] if video_time_window is not None else 0.0

        video_chunk_index = (
            str(camera_video_metadata.get("chunk_index"))
            if camera_video_metadata.get("chunk_index") is not None
            else (data_chunk_index or "0")
        )
        video_chunk_index = video_chunk_index.split("/")[-1]
        video_file_index = (
            str(camera_video_metadata.get("file_index"))
            if camera_video_metadata.get("file_index") is not None
            else (data_file_index or "0")
        )
        video_file_index = video_file_index.split("/")[-1]
        video_relative_path = source_archive["video_path"].format(
            chunk_index=int(video_chunk_index or 0),
            file_index=int(video_file_index or 0),
            video_key=camera_key,
            camera_key=camera_key,
        )
        local_video_path = (
            cache_directory
            / "videos"
            / (f"{camera_key.replace('/', '_').replace('.', '_')}-chunk{video_chunk_index}.mp4")
        )
        if not local_video_path.exists():
            _download_file(f"{dataset_base_url}/{video_relative_path}", local_video_path)

        if video_start_seconds > 0.0 or video_end_seconds > 0.0:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temporary_video_file:
                temporary_video_path = Path(temporary_video_file.name)
            try:
                sliced_video_path = _slice_video(
                    local_video_path,
                    video_start_seconds,
                    video_end_seconds,
                    temporary_video_path,
                )
                access_units = _transcode_mp4_to_h264(
                    sliced_video_path, 1.0, float(frames_per_second)
                )
                presentation_timestamps = _get_video_pts_times(sliced_video_path)
            finally:
                temporary_video_path.unlink(missing_ok=True)
        else:
            access_units = _transcode_mp4_to_h264(local_video_path, 1.0, float(frames_per_second))
            presentation_timestamps = _get_video_pts_times(local_video_path)

        if len(access_units) != frame_count:
            raise ValueError(
                f"camera {camera_key}: frame count mismatch: {frame_count} "
                f"parquet vs {len(access_units)} video access units"
            )
        if len(presentation_timestamps) != frame_count:
            raise ValueError(
                f"camera {camera_key}: PTS count mismatch: {frame_count} "
                f"vs {len(presentation_timestamps)}"
            )
        for frame_index in range(frame_count):
            if (
                abs(timestamp_rows[frame_index] - presentation_timestamps[frame_index])
                > PRESENTATION_TIMESTAMP_EPSILON_S
            ):
                raise ValueError(
                    f"camera {camera_key} timestamp disagreement at frame {frame_index}: "
                    f"parquet={timestamp_rows[frame_index]:.6f}s "
                    f"video_pts={presentation_timestamps[frame_index]:.6f}s"
                )

        video_data_by_camera[camera_key] = (access_units, presentation_timestamps)

    # Write MCAP
    output_file_name = f"lerobot_episode_{episode_index + 1:04d}.mcap"
    output_path = output_dir / "landing" / output_file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from foxglove_schemas_protobuf.CompressedVideo_pb2 import CompressedVideo
    from mcap_protobuf.schema import build_file_descriptor_set

    video_schema_data = build_file_descriptor_set(CompressedVideo).SerializeToString()
    state_schema_name = "lerobot_msgs/msg/State"
    action_schema_name = "lerobot_msgs/msg/Action"
    state_schema_text = f"float32[{numeric_schemas['observation.state'].dim}] position"
    action_schema_text = f"float32[{numeric_schemas['action'].dim}] action"
    state_schema_data = state_schema_text.encode("utf-8")
    action_schema_data = action_schema_text.encode("utf-8")

    source_uri = f"hf://datasets/{dataset_source.repo_id}@{dataset_source.revision}"
    with tempfile.TemporaryDirectory(prefix="lerobot-source-episode-") as temporary_directory:
        source_episode_path = Path(temporary_directory) / output_path.name
        with source_episode_path.open("wb") as source_stream:
            mcap_writer = McapWriter(source_stream)
            mcap_writer.start(profile="", library="hflow LeRobot source adapter")

            video_schema_id = mcap_writer.register_schema(
                "foxglove.CompressedVideo", "protobuf", video_schema_data
            )
            state_schema_id = mcap_writer.register_schema(
                state_schema_name, "ros2msg", state_schema_data
            )
            action_schema_id = mcap_writer.register_schema(
                action_schema_name, "ros2msg", action_schema_data
            )

            video_channel_ids: dict[str, int] = {}
            for camera_key in camera_keys:
                video_channel_ids[camera_key] = mcap_writer.register_channel(
                    topic=f"/{camera_key}",
                    message_encoding="protobuf",
                    schema_id=video_schema_id,
                )
            state_channel_id = mcap_writer.register_channel(
                topic="/observation.state", message_encoding="cdr", schema_id=state_schema_id
            )
            action_channel_id = mcap_writer.register_channel(
                topic="/action", message_encoding="cdr", schema_id=action_schema_id
            )

            for frame_index in range(frame_count):
                log_time_ns = EPISODE_START_TIME_NS + round(
                    frame_index * NANOSECONDS_PER_SECOND / frames_per_second
                )
                for camera_key in camera_keys:
                    access_units, _presentation_timestamps = video_data_by_camera[camera_key]
                    video_message = CompressedVideo()
                    video_message.timestamp.seconds = log_time_ns // NANOSECONDS_PER_SECOND
                    video_message.timestamp.nanos = log_time_ns % NANOSECONDS_PER_SECOND
                    video_message.frame_id = camera_key
                    video_message.data = access_units[frame_index]
                    video_message.format = "h264"
                    mcap_writer.add_message(
                        channel_id=video_channel_ids[camera_key],
                        log_time=log_time_ns,
                        data=video_message.SerializeToString(),
                        publish_time=log_time_ns,
                        sequence=frame_index,
                    )
                mcap_writer.add_message(
                    channel_id=state_channel_id,
                    log_time=log_time_ns,
                    data=_encode_cdr_float32_array(state_rows[frame_index]),
                    publish_time=log_time_ns,
                    sequence=frame_index,
                )
                mcap_writer.add_message(
                    channel_id=action_channel_id,
                    log_time=log_time_ns,
                    data=_encode_cdr_float32_array(action_rows[frame_index]),
                    publish_time=log_time_ns,
                    sequence=frame_index,
                )

            mcap_writer.add_metadata(
                name="episode/v1",
                data={
                    "task": str(episode_row["task"] or ""),
                    "operator": "lerobot_converter",
                    "success": "true",
                    "embodiment": str(source_archive["info"].get("robot_type") or "unknown"),
                    "source_dataset": dataset_source.repo_id,
                    "source_revision": dataset_source.revision,
                    "source_episode_index": str(episode_index),
                    "converter_version": CONVERTER_VERSION,
                },
            )
            mcap_writer.add_metadata(
                name="source-provenance/v1",
                data={
                    "converter_version": CONVERTER_VERSION,
                    "ffmpeg_version": ffmpeg_version(),
                    "source_uri": source_uri,
                },
            )
            mcap_writer.finish()

        write_canonical_episode(
            source_episode_path,
            output_path,
            TransformConfig(gop_seconds=1.0),
            source_uri=source_uri,
        )

    logger.info(
        "wrote canonical LeRobot episode %s (%.2f MB)",
        output_path,
        output_path.stat().st_size / 1_000_000,
    )
    return output_path


__all__ = ["import_lerobot_dataset"]
