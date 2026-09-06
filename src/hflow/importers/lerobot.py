"""Import LeRobot Dataset v3 repositories as canonical MCAP episodes.

The converter reads repository metadata (feature schema, fps, episode
boundaries, video paths) instead of encoding dataset-specific assumptions.
Every selected RGB camera is converted into its own foxglove.CompressedVideo
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
import re
import struct
import subprocess
import tempfile
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import NotRequired, TypedDict
from urllib.parse import urlsplit

from mcap.reader import make_reader
from mcap.writer import Writer as McapWriter

from hflow.catalog import content_episode_id
from hflow.ffmpeg import ffmpeg_path, ffmpeg_version, ffprobe_path
from hflow.format import METADATA_RECORD_EPISODE, METADATA_RECORD_PROVENANCE
from hflow.reader import open_reader
from hflow.storage import LocalStorageRoot, StorageRoot, parse_storage_root
from hflow.transform import TransformConfig, write_canonical_episode

logger = logging.getLogger(__name__)

DEFAULT_REPO = "lerobot/pusht"
DEFAULT_REVISION = "main"
DEFAULT_OUTPUT_DIR = Path("./data/lerobot_pusht")
DEFAULT_CAMERA_KEY = "observation.image"

# "v5": episode metadata is read from every meta/episodes shard, not only the
# first, and shards are cached under their own chunk directory (#293).
# Multi-shard corpora previously published episodes with the wrong video
# window or the wrong source episode, so their outputs must not share an
# identity with the corrected ones.
# "v6": episode/v1 records camera_keys and gop_seconds so a resumable import
# can prove a landing file belongs to this exact selection (#303). Those
# fields change the canonical bytes that content_episode_id hashes, so v5
# and v6 outputs must not share a converter identity.
CONVERTER_VERSION = "lerobot-converter-v7"
# Canonical transform knobs that affect published bytes for this importer.
IMPORT_GOP_SECONDS = 1.0
# The v3 per-episode aggregate of the collector's frame-level next.success
# label. Optional: a corpus that declares no outcome feature has no such column.
_OUTCOME_AGGREGATE_COLUMN = "stats/next.success/max"
_SUCCESS_DERIVATION = f"max({_OUTCOME_AGGREGATE_COLUMN.removesuffix('/max')})"
PRESENTATION_TIMESTAMP_EPSILON_S = 0.050
EPISODE_METADATA_TREE_PREFIX = PurePosixPath("meta/episodes")

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
    # MAX of the episode's collector-labeled next.success frames, present only
    # when the source declares that outcome feature at all.
    success_outcome: NotRequired[bool]


class _VideoWindow(TypedDict):
    chunk_index: str
    file_index: str
    from_timestamp: float
    to_timestamp: float


class _PublishedEpisode(TypedDict):
    """One delivered episode as the manifest receipt describes it.

    ``uri`` is the published object URI (a bucket root has no local paths),
    ``content_id`` is ``content_episode_id`` over the canonical file taken
    while it is still on local disk, and ``size_bytes`` is that same file's
    size. Together they carry everything a delivery-verification reader of
    the manifest needs without a second schema bump.
    """

    uri: str
    content_id: str
    size_bytes: int


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


def _landing_relative_key(episode_index: int) -> str:
    return f"landing/lerobot_episode_{episode_index + 1:04d}.mcap"


def _encode_camera_keys(camera_keys: Sequence[str]) -> str:
    return json.dumps(list(camera_keys), separators=(",", ":"))


def _decode_camera_keys(encoded: str) -> tuple[str, ...] | None:
    try:
        decoded = json.loads(encoded)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, list) or not all(isinstance(key, str) for key in decoded):
        return None
    return tuple(decoded)


def _episode_identity_matches(
    local_episode_path: Path,
    *,
    dataset_source: DatasetSource,
    episode_index: int,
    camera_keys: tuple[str, ...],
) -> bool:
    """True when a published landing file belongs to this exact import.

    Identity over the metadata records, then a CRC-validated full message
    pass: a reused episode must be whole, not merely labeled. Metadata-only
    reads never touch chunk payloads, so without this pass a payload-damaged
    file would match identity and be stamped with a fresh receipt.
    """
    from mcap.exceptions import McapError

    reader = None
    try:
        reader = open_reader(local_episode_path)
        metadata_records = reader.metadata()
    except (OSError, McapError, ValueError):
        # Damaged, truncated, or non-MCAP landing files are not completed work.
        # Broader exceptions (bugs in the reader) must still surface.
        return False
    finally:
        if reader is not None:
            reader.close()

    episode_metadata = metadata_records.get(METADATA_RECORD_EPISODE, {})
    # write_canonical_episode preserves source episode/v1 and stamps
    # provenance/v1 with the transform's gop_seconds; source-provenance/v1
    # is copied through from the importer.
    provenance_metadata = metadata_records.get(METADATA_RECORD_PROVENANCE, {})
    source_provenance = metadata_records.get("source-provenance/v1", {})
    recorded_camera_keys = _decode_camera_keys(episode_metadata.get("camera_keys", ""))
    if recorded_camera_keys is None:
        return False
    expected_gop = f"{IMPORT_GOP_SECONDS:g}"
    if not (
        episode_metadata.get("source_dataset") == dataset_source.repo_id
        and episode_metadata.get("source_revision") == dataset_source.revision
        and episode_metadata.get("source_episode_index") == str(episode_index)
        and episode_metadata.get("converter_version") == CONVERTER_VERSION
        and source_provenance.get("converter_version") == CONVERTER_VERSION
        and recorded_camera_keys == camera_keys
        and episode_metadata.get("gop_seconds") == expected_gop
        and provenance_metadata.get("gop_seconds") == expected_gop
    ):
        return False
    # Metadata matching is not integrity: the metadata records live outside
    # the chunks, so payload damage never reaches them. Reuse must hold the
    # file to the same standard hflow doctor applies to any canonical file.
    try:
        with local_episode_path.open("rb") as stream:
            validated_reader = make_reader(stream, validate_crcs=True)
            for _ in validated_reader.iter_messages(log_time_order=False):
                pass
    except (OSError, McapError, ValueError) as error:
        # Reached only when identity already matched, so this is our own prior
        # output found damaged, not a stranger's file. Re-conversion repairs it
        # silently otherwise, which would hide bit rot in a landing tree for as
        # long as the imports keep succeeding.
        logger.warning(
            "re-converting landing episode %s: identity matches but the payload "
            "failed CRC validation (%s)",
            local_episode_path,
            error,
        )
        return False
    return True


def _try_reuse_completed_episode(
    storage: StorageRoot,
    *,
    dataset_source: DatasetSource,
    episode_index: int,
    camera_keys: tuple[str, ...],
) -> _PublishedEpisode | None:
    """Reuse a matching landing episode, or None when it must be converted."""
    relative_key = _landing_relative_key(episode_index)
    if not storage.exists(relative_key):
        return None
    try:
        # Avoid fetching an empty remote object only to reject it as invalid MCAP.
        if storage.file_size(relative_key) < 1:
            return None
        local_episode_path = storage.fetch(relative_key)
    except (OSError, FileNotFoundError, ValueError):
        return None
    if not _episode_identity_matches(
        local_episode_path,
        dataset_source=dataset_source,
        episode_index=episode_index,
        camera_keys=camera_keys,
    ):
        return None
    return {
        "uri": storage.uri_for(relative_key),
        "content_id": content_episode_id(local_episode_path),
        "size_bytes": local_episode_path.stat().st_size,
    }


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
        try:
            repository_information = json.loads(response.read().decode())
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Hugging Face repository response for {repo_id}@{revision} is not valid "
                f"JSON: {error}"
            ) from error
    if not isinstance(repository_information, dict):
        raise ValueError("Hugging Face repository response is not a JSON object")
    resolved_revision = repository_information.get("sha")
    if not isinstance(resolved_revision, str) or not resolved_revision.strip():
        raise ValueError(
            f"Hugging Face did not resolve {repo_id}@{revision} to an immutable commit"
        )
    if not re.fullmatch(r"[0-9a-f]{7,64}", resolved_revision):
        raise ValueError(f"Hugging Face returned a malformed commit sha for {repo_id}@{revision}")
    card_data = repository_information.get("cardData")
    license_name = card_data.get("license") if isinstance(card_data, dict) else None
    return {
        "sha": resolved_revision,
        "license": str(license_name or "unknown"),
    }


def _hf_tree(repo_id: str, revision: str, path: str) -> list[dict]:
    """List files under a HF dataset tree path (recursive)."""
    url = f"https://huggingface.co/api/datasets/{repo_id}/tree/{revision}/{path}?recursive=true"
    initial_url_parts = urlsplit(url)
    initial_origin = (initial_url_parts.scheme, initial_url_parts.netloc)
    all_tree_entries: list[dict] = []
    seen_paths: set[str] = set()
    visited_urls: set[str] = set()
    next_url: str | None = url
    while next_url is not None:
        if next_url in visited_urls:
            raise ValueError(
                f"Hugging Face tree response for {repo_id}@{revision} path {path!r} "
                f"repeated an already fetched pagination URL: {next_url!r}"
            )
        try:
            next_url_parts = urlsplit(next_url)
        except ValueError as error:
            raise ValueError(
                f"Hugging Face tree response for {repo_id}@{revision} path {path!r} "
                f"contains an invalid pagination URL: {next_url!r}"
            ) from error
        next_origin = (next_url_parts.scheme, next_url_parts.netloc)
        if next_origin != initial_origin:
            raise ValueError(
                f"Hugging Face tree response for {repo_id}@{revision} path {path!r} "
                f"contains a pagination URL with a different scheme and host: {next_url!r}"
            )
        visited_urls.add(next_url)
        with urllib.request.urlopen(_hugging_face_request(next_url), timeout=60) as response:
            try:
                tree_entries = json.loads(response.read().decode())
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Hugging Face tree response for {repo_id}@{revision} path {path!r} is not "
                    f"valid JSON: {error}"
                ) from error
            link_header = response.headers.get("Link")
        if not isinstance(tree_entries, list) or not all(
            isinstance(tree_entry, dict) for tree_entry in tree_entries
        ):
            raise ValueError(f"Hugging Face tree response for {path!r} is not a list of objects")
        for tree_entry in tree_entries:
            tree_entry_path = tree_entry.get("path")
            if isinstance(tree_entry_path, str):
                if tree_entry_path in seen_paths:
                    continue
                seen_paths.add(tree_entry_path)
            all_tree_entries.append(tree_entry)

        next_url = None
        for link_entry in (link_header or "").split(","):
            link_match = re.fullmatch(r"\s*<([^>]+)>\s*;\s*(.*)", link_entry)
            if link_match is None:
                continue
            relation_match = re.search(
                r"(?:^|;)\s*rel\s*=\s*(?:\"([^\"]+)\"|([^;\s]+))",
                link_match.group(2),
                flags=re.IGNORECASE,
            )
            if relation_match is None:
                continue
            relations = (relation_match.group(1) or relation_match.group(2)).split()
            if any(relation.lower() == "next" for relation in relations):
                next_url = link_match.group(1)
                break
    return all_tree_entries


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
        try:
            dataset_information = json.loads(response.read().decode())
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"LeRobot meta/info.json for {repo_id}@{revision} is not valid JSON: {error}"
            ) from error
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


def _episode_metadata_cache_path(
    episodes_metadata_directory: Path, tree_entry_path: str, *, repo_id: str
) -> Path:
    """Where one ``meta/episodes`` tree entry lands in the local cache.

    The path below ``meta/episodes`` is kept rather than flattened to its
    basename. Dataset v3 shards episode metadata as
    ``chunk-XXX/file-YYY.parquet`` and reuses file names across chunk
    directories, so two shards flattened to one cache file would make the
    second look already downloaded and its episodes vanish (#293). The tree
    listing is remote input, so an entry that would land outside the
    metadata directory is refused rather than joined.
    """
    tree_path = PurePosixPath(tree_entry_path)
    try:
        relative_path = tree_path.relative_to(EPISODE_METADATA_TREE_PREFIX)
    except ValueError:
        relative_path = None
    if relative_path is None or not relative_path.parts or ".." in relative_path.parts:
        raise ValueError(
            f"Hugging Face tree response for {repo_id} lists {tree_entry_path!r}, which is "
            f"not a file below {EPISODE_METADATA_TREE_PREFIX}/"
        )
    return episodes_metadata_directory / relative_path


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
            destination_path = _episode_metadata_cache_path(
                episodes_metadata_directory, entry["path"], repo_id=dataset_source.repo_id
            )
            if not destination_path.exists():
                _download_file(f"{dataset_base_url}/{entry['path']}", destination_path)
            episode_metadata_files.append(destination_path)

    if not episode_metadata_files:
        raise RuntimeError("no meta/episodes parquet files found")

    # Index of per-episode data windows across chunks. Every shard is one
    # relation: v3 splits episode metadata by size, so an episode and its
    # video window can sit in any file, not only the first (#293).
    episode_metadata_relation = (
        "read_parquet(["
        + ", ".join(
            "'" + str(episode_metadata_file).replace("'", "''") + "'"
            for episode_metadata_file in episode_metadata_files
        )
        + "], union_by_name=true)"
    )
    connection = duckdb.connect()
    try:
        # Column discovery first: the outcome aggregate is optional in v3
        # (not every corpus declares a collector-labeled next.success), and
        # naming a missing column would fail the read. A corpus without one
        # is normal, not malformed.
        episodes_columns = [
            column_description[0]
            for column_description in connection.execute(
                f"SELECT * FROM {episode_metadata_relation} LIMIT 1"
            ).description
        ]
        # Built outside the f-string below: a quoted identifier cannot be
        # nested in a same-quoted f-string on Python 3.11, which this repo
        # still supports.
        has_outcome_aggregate = _OUTCOME_AGGREGATE_COLUMN in episodes_columns
        outcome_aggregate_selector = (
            f', "{_OUTCOME_AGGREGATE_COLUMN}"' if has_outcome_aggregate else ""
        )

        episode_rows: list[_EpisodeRow] = []
        parquet_episode_rows = connection.execute(
            f"""
            SELECT "episode_index", "tasks", "length",
                   "data/chunk_index", "data/file_index",
                   "dataset_from_index", "dataset_to_index"
                   {outcome_aggregate_selector}
            FROM {episode_metadata_relation}
            ORDER BY "episode_index"
            """
        ).fetchall()
        for parquet_episode_row in parquet_episode_rows:
            tasks = parquet_episode_row[1]
            task = (str(tasks[0]) if tasks else "") if isinstance(tasks, list) else str(tasks or "")
            episode_row_value: _EpisodeRow = {
                "episode_index": int(parquet_episode_row[0]),
                "task": task,
                "length": int(parquet_episode_row[2]),
                "data_chunk": str(parquet_episode_row[3]).split("/")[-1],
                "data_file": str(parquet_episode_row[4]).split("/")[-1],
                "data_from": int(parquet_episode_row[5]),
                "data_to": int(parquet_episode_row[6]),
            }
            if has_outcome_aggregate:
                # Episode outcome = MAX over the episode's collector-labeled
                # next.success frames: any success frame makes the episode a
                # success. An empty aggregate carries no label either way.
                outcome_frames = parquet_episode_row[7]
                if outcome_frames is not None and len(outcome_frames) > 0:
                    episode_row_value["success_outcome"] = any(
                        bool(value) for value in outcome_frames
                    )
            episode_rows.append(episode_row_value)

        # Video window columns: videos/<camera>/{chunk_index,file_index,from_timestamp,to_timestamp}
        flattened_column_names = [
            column_description[0]
            for column_description in connection.execute(
                f"SELECT * FROM {episode_metadata_relation} LIMIT 1"
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
        video_window_rows = connection.execute(
            f"SELECT {video_window_select_sql} FROM {episode_metadata_relation}"
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


def _is_depth_map_feature(feature_specification: object) -> bool:
    """Whether LeRobot marks this feature as depth, by its own definition.

    Deliberately truthy rather than ``is True``, matching LeRobot's own
    ``is_depth_map()`` in ``lerobot/configs/video.py``, which returns
    ``bool(info.get("is_depth_map") or ...)``. Anything LeRobot calls depth
    must be refused here; a stricter test would let a corpus carrying
    ``"is_depth_map": "true"`` or ``1`` through the RGB path, which is the
    outcome the refusal exists to prevent.

    Canonically ``feature["info"]["is_depth_map"]``, with the legacy
    ``video.is_depth_map`` spelling in either ``info`` or a separate
    ``video_info`` dict.
    """
    if not isinstance(feature_specification, dict):
        return False
    feature_information = feature_specification.get("info")
    legacy_video_information = feature_specification.get("video_info")
    return bool(
        (
            isinstance(feature_information, dict)
            and (
                feature_information.get("is_depth_map")
                or feature_information.get("video.is_depth_map")
            )
        )
        or (
            isinstance(legacy_video_information, dict)
            and legacy_video_information.get("video.is_depth_map")
        )
    )


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
    output_dir: Path | str | StorageRoot = DEFAULT_OUTPUT_DIR,
    episode_index: int | None = None,
    camera_keys: str | Sequence[str] = (DEFAULT_CAMERA_KEY,),
) -> list[str]:
    """Import selected LeRobot Dataset v3 episodes as canonical MCAP.

    ``dataset_repo`` is a Hugging Face dataset repository. ``revision`` may
    name a branch, tag, or commit; the importer resolves it to an immutable
    commit before downloading data and records that commit in episode
    metadata and the prepared manifest. ``camera_keys`` accepts a sequence of
    camera features; a comma-separated string is accepted for compatibility.
    When ``episode_index`` is omitted, every episode is imported.

    ``output_dir`` is any HFlow data root: a local directory or an object-store
    prefix (``s3://``, ``gs://``, ``az://``). Hugging Face downloads and MCAP
    construction use local staging under the root's workspace. Durable outputs
    are published as ``landing/*.mcap`` plus ``prepared-manifest.json`` after
    every selected episode succeeds. Matching episodes already published under
    ``landing/`` are reused when their recorded identity matches this import
    (resolved commit, source episode, camera keys, converter version, and the
    importer's canonical GOP setting), so a mid-batch failure can resume
    without rewriting completed work. ``episodes_converted`` in the manifest
    counts episodes converted on this run, not ones reused. The source cache
    (``_lerobot_cache``) stays in the workspace -- the local directory itself,
    or the bucket mirror under ``HFLOW_MIRROR_DIR`` -- and is never uploaded
    into a bucket root.

    Dataset v3 RGB video features and one-dimensional, fixed-width float32
    state and action vectors are supported. Depth-marked videos and other
    unsupported feature layouts fail before any episode is published. The
    returned values are the published episode
    URIs (absolute path strings for local roots; ``s3://`` / ``gs://`` /
    ``az://`` object URIs for buckets).
    """
    storage = parse_storage_root(output_dir)
    if (
        isinstance(storage, LocalStorageRoot)
        and storage.path.exists()
        and not storage.path.is_dir()
    ):
        raise NotADirectoryError(f"output_dir is not a directory: {storage.path}")

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
    cache_directory = storage.workspace / "_lerobot_cache" / repository_information["sha"]
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

    dataset_features = source_archive["info"].get("features")
    if isinstance(dataset_features, dict):
        for camera_key in resolved_camera_keys:
            if _is_depth_map_feature(dataset_features.get(camera_key)):
                raise ValueError(
                    f"LeRobot feature '{camera_key}' is a depth-map video; HFlow's RGB H.264 "
                    "conversion cannot preserve depth values"
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
    published_episodes: list[_PublishedEpisode] = []
    episodes_converted = 0

    dataset_source = source_archive["dataset"]
    for selected_episode_index in selected_episode_indexes:
        reused_episode = _try_reuse_completed_episode(
            storage,
            dataset_source=dataset_source,
            episode_index=selected_episode_index,
            camera_keys=resolved_camera_keys,
        )
        if reused_episode is not None:
            logger.info(
                "reusing verified completed LeRobot episode %s (content_id %s)",
                reused_episode["uri"],
                reused_episode["content_id"],
            )
            published_episodes.append(reused_episode)
            continue
        published_episodes.append(
            _convert_single_episode(
                source_archive=source_archive,
                dataset_source=dataset_source,
                storage=storage,
                episode_index=selected_episode_index,
                camera_keys=resolved_camera_keys,
                numeric_schemas=numeric_schemas,
                frames_per_second=int(source_archive["fps"]),
            )
        )
        episodes_converted += 1

    manifest_contents = json.dumps(
        {
            "schema_version": 3,
            "dataset": {
                "repo_id": dataset_source.repo_id,
                "revision": dataset_source.revision,
                "license": dataset_source.license,
            },
            "camera_keys": list(resolved_camera_keys),
            "episodes_converted": episodes_converted,
            "episodes": list(published_episodes),
            "converter_version": CONVERTER_VERSION,
        },
        indent=2,
    )
    # Manifest-last: only publish after every selected episode object succeeded.
    with tempfile.TemporaryDirectory(prefix="lerobot-import-manifest-") as temporary_directory:
        temporary_manifest_path = Path(temporary_directory) / "prepared-manifest.json"
        temporary_manifest_path.write_text(manifest_contents + "\n", encoding="utf-8")
        published_manifest_uri = storage.publish(temporary_manifest_path, "prepared-manifest.json")
    logger.info("wrote LeRobot import manifest %s", published_manifest_uri)
    return [episode["uri"] for episode in published_episodes]


def _convert_single_episode(
    source_archive: _SourceArchive,
    dataset_source: DatasetSource,
    storage: StorageRoot,
    episode_index: int,
    camera_keys: tuple[str, ...],
    numeric_schemas: dict[str, _NumericSchema],
    frames_per_second: int,
) -> _PublishedEpisode:
    """Convert a single episode to canonical MCAP and publish it.

    Returns the published episode's manifest receipt: the published object
    URI, the content id of the canonical bytes, and the byte size.
    """
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
            / (
                f"{camera_key.replace('/', '_').replace('.', '_')}"
                f"-chunk{video_chunk_index}-file{video_file_index}.mp4"
            )
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
                    sliced_video_path, IMPORT_GOP_SECONDS, float(frames_per_second)
                )
                presentation_timestamps = _get_video_pts_times(sliced_video_path)
            finally:
                temporary_video_path.unlink(missing_ok=True)
        else:
            access_units = _transcode_mp4_to_h264(
                local_video_path, IMPORT_GOP_SECONDS, float(frames_per_second)
            )
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

    # Write MCAP into local staging, then publish the complete file.
    landing_relative_key = _landing_relative_key(episode_index)
    output_file_name = Path(landing_relative_key).name

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
        temporary_root = Path(temporary_directory)
        source_episode_path = temporary_root / output_file_name
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

            episode_record: dict[str, str] = {
                "task": str(episode_row["task"] or ""),
                "operator": "lerobot_converter",
                "embodiment": str(source_archive["info"].get("robot_type") or "unknown"),
                "source_dataset": dataset_source.repo_id,
                "source_revision": dataset_source.revision,
                "source_episode_index": str(episode_index),
                "converter_version": CONVERTER_VERSION,
                "camera_keys": _encode_camera_keys(camera_keys),
                "gop_seconds": f"{IMPORT_GOP_SECONDS:g}",
            }
            # success is the collector's label, never ours: when the source
            # declares the outcome feature, report MAX over the episode's
            # frames and name the derivation so the methodology travels with
            # the data; when it does not, the key is omitted rather than
            # invented (FORMAT.md: every episode/v1 key is optional and the
            # record is copied/merged from the source recording).
            success_outcome = episode_row.get("success_outcome")
            if success_outcome is not None:
                episode_record["success"] = "true" if success_outcome else "false"
                episode_record["success_derivation"] = _SUCCESS_DERIVATION
            mcap_writer.add_metadata(
                name="episode/v1",
                data=episode_record,
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

        canonical_episode_path = temporary_root / f"canonical-{output_file_name}"
        write_canonical_episode(
            source_episode_path,
            canonical_episode_path,
            TransformConfig(gop_seconds=IMPORT_GOP_SECONDS),
            source_uri=source_uri,
        )
        # Hash and size the canonical file while it is still on local disk,
        # before storage.publish: for a bucket root, reading the content id
        # back from the published object means downloading our own upload.
        episode_content_id = content_episode_id(canonical_episode_path)
        episode_size_bytes = canonical_episode_path.stat().st_size
        published_uri = storage.publish(canonical_episode_path, landing_relative_key)

    logger.info(
        "wrote canonical LeRobot episode %s (%.2f MB)",
        published_uri,
        episode_size_bytes / 1_000_000,
    )
    return {
        "uri": published_uri,
        "content_id": episode_content_id,
        "size_bytes": episode_size_bytes,
    }


__all__ = ["import_lerobot_dataset"]
