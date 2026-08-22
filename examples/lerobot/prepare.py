"""Download and prepare the LeRobot pusht dataset as canonical MCAP episodes.

The LeRobot pusht dataset (v3.0) stores data as Parquet + MP4 (av1 codec).
This converter transcodes the av1 video to canonical H.264, writes proper
MCAP channels per FORMAT.md, and stamps metadata per the mapping plan.

Usage:
    uv run python examples/lerobot/prepare.py \
        --repo lerobot/pusht \
        --revision main \
        --output-dir ./data/lerobot_pusht \
        --camera-key observation.image \
        --episode-index 0

Output:
    One canonical MCAP file per episode under <output-dir>/landing/
    A prepared-manifest.json summarizing the corpus.
"""

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from hflow.mcap_writer import CanonicalMcapWriter

DEFAULT_REPO = "lerobot/pusht"
DEFAULT_REVISION = "main"
DEFAULT_OUTPUT_DIR = Path("./data/lerobot_pusht")
DEFAULT_CAMERA_KEY = "observation.image"

CONVERTER_VERSION = "lerobot-converter-v1"

# LeRobot pusht constants
PUSHT_GOP_SECONDS = 1.0

# Timestamp handling
NANOSECONDS_PER_SECOND = 1_000_000_000
EPISODE_START_TIME_NS = 1_755_000_000_000_000_000


@dataclass(frozen=True)
class DatasetSource:
    repo_id: str
    revision: str
    license: str


@dataclass(frozen=True)
class SourceArchive:
    path: str
    sha256: str


@dataclass(frozen=True)
class SourceVideo:
    member: str
    sha256: str
    duration_s: float
    task: str


@dataclass(frozen=True)
class EpisodePlan:
    total_episodes: int
    duration_s: float
    first_source_start_s: float
    source_stride_s: float


@dataclass(frozen=True)
class PlannedEpisode:
    episode_id: str
    source_member: str
    source_start_s: float
    duration_s: float
    task: str


@dataclass(frozen=True)
class CorpusManifest:
    schema_version: int
    dataset: DatasetSource
    archive: SourceArchive
    sources: list[SourceVideo]
    episode_plan: EpisodePlan
    episodes: list[PlannedEpisode]


def _require_object(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a JSON object with string keys")
    return value


def _require_array(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a JSON array")
    return value


def _require_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _require_number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{context} must be a number")
    return float(value)


def _require_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    return value


def _require_ffmpeg() -> None:
    """Ensure ffmpeg is available on PATH."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found in PATH")


def _get_ffmpeg_version() -> str:
    """Get the first line of ffmpeg -version output."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.splitlines()[0].strip()
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        return "unknown"


def _expand_episode_plan(
    sources: list[SourceVideo],
    episode_plan: EpisodePlan,
) -> list[PlannedEpisode]:
    episodes: list[PlannedEpisode] = []
    for episode_number in range(1, episode_plan.total_episodes + 1):
        zero_based_episode_index = episode_number - 1
        source_video = sources[zero_based_episode_index % len(sources)]
        source_window_index = zero_based_episode_index // len(sources)
        source_start_s = (
            episode_plan.first_source_start_s + source_window_index * episode_plan.source_stride_s
        )
        if source_start_s + episode_plan.duration_s > source_video.duration_s:
            raise ValueError(
                f"episode {episode_number} ends after {source_video.member}: "
                f"{source_start_s + episode_plan.duration_s:g}s > "
                f"{source_video.duration_s:g}s"
            )

        episodes.append(
            PlannedEpisode(
                episode_id=f"pusht_episode_{episode_number:04d}",
                source_member=source_video.member,
                source_start_s=source_start_s,
                duration_s=episode_plan.duration_s,
                task=source_video.task,
            )
        )
    return episodes


def _load_manifest(manifest_path: Path) -> CorpusManifest:
    root = _require_object(json.loads(manifest_path.read_text()), "manifest")
    schema_version_value = root.get("schema_version")
    if schema_version_value != 1:
        raise ValueError(f"unsupported manifest schema_version {schema_version_value!r}")

    dataset_object = _require_object(root.get("dataset"), "dataset")
    archive_object = _require_object(root.get("archive"), "archive")
    source_objects = _require_array(root.get("sources"), "sources")
    episode_plan_object = _require_object(root.get("episode_plan"), "episode_plan")

    dataset = DatasetSource(
        repo_id=_require_string(dataset_object.get("repo_id"), "dataset.repo_id"),
        revision=_require_string(dataset_object.get("revision"), "dataset.revision"),
        license=_require_string(dataset_object.get("license"), "dataset.license"),
    )
    archive = SourceArchive(
        path=_require_string(archive_object.get("path"), "archive.path"),
        sha256=str(archive_object.get("sha256", "") or ""),
    )
    sources = [
        SourceVideo(
            member=_require_string(source_object.get("member"), f"sources[{source_index}].member"),
            sha256=str(source_object.get("sha256", "") or ""),
            duration_s=_require_number(
                source_object.get("duration_s"), f"sources[{source_index}].duration_s"
            ),
            task=_require_string(source_object.get("task"), f"sources[{source_index}].task"),
        )
        for source_index, source_value in enumerate(source_objects)
        for source_object in [_require_object(source_value, f"sources[{source_index}]")]
    ]

    episode_plan = EpisodePlan(
        total_episodes=_require_integer(
            episode_plan_object.get("total_episodes"), "episode_plan.total_episodes"
        ),
        duration_s=_require_number(
            episode_plan_object.get("duration_s"), "episode_plan.duration_s"
        ),
        first_source_start_s=_require_number(
            episode_plan_object.get("first_source_start_s"),
            "episode_plan.first_source_start_s",
        ),
        source_stride_s=_require_number(
            episode_plan_object.get("source_stride_s"), "episode_plan.source_stride_s"
        ),
    )

    source_members = {source.member for source in sources}
    if not sources:
        raise ValueError("sources must contain at least one video")
    if len(source_members) != len(sources):
        raise ValueError("sources contains duplicate member names")
    if episode_plan.total_episodes < 1:
        raise ValueError("episode_plan.total_episodes must be at least one")
    if episode_plan.duration_s <= 0:
        raise ValueError("episode_plan.duration_s must be positive")
    if episode_plan.first_source_start_s < 0 or episode_plan.source_stride_s < 0:
        raise ValueError("episode plan source timings cannot be negative")

    episodes = _expand_episode_plan(sources, episode_plan)

    return CorpusManifest(
        schema_version=1,
        dataset=dataset,
        archive=archive,
        sources=sources,
        episode_plan=episode_plan,
        episodes=episodes,
    )


def _sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as input_stream:
        while chunk := input_stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sha256(file_path: Path, expected_sha256: str) -> None:
    actual_sha256 = _sha256_file(file_path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {file_path}: expected {expected_sha256}, got {actual_sha256}"
        )


def _ensure_source_archive(manifest: CorpusManifest, data_root: Path) -> Path:
    """Download all source files from Hugging Face to local directory."""
    _require_ffmpeg()
    download_root = data_root / "huggingface"
    download_root.mkdir(parents=True, exist_ok=True)

    base_url = f"https://huggingface.co/datasets/{manifest.dataset.repo_id}/resolve/{manifest.dataset.revision}"

    for source_video in manifest.sources:
        file_path = download_root / source_video.member
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if file_path.is_file():
            if source_video.sha256:
                _verify_sha256(file_path, source_video.sha256)
            continue

        url = f"{base_url}/{source_video.member}"
        print(f"Downloading {source_video.member}...")

        import urllib.request

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response, file_path.open("wb") as out_file:
            shutil.copyfileobj(response, out_file)

        if source_video.sha256:
            _verify_sha256(file_path, source_video.sha256)

    return download_root


def _extract_source_videos(
    manifest: CorpusManifest,
    download_root: Path,
) -> dict[str, Path]:
    """Return paths to source files (already downloaded, no extraction needed).
    Points downstream code at the original download directory to avoid copying.
    """
    source_paths: dict[str, Path] = {}

    for source_video in manifest.sources:
        src_path = download_root / source_video.member

        if not src_path.is_file():
            raise RuntimeError(f"source file not found: {src_path}")

        if source_video.sha256:
            _verify_sha256(src_path, source_video.sha256)
        source_paths[source_video.member] = src_path

    return source_paths


def _split_h264_by_aud(h264_data: bytes) -> list[bytes]:
    """Split H.264 Annex B stream into access units by AUD NALs (type 9).
    Requires the stream to contain access-unit delimiter NALs (type 9).
    """
    ANNEX_B_START_CODE = b"\x00\x00\x01"
    NAL_TYPE_ACCESS_UNIT_DELIMITER = 9
    VCL_NAL_TYPES = frozenset({1, 2, 3, 4, 5})

    nal_offsets_and_types: list[tuple[int, int]] = []
    search_offset = 0
    while True:
        code_offset = h264_data.find(ANNEX_B_START_CODE, search_offset)
        if code_offset == -1:
            break
        type_byte_offset = code_offset + len(ANNEX_B_START_CODE)
        if type_byte_offset >= len(h264_data):
            break
        has_four_byte_start_code = code_offset > 0 and h264_data[code_offset - 1] == 0
        nal_start_offset = code_offset - 1 if has_four_byte_start_code else code_offset
        nal_type = h264_data[type_byte_offset] & 0x1F
        nal_offsets_and_types.append((nal_start_offset, nal_type))
        search_offset = type_byte_offset

    aud_nal_indices = [
        nal_index
        for nal_index, (_, nal_type) in enumerate(nal_offsets_and_types)
        if nal_type == NAL_TYPE_ACCESS_UNIT_DELIMITER
    ]
    if not aud_nal_indices:
        raise ValueError(
            "no access-unit delimiter (AUD, NAL type 9) found; splitting requires "
            "a stream encoded with aud=1"
        )
    first_aud_nal_index = aud_nal_indices[0]
    vcl_nal_precedes_first_aud = any(
        nal_type in VCL_NAL_TYPES for _, nal_type in nal_offsets_and_types[:first_aud_nal_index]
    )
    if vcl_nal_precedes_first_aud:
        raise ValueError(
            "a VCL NAL precedes the first access-unit delimiter; splitting requires "
            "a stream encoded with aud=1"
        )

    unit_start_offsets = [nal_offsets_and_types[i][0] for i in aud_nal_indices]
    unit_start_offsets[0] = 0
    unit_end_offsets = [*unit_start_offsets[1:], len(h264_data)]

    access_units: list[bytes] = []
    for unit_start, unit_end in zip(unit_start_offsets, unit_end_offsets, strict=True):
        access_units.append(h264_data[unit_start:unit_end])
    return access_units


def _encode_cdr_float32_array(arr: list[float]) -> bytes:
    """Encode a float32[2] array as ROS 2 CDR (XCDR1 little-endian).
    Includes 4-byte encapsulation header (00 01 00 00) and 4-byte alignment.
    Schema: float32[2] position
    """
    # CDR encapsulation header: little-endian, XCDR1
    encapsulation = b"\x00\x01\x00\x00"
    # Payload: float32[2] with 4-byte alignment from body start
    payload = struct.pack(f"<{len(arr)}f", *arr)
    return encapsulation + payload


def _transcode_mp4_to_h264(mp4_path: Path, gop_seconds: float, fps: float) -> list[bytes]:
    """Transcode MP4 to H.264 Annex B stream using ffmpeg directly.

    Uses exact x264 parameters from src/hflow/video.py:
    -c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p
    -x264-params keyint=<gop_frames>:min-keyint=<gop_frames>:scenecut=0:bframes=0:repeat-headers=1:aud=1
    """
    _require_ffmpeg()

    gop_frames = max(1, round(gop_seconds * fps))
    x264_params = (
        f"keyint={gop_frames}:min-keyint={gop_frames}:scenecut=0:bframes=0:repeat-headers=1:aud=1"
    )

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(mp4_path),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-x264-params",
        x264_params,
        "-f",
        "h264",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg transcode failed: {result.stderr.decode('utf-8', errors='replace')}"
        )
    return _split_h264_by_aud(result.stdout)


def _get_video_pts_times(mp4_path: Path) -> list[float]:
    """Get per-frame PTS times from MP4 using ffprobe.
    Returns list of PTS times in seconds for each frame.
    """
    _require_ffmpeg()
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "packet=pts_time",
        "-of",
        "csv=p=0",
        str(mp4_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    pts_times = []
    for line in result.stdout.strip().splitlines():
        if line.strip():
            with contextlib.suppress(ValueError):
                pts_times.append(float(line.strip()))
    return pts_times


def _slice_video(
    video_path: Path,
    from_ts: float,
    to_ts: float,
) -> Path:
    """Slice video to a temp file (MP4 muxer requires seekable output)."""
    _require_ffmpeg()
    fd, sliced_video_path_str = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    sliced_video_path = Path(sliced_video_path_str)
    try:
        slice_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            str(from_ts),
            "-to",
            str(to_ts),
            "-i",
            str(video_path),
            "-c",
            "copy",
            "-f",
            "mp4",
            str(sliced_video_path),
        ]
        slice_result = subprocess.run(slice_cmd, capture_output=True)
        if slice_result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg slice failed: {slice_result.stderr.decode('utf-8', errors='replace')}"
            )
        return sliced_video_path
    except Exception:
        sliced_video_path.unlink(missing_ok=True)
        raise


def lerobot_to_mcap(
    dataset_repo: str = "lerobot/pusht",
    revision: str = "main",
    output_dir: Path = Path("./data/lerobot_pusht"),
    episode_index: int | None = None,
    camera_key: str = "observation.image",
) -> list[Path]:
    """Convert LeRobot pusht dataset episodes to canonical MCAP.

    Args:
        dataset_repo: Hugging Face dataset repository (default: lerobot/pusht)
        revision: Dataset revision (default: main)
        output_dir: Output directory for prepared episodes
        episode_index: Episode index to convert, or None for all episodes
        camera_key: Camera key in dataset (default: observation.image)

    Returns:
        List of output MCAP file paths.
    """
    _require_ffmpeg()

    # Build manifest inline (no external manifest.json needed for pusht)
    dataset = DatasetSource(
        repo_id=dataset_repo,
        revision=revision,
        license="CC-BY-4.0",
    )
    archive = SourceArchive(
        path="pusht_dataset",
        sha256="",
    )
    sources = [
        SourceVideo(
            member="data/chunk-000/file-000.parquet",
            sha256="",
            duration_s=2060.0,
            task="push_t",
        ),
        SourceVideo(
            member="videos/observation.image/chunk-000/file-000.mp4",
            sha256="",
            duration_s=2060.0,
            task="push_t",
        ),
        SourceVideo(
            member="meta/episodes/chunk-000/file-000.parquet",
            sha256="",
            duration_s=2060.0,
            task="push_t",
        ),
    ]
    episode_plan = EpisodePlan(
        total_episodes=206,
        duration_s=10.0,
        first_source_start_s=0.0,
        source_stride_s=10.0,
    )

    manifest = CorpusManifest(
        schema_version=1,
        dataset=dataset,
        archive=archive,
        sources=sources,
        episode_plan=episode_plan,
        episodes=_expand_episode_plan(
            sources,
            EpisodePlan(
                total_episodes=206,
                duration_s=10.0,
                first_source_start_s=0.0,
                source_stride_s=10.0,
            ),
        ),
    )

    # Download/verify source files
    download_root = _ensure_source_archive(manifest, output_dir)
    source_paths = _extract_source_videos(manifest, download_root)

    # Determine which episodes to process
    if episode_index is not None:
        episode_indices = [episode_index]
    else:
        episode_indices = list(range(manifest.episode_plan.total_episodes))

    output_paths: list[Path] = []

    for ep_idx in episode_indices:
        output_path = _convert_single_episode(
            manifest=manifest,
            source_paths=dict(source_paths),
            output_dir=output_dir,
            episode_index=ep_idx,
            camera_key=camera_key,
        )
        output_paths.append(output_path)

    return output_paths


def _convert_single_episode(
    manifest: CorpusManifest,
    source_paths: dict[str, Path],
    output_dir: Path,
    episode_index: int,
    camera_key: str,
) -> Path:
    """Convert a single episode to canonical MCAP. Returns output path."""
    import duckdb

    episode = manifest.episodes[episode_index]

    # Select the specific parquet files by their manifest role
    # Find the data parquet (not meta/episodes)
    data_parquet = None
    episodes_parquet = None
    video_mp4 = None

    for source_video in manifest.sources:
        if source_video.member.endswith(".parquet") and not source_video.member.startswith("meta/"):
            data_parquet = source_video.member
        elif source_video.member.startswith("meta/episodes/") and source_video.member.endswith(
            ".parquet"
        ):
            episodes_parquet = source_video.member
        elif source_video.member.endswith(".mp4"):
            video_mp4 = source_video.member

    if not data_parquet or not video_mp4:
        raise RuntimeError("Required source files not found in manifest")

    data_parquet_path = source_paths.get(data_parquet)
    episodes_parquet_path = source_paths.get(episodes_parquet) if episodes_parquet else None
    video_path = source_paths.get(video_mp4)

    if not data_parquet_path or not data_parquet_path.is_file():
        raise RuntimeError(f"data parquet not found: {data_parquet_path}")
    if not video_path or not video_path.is_file():
        raise RuntimeError(f"video not found: {video_path}")

    # Probe video for actual frame rate
    _require_ffmpeg()
    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
    if probe_result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {probe_result.stderr}")
    fps_fraction = probe_result.stdout.strip()
    if "/" in fps_fraction:
        num, den = map(int, fps_fraction.split("/"))
        fps = num / den
    else:
        fps = float(fps_fraction)

    # Query the episode data from parquet (select specific file by manifest role)
    data_parquet_escaped = str(data_parquet_path).replace("'", "''")
    conn = duckdb.connect()
    try:
        conn.execute(f"CREATE VIEW data AS SELECT * FROM read_parquet('{data_parquet_escaped}')")

        # Get episode data
        episode_data = conn.execute(
            f"SELECT * FROM data WHERE episode_index = {episode_index} ORDER BY frame_index"
        ).fetchall()

        # Get column names
        columns = [desc[0] for desc in conn.description]

        if not episode_data:
            raise ValueError(f"no data found for episode_index {episode_index}")
    finally:
        conn.close()

    # Extract state, action, timestamp arrays
    state_idx = columns.index("observation.state")
    action_idx = columns.index("action")
    timestamp_idx = columns.index("timestamp")

    states = [row[state_idx] for row in episode_data]
    actions = [row[action_idx] for row in episode_data]
    timestamps = [row[timestamp_idx] for row in episode_data]

    # Get episode timestamps from episodes parquet for video slicing
    from_ts = 0.0
    to_ts = 0.0
    if episodes_parquet_path and episodes_parquet_path.is_file():
        episodes_parquet_escaped = str(episodes_parquet_path).replace("'", "''")
        conn = duckdb.connect()
        try:
            conn.execute(
                f"CREATE VIEW episodes AS SELECT * FROM read_parquet('{episodes_parquet_escaped}')"
            )
            ep_row = conn.execute(
                f'SELECT "videos/observation.image/from_timestamp", "videos/observation.image/to_timestamp" FROM episodes WHERE episode_index = {episode_index}'
            ).fetchone()
            if ep_row:
                from_ts, to_ts = ep_row
        finally:
            conn.close()

    # Slice video if we have timestamps
    if from_ts > 0.0 or to_ts > 0.0:
        sliced_video_path = _slice_video(video_path, from_ts, to_ts)
        try:
            access_units = _transcode_mp4_to_h264(sliced_video_path, PUSHT_GOP_SECONDS, fps)
            pts_times = _get_video_pts_times(sliced_video_path)
        finally:
            sliced_video_path.unlink(missing_ok=True)
    else:
        access_units = _transcode_mp4_to_h264(video_path, PUSHT_GOP_SECONDS, fps)
        pts_times = _get_video_pts_times(video_path)

    # Cross-check: |parquet_timestamp - video_pts| > 50ms
    frame_count = len(states)
    if len(access_units) != frame_count:
        raise ValueError(
            f"frame count mismatch: {frame_count} parquet frames vs {len(access_units)} video access units"
        )

    if len(pts_times) != frame_count:
        raise ValueError(
            f"PTS count mismatch: {frame_count} frames vs {len(pts_times)} video PTS entries"
        )

    for i in range(frame_count):
        parquet_ts = timestamps[i]
        video_pts = pts_times[i] if i < len(pts_times) else (i / fps)
        if abs(parquet_ts - video_pts) > 0.050:  # 50ms threshold
            raise ValueError(
                f"timestamp disagreement at frame {i}: parquet={parquet_ts:.6f}s, "
                f"video_pts={video_pts:.6f}s, diff={abs(parquet_ts - video_pts) * 1000:.1f}ms > 50ms"
            )

    # Write MCAP
    output_path = output_dir / "landing" / f"pusht_episode_{episode_index + 1:04d}.mcap"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build foxglove.CompressedVideo schema
    from foxglove_schemas_protobuf.CompressedVideo_pb2 import CompressedVideo
    from mcap_protobuf.schema import build_file_descriptor_set

    schema_data = build_file_descriptor_set(CompressedVideo).SerializeToString()

    # Custom CDR schema for state/action: float32[2] position (ROS 2 message definition)
    STATE_SCHEMA_NAME = "lerobot_msgs/msg/State"
    ACTION_SCHEMA_NAME = "lerobot_msgs/msg/Action"
    STATE_SCHEMA_TEXT = """float32[2] position"""
    ACTION_SCHEMA_TEXT = """float32[2] action"""
    state_schema_data = STATE_SCHEMA_TEXT.encode("utf-8")
    action_schema_data = ACTION_SCHEMA_TEXT.encode("utf-8")

    with CanonicalMcapWriter(output_path) as writer:
        # Register schemas
        video_schema_id = writer.register_schema(
            "foxglove.CompressedVideo", "protobuf", schema_data
        )
        state_schema_id = writer.register_schema(STATE_SCHEMA_NAME, "ros2msg", state_schema_data)
        action_schema_id = writer.register_schema(ACTION_SCHEMA_NAME, "ros2msg", action_schema_data)

        # Register channels
        video_channel_id = writer.register_channel(
            "/observation.image",
            "protobuf",
            video_schema_id,
            group="cameras",
        )
        state_channel_id = writer.register_channel(
            "/observation.state",
            "cdr",
            state_schema_id,
            group="state",
        )
        action_channel_id = writer.register_channel(
            "/action",
            "cdr",
            action_schema_id,
            group="state",
        )

        # Write messages
        for i in range(frame_count):
            log_time_ns = EPISODE_START_TIME_NS + round(i * NANOSECONDS_PER_SECOND / fps)

            # Video frame
            if i < len(access_units):
                video_msg = CompressedVideo()
                video_msg.timestamp.seconds = log_time_ns // 1_000_000_000
                video_msg.timestamp.nanos = log_time_ns % 1_000_000_000
                video_msg.frame_id = "observation.image"
                video_msg.data = access_units[i]
                video_msg.format = "h264"
                writer.write_message(
                    video_channel_id,
                    log_time_ns,
                    video_msg.SerializeToString(),
                )

            # State (float32[2] CDR)
            state_data = _encode_cdr_float32_array(states[i])
            writer.write_message(state_channel_id, log_time_ns, state_data)

            # Action (float32[2] CDR)
            action_data = _encode_cdr_float32_array(actions[i])
            writer.write_message(action_channel_id, log_time_ns, action_data)

        # Add episode metadata
        ffmpeg_version = _get_ffmpeg_version()
        writer.add_metadata(
            "episode/v1",
            {
                "task": episode.task,
                "operator": "lerobot_converter",
                "success": "true",
                "embodiment": "pusht",
                "source_dataset": manifest.dataset.repo_id,
                "source_revision": manifest.dataset.revision,
                "source_episode_index": str(episode_index),
                "converter_version": CONVERTER_VERSION,
            },
        )
        writer.add_metadata(
            "provenance/v1",
            {
                "schema_version": "1",
                "pipeline_version": CONVERTER_VERSION,
                "ffmpeg_version": ffmpeg_version,
                "gop_preset": "vla",
                "gop_seconds": str(PUSHT_GOP_SECONDS),
                "source_uri": f"hf://datasets/{manifest.dataset.repo_id}@{manifest.dataset.revision}",
            },
        )

    print(f"wrote {output_path} ({output_path.stat().st_size / 1_000_000:.1f} MB)")
    return output_path


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", default=DEFAULT_REPO, help=f"Hugging Face dataset repo (default: {DEFAULT_REPO})"
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help=f"Dataset revision (default: {DEFAULT_REVISION})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for prepared episodes (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--camera-key",
        default=DEFAULT_CAMERA_KEY,
        help=f"Camera key in dataset (default: {DEFAULT_CAMERA_KEY})",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=None,
        help="Episode index to convert, or None for all episodes (default: None)",
    )
    args = parser.parse_args()

    output_paths = lerobot_to_mcap(
        dataset_repo=args.repo,
        revision=args.revision,
        output_dir=args.output_dir,
        episode_index=args.episode_index,
        camera_key=args.camera_key,
    )

    print(f"Converted {len(output_paths)} episode(s)")


if __name__ == "__main__":
    main()
