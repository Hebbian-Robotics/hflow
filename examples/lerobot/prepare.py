#!/usr/bin/env python3
"""Download and prepare LeRobot Dataset v3 repositories as canonical MCAP episodes.

The converter reads repository metadata (feature schema, fps, episode
boundaries, video paths) instead of encoding dataset-specific assumptions.
Every selected camera is converted into its own foxglove.CompressedVideo
channel; numeric state and action schemas are derived from the declared
dtype and shape, failing loud before conversion when a feature is
unsupported.

Usage:
    uv run python examples/lerobot/prepare.py \
        --repo lerobot/pusht \
        --revision main \
        --output-dir ./data/lerobot_pusht \
        --camera-key observation.image \
        --episode-index 0

    uv run python examples/lerobot/prepare.py \
        --repo lerobot/svla_so101_pickplace \
        --revision f641879e22172be7e8161d5e6c1503c2d2feb657 \
        --output-dir ./data/lerobot_svla \
        --camera-key observation.images.up,observation.images.side \
        --episode-index 0

Output:
    One canonical MCAP file per episode under <output-dir>/landing/
    A prepared-manifest.json summarizing the corpus.
"""

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, TypedDict

from mcap.writer import Writer as McapWriter

import hflow

DEFAULT_REPO = "lerobot/pusht"
DEFAULT_REVISION = "main"
DEFAULT_OUTPUT_DIR = Path("./data/lerobot_pusht")
DEFAULT_CAMERA_KEY = "observation.image"

CONVERTER_VERSION = "lerobot-converter-v2"
PRESENTATION_TIMESTAMP_EPSILON_S = 0.050

# Timestamp handling
NANOSECONDS_PER_SECOND = 1_000_000_000
EPISODE_START_TIME_NS = 1_755_000_000_000_000_000


class _EpisodeRow(TypedDict):
    episode_index: int
    task: str
    length: int
    data_chunk: str
    data_file: str
    data_from: int
    data_to: int
    video_windows: NotRequired[dict[str, dict[str, float | str]]]


@dataclass(frozen=True)
class DatasetSource:
    repo_id: str
    revision: str
    license: str


@dataclass(frozen=True)
class CorpusManifest:
    schema_version: int
    dataset: DatasetSource
    sources: list
    episodes: list
    camera_keys: tuple[str, ...]


@dataclass(frozen=True)
class SourceVideo:
    member: str
    sha256: str
    kind: str = "data"


@dataclass(frozen=True)
class EpisodePlan:
    total_episodes: int
    duration_s: float
    first_source_start_s: float
    source_stride_s: float


@dataclass(frozen=True)
class PlannedEpisode:
    task: str
    src_start_s: float
    src_end_s: float
    num_frames: int
    duration_s: float


def _require_ffmpeg() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError(
            "ffmpeg and ffprobe are required. Install them (e.g. apt install ffmpeg) "
            "or place static binaries on PATH."
        )


def _get_ffmpeg_version() -> str:
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=15)
    return result.stdout.splitlines()[0] if result.returncode == 0 else "unknown"


def _sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _encode_cdr_float32_array(arr: list[float] | tuple[float, ...]) -> bytes:
    """Encode a float32[N] array as ROS 2 CDR (XCDR1 little-endian).

    CDR encapsulation header (00 01 00 00) followed by the packed floats;
    byte-compatible with hflow's mcap_ros2 decoder.
    """
    encapsulation = b"\x00\x01\x00\x00"
    payload = struct.pack(f"<{len(arr)}f", *(float(v) for v in arr))
    return encapsulation + payload


def _transcode_mp4_to_h264(mp4_path: Path, gop_seconds: float, fps: float) -> list[bytes]:
    """Transcode an mp4 to H.264 access units split on AUD markers."""
    keyint = max(1, round(gop_seconds * fps))
    cmd = [
        "ffmpeg",
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
        str(keyint),
        "-keyint_min",
        str(keyint),
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
    result = subprocess.run(cmd, capture_output=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg transcode failed: {result.stderr.decode(errors='ignore')}")
    return _split_h264_by_aud(result.stdout)


def _split_h264_by_aud(h264_data: bytes) -> list[bytes]:
    """Split H.264 stream on AUD (0x00000109) boundaries."""
    units: list[bytes] = []
    pattern = b"\x00\x00\x00\x01\x09"
    offset = 0
    while True:
        idx = h264_data.find(pattern, offset)
        if idx == -1:
            break
        if idx > offset:
            units.append(h264_data[offset:idx])
        offset = idx + 5
    if offset < len(h264_data):
        units.append(h264_data[offset:])
    return units


def _get_video_pts_times(mp4_path: Path) -> list[float]:
    result = subprocess.run(
        [
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
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    times: list[float] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line and line != "N/A":
            times.append(float(line))
    return times


def _slice_video(
    video_path: Path, from_s: float, to_s: float, out_path: Path | None = None
) -> Path:
    """Slice a video to a time window with stream copy (no re-encode)."""
    out = out_path or (video_path.parent / f"{video_path.stem}_slice{video_path.suffix}")
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{from_s:.6f}",
        "-to",
        f"{to_s:.6f}",
        "-i",
        str(video_path),
        "-c",
        "copy",
        "-an",
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg slice failed: {result.stderr.decode(errors='ignore')}")
    return out


def _hf_repo_info(repo_id: str, revision: str) -> dict:
    """Resolve the immutable commit sha and license for a HF dataset repo."""
    url = f"https://huggingface.co/api/datasets/{repo_id}/revision/{revision}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        info = json.loads(resp.read().decode())
    license_name = (info.get("cardData") or {}).get("license") or "unknown"
    return {"sha": info.get("sha", revision), "license": str(license_name)}


def _hf_tree(repo_id: str, revision: str, path: str) -> list[dict]:
    """List files under a HF dataset tree path (recursive)."""
    url = f"https://huggingface.co/api/datasets/{repo_id}/tree/{revision}/{path}?recursive=true"
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode())


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
    req = urllib.request.Request(load_url, headers={"User-Agent": "hflow-lerobot"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        info = json.loads(resp.read().decode())
    (cache_dir / "meta").mkdir(parents=True, exist_ok=True)
    (cache_dir / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    return info


def _download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "hflow-lerobot"})
    with urllib.request.urlopen(req, timeout=300) as resp, dest.open("wb") as out:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)


def _ensure_source_archive(dataset: DatasetSource, cache_dir: Path) -> dict:
    """Download the corpus parquets and video chunks needed for the given episodes."""
    import duckdb

    base = f"https://huggingface.co/datasets/{dataset.repo_id}/resolve/{dataset.revision}"
    meta_dir = cache_dir / "meta"
    info = _fetch_info_json(dataset.repo_id, dataset.revision, cache_dir)
    fps = info["fps"]
    data_path = info["data_path"]  # e.g. "data/{chunk_index:06d}/parquet/{file_index:06d}.parquet"
    video_tpl = info.get("video_path", "videos/{camera_key}/{chunk_index:06d}/{file_index:06d}.mp4")

    def _template(tpl: str, **kw: object) -> str:
        return tpl.format(
            chunk_index=kw.get("chunk_index", 0),
            file_index=kw.get("file_index", 0),
            camera_key=kw.get("camera_key", ""),
        )

    # Determine the episodes parquet location (v3 uses meta/episodes/*.parquet)
    episodes_dir = meta_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    ep_files: list[Path] = []
    entries = _hf_tree(dataset.repo_id, dataset.revision, "meta/episodes")
    for entry in entries:
        if entry.get("type") == "file" and entry["path"].endswith(".parquet"):
            dest = episodes_dir / Path(entry["path"]).name
            if not dest.exists():
                _download_file(f"{base}/{entry['path']}", dest)
            ep_files.append(dest)

    if not ep_files:
        raise RuntimeError("no meta/episodes parquet files found")

    # Index of per-episode data windows across chunks
    conn = duckdb.connect()
    try:
        rows: list[_EpisodeRow] = []
        for ep_file in ep_files:
            q = conn.execute(
                f"""
                SELECT "episode_index", "tasks", "length",
                       "data/chunk_index", "data/file_index",
                       "dataset_from_index", "dataset_to_index"
                FROM read_parquet('{str(ep_file).replace("'", "''")}')
                ORDER BY "episode_index"
                """
            ).fetchall()
            for row in q:
                tasks = row[1]
                if isinstance(tasks, list):
                    task = str(tasks[0]) if tasks else ""
                else:
                    task = str(tasks or "")
                rows.append(
                    {
                        "episode_index": int(row[0]),
                        "task": task,
                        "length": int(row[2]),
                        "data_chunk": str(row[3]).split("/")[-1],
                        "data_file": str(row[4]).split("/")[-1],
                        "data_from": int(row[5]),
                        "data_to": int(row[6]),
                    }
                )
        rows.sort(key=lambda episode: episode["episode_index"])

        # Video window columns: videos/<camera>/{chunk_index,file_index,from_timestamp,to_timestamp}
        flat_cols = [
            d[0]
            for d in conn.execute(
                "SELECT * FROM read_parquet('" + str(ep_files[0]).replace("'", "''") + "') LIMIT 1"
            ).description
        ]
        video_keys = sorted(
            {
                col.split("/")[1]
                for col in flat_cols
                if col.startswith("videos/") and col.endswith("/from_timestamp")
            }
        )

        # Gather video windows per episode+camera
        video_windows = {}
        selectors = []
        for cam in video_keys:
            selectors += [
                f'"videos/{cam}/chunk_index" as "vc_{cam}",',
                f'"videos/{cam}/file_index" as "vf_{cam}",',
                f'"videos/{cam}/from_timestamp" as "vfrom_{cam}",',
                f'"videos/{cam}/to_timestamp" as "vto_{cam}",',
            ]
        sel_sql = "episode_index, " + " ".join(selectors).rstrip(",")
        first_ep = str(ep_files[0]).replace("'", "''")
        vrows = conn.execute(f"SELECT {sel_sql} FROM read_parquet('{first_ep}')").fetchall()
        vcols = [d[0] for d in conn.description]
        for row in vrows:
            d = dict(zip(vcols, row, strict=True))
            epi = int(d["episode_index"])
            video_windows[epi] = {}
            for cam in video_keys:
                vc = d.get(f"vc_{cam}")
                vf = d.get(f"vf_{cam}")
                video_windows[epi][cam] = {
                    "chunk_index": "" if vc is None else str(vc).split("/")[-1],
                    "file_index": "" if vf is None else str(vf).split("/")[-1],
                    "from_timestamp": float(d.get(f"vfrom_{cam}") or 0.0),
                    "to_timestamp": float(d.get(f"vto_{cam}") or 0.0),
                }
        for ep in rows:
            ep["video_windows"] = dict(video_windows.get(ep["episode_index"], {}))
    finally:
        conn.close()

    v3_features = info.get("features") or {}
    video_keys_from_schema = sorted(
        k for k, v in v3_features.items() if isinstance(v, dict) and v.get("dtype") == "video"
    )
    if not video_keys:
        video_keys = video_keys_from_schema
    numeric_features = {
        k: v for k, v in v3_features.items() if isinstance(v, dict) and v.get("dtype") == "float32"
    }

    return {
        "info": info,
        "fps": fps,
        "data_path": data_path,
        "video_path": video_tpl,
        "episodes": rows,
        "video_keys": video_keys,
        "numeric_features": numeric_features,
        "cache_dir": cache_dir,
        "dataset": dataset,
    }


@dataclass
class _NumericSchema:
    name: str
    dim: int


def _derive_numeric_schema(feature_name: str, spec: dict) -> _NumericSchema:
    dtype = spec.get("dtype")
    shape = spec.get("shape") or []
    if dtype != "float32":
        raise ValueError(
            f"unsupported feature {feature_name}: dtype={dtype}, shape={shape} "
            "(only float32 fixed-width numeric vectors are supported)"
        )
    if len(shape) != 1 or not isinstance(shape[0], int) or shape[0] < 1:
        raise ValueError(
            f"unsupported feature {feature_name}: dtype={dtype}, shape={shape} "
            "(only 1-D fixed-width numeric vectors are supported)"
        )
    return _NumericSchema(name=feature_name, dim=int(shape[0]))


def lerobot_to_mcap(
    dataset_repo: str = DEFAULT_REPO,
    revision: str = DEFAULT_REVISION,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    episode_index: int | None = None,
    camera_key: str = DEFAULT_CAMERA_KEY,
) -> list[Path]:
    """Convert episodes from a LeRobot Dataset v3 repository to canonical MCAP."""
    _require_ffmpeg()

    repo_info = _hf_repo_info(dataset_repo, revision)
    cache_dir = output_dir / "_lerobot_cache"
    corpus = _ensure_source_archive(
        DatasetSource(
            repo_id=dataset_repo, revision=repo_info["sha"], license=repo_info["license"]
        ),
        cache_dir,
    )

    camera_keys = tuple(k.strip() for k in camera_key.split(",") if k.strip())
    for k in camera_keys:
        if k not in corpus["video_keys"]:
            raise ValueError(
                f"camera key '{k}' not found in dataset. Available: {corpus['video_keys']}"
            )

    # Numeric schemas derived from metadata (fail before any conversion)
    numeric_schemas = {
        name: _derive_numeric_schema(name, spec)
        for name, spec in corpus["numeric_features"].items()
        if name in ("observation.state", "action") or name.startswith("observation.")
    }

    episodes = corpus["episodes"]
    selected = [episode_index] if episode_index is not None else list(range(len(episodes)))
    output_paths: list[Path] = []

    dataset = corpus["dataset"]
    for ep_idx in selected:
        out = _convert_single_episode(
            corpus=corpus,
            dataset=dataset,
            output_dir=output_dir,
            episode_index=ep_idx,
            camera_keys=camera_keys,
            numeric_schemas=numeric_schemas,
            fps=int(corpus["fps"]),
        )
        output_paths.append(out)

    manifest_path = output_dir / "prepared-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "dataset": {
                    "repo_id": dataset.repo_id,
                    "revision": dataset.revision,
                    "license": dataset.license,
                },
                "camera_keys": list(camera_keys),
                "episodes_converted": len(output_paths),
                "converter_version": CONVERTER_VERSION,
            },
            indent=2,
        )
    )
    print(f"wrote {manifest_path}")
    return output_paths


def _convert_single_episode(
    corpus: dict,
    dataset: DatasetSource,
    output_dir: Path,
    episode_index: int,
    camera_keys: tuple[str, ...],
    numeric_schemas: dict[str, _NumericSchema],
    fps: int,
) -> Path:
    """Convert a single episode to canonical MCAP. Returns output path."""
    import duckdb

    ep = corpus["episodes"][episode_index]
    if ep["length"] is None or ep["length"] < 1:
        raise ValueError(f"episode {episode_index} has no frames")

    base = f"https://huggingface.co/datasets/{dataset.repo_id}/resolve/{dataset.revision}"
    cache = corpus["cache_dir"]

    # Locate the data parquet for this episode
    data_chunk = ep["data_chunk"]
    data_file = ep["data_file"]
    data_rel = corpus["data_path"].format(chunk_index=int(data_chunk), file_index=int(data_file))
    data_local = cache / "data" / f"chunk-{int(data_chunk):06d}-file-{int(data_file):06d}.parquet"
    if not data_local.exists():
        _download_file(f"{base}/{data_rel}", data_local)

    # Episode video windows per camera (v3 flat columns: videos/<cam>/from_timestamp etc.)
    from_to_per_camera: dict[str, tuple[float, float]] = {}
    video_meta_per_camera: dict[str, dict] = {}
    for cam in camera_keys:
        vw = ep.get("video_windows", {}).get(cam)
        if vw:
            from_to_per_camera[cam] = (vw["from_timestamp"], vw["to_timestamp"])
            video_meta_per_camera[cam] = vw

    # Query the data window
    conn = duckdb.connect()
    try:
        # data window query uses `index` (row position in chunk file), not frame_index
        data_escaped = str(data_local).replace("'", "''")
        index_col = (
            "index"
            if "index"
            in [
                d[0]
                for d in conn.execute(
                    f"SELECT * FROM read_parquet('{data_escaped}') LIMIT 0"
                ).description
            ]
            else "frame_index"
        )
        data_from, data_to = int(ep["data_from"]), int(ep["data_to"])
        ep_data = conn.execute(
            f"SELECT * FROM read_parquet('{data_escaped}') WHERE {index_col} >= {data_from} AND {index_col} < {data_to} ORDER BY {index_col}"
        ).fetchall()
        cols = [d[0] for d in conn.description]
        if not ep_data:
            raise ValueError(
                f"no data rows for episode {episode_index} window {data_from}-{data_to}"
            )
    finally:
        conn.close()

    # Build feature arrays from the window
    def _feature_rows(name: str) -> list | None:
        if name not in cols:
            return None
        idx = cols.index(name)
        return [row[idx] for row in ep_data]

    states = _feature_rows("observation.state")
    actions = _feature_rows("action")
    timestamps = _feature_rows("timestamp")
    if states is None or actions is None or timestamps is None:
        raise RuntimeError("required features observation.state/action/timestamp missing")

    frame_count = len(states)

    # Per-camera video: download chunk video, slice to episode window, transcode
    video_units_per_camera: dict[str, tuple[list[bytes], list[float]]] = {}
    for cam in camera_keys:
        cam_video_meta = video_meta_per_camera.get(cam, {})
        fts = from_to_per_camera.get(cam)
        from_ts = fts[0] if fts is not None else 0.0
        to_ts = fts[1] if fts is not None else 0.0

        cam_key = (
            cam  # v3 video_path template keys on the full feature name (e.g. observation.images.up)
        )
        vchunk = (
            str(cam_video_meta.get("chunk_index"))
            if cam_video_meta.get("chunk_index") is not None
            else (data_chunk or "0")
        )
        vchunk = vchunk.split("/")[-1]
        vfile = (
            str(cam_video_meta.get("file_index"))
            if cam_video_meta.get("file_index") is not None
            else (data_file or "0")
        )
        vfile = vfile.split("/")[-1]
        vrel = corpus["video_path"].format(
            chunk_index=int(vchunk or 0),
            file_index=int(vfile or 0),
            video_key=cam_key,
            camera_key=cam_key,
        )
        vlocal = cache / "videos" / f"{cam.replace('/', '_').replace('.', '_')}-chunk{vchunk}.mp4"
        if not vlocal.exists():
            _download_file(f"{base}/{vrel}", vlocal)

        if from_ts > 0.0 or to_ts > 0.0:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_name = tmp.name
            try:
                sliced = _slice_video(vlocal, from_ts, to_ts, Path(tmp_name))
                units = _transcode_mp4_to_h264(sliced, 1.0, float(fps))
                pts = _get_video_pts_times(sliced)
            finally:
                Path(tmp_name).unlink(missing_ok=True)
        else:
            units = _transcode_mp4_to_h264(vlocal, 1.0, float(fps))
            pts = _get_video_pts_times(vlocal)

        if len(units) != frame_count:
            raise ValueError(
                f"camera {cam}: frame count mismatch: {frame_count} parquet vs {len(units)} video access units"
            )
        if len(pts) != frame_count:
            raise ValueError(f"camera {cam}: PTS count mismatch: {frame_count} vs {len(pts)}")
        for i in range(frame_count):
            if abs(timestamps[i] - pts[i]) > PRESENTATION_TIMESTAMP_EPSILON_S:
                raise ValueError(
                    f"camera {cam} timestamp disagreement at frame {i}: "
                    f"parquet={timestamps[i]:.6f}s video_pts={pts[i]:.6f}s"
                )

        video_units_per_camera[cam] = (units, pts)

    # Write MCAP
    out_name = f"lerobot_episode_{episode_index + 1:04d}.mcap"
    output_path = output_dir / "landing" / out_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from foxglove_schemas_protobuf.CompressedVideo_pb2 import CompressedVideo
    from mcap_protobuf.schema import build_file_descriptor_set

    schema_data = build_file_descriptor_set(CompressedVideo).SerializeToString()
    state_schema_name = "lerobot_msgs/msg/State"
    action_schema_name = "lerobot_msgs/msg/Action"
    state_schema_text = f"float32[{numeric_schemas['observation.state'].dim}] position"
    action_schema_text = f"float32[{numeric_schemas['action'].dim}] action"
    state_schema_data = state_schema_text.encode("utf-8")
    action_schema_data = action_schema_text.encode("utf-8")

    source_uri = f"hf://datasets/{dataset.repo_id}@{dataset.revision}"
    with tempfile.TemporaryDirectory(prefix="lerobot-source-episode-") as temporary_directory:
        source_episode_path = Path(temporary_directory) / output_path.name
        with source_episode_path.open("wb") as source_stream:
            writer = McapWriter(source_stream)
            writer.start(profile="", library="hflow LeRobot source adapter")

            video_schema_id = writer.register_schema(
                "foxglove.CompressedVideo", "protobuf", schema_data
            )
            state_schema_id = writer.register_schema(
                state_schema_name, "ros2msg", state_schema_data
            )
            action_schema_id = writer.register_schema(
                action_schema_name, "ros2msg", action_schema_data
            )

            video_channels: dict[str, int] = {}
            for cam in camera_keys:
                video_channels[cam] = writer.register_channel(
                    topic=f"/{cam}", message_encoding="protobuf", schema_id=video_schema_id
                )
            state_channel_id = writer.register_channel(
                topic="/observation.state", message_encoding="cdr", schema_id=state_schema_id
            )
            action_channel_id = writer.register_channel(
                topic="/action", message_encoding="cdr", schema_id=action_schema_id
            )

            for frame_index in range(frame_count):
                log_time_ns = EPISODE_START_TIME_NS + round(
                    frame_index * NANOSECONDS_PER_SECOND / fps
                )
                for cam in camera_keys:
                    units, _ = video_units_per_camera[cam]
                    video_message = CompressedVideo()
                    video_message.timestamp.seconds = log_time_ns // NANOSECONDS_PER_SECOND
                    video_message.timestamp.nanos = log_time_ns % NANOSECONDS_PER_SECOND
                    video_message.frame_id = cam
                    video_message.data = units[frame_index]
                    video_message.format = "h264"
                    writer.add_message(
                        channel_id=video_channels[cam],
                        log_time=log_time_ns,
                        data=video_message.SerializeToString(),
                        publish_time=log_time_ns,
                        sequence=frame_index,
                    )
                writer.add_message(
                    channel_id=state_channel_id,
                    log_time=log_time_ns,
                    data=_encode_cdr_float32_array(states[frame_index]),
                    publish_time=log_time_ns,
                    sequence=frame_index,
                )
                writer.add_message(
                    channel_id=action_channel_id,
                    log_time=log_time_ns,
                    data=_encode_cdr_float32_array(actions[frame_index]),
                    publish_time=log_time_ns,
                    sequence=frame_index,
                )

            writer.add_metadata(
                name="episode/v1",
                data={
                    "task": str(ep["task"] or ""),
                    "operator": "lerobot_converter",
                    "success": "true",
                    "embodiment": corpus["info"].get("robot_type", "unknown"),
                    "source_dataset": dataset.repo_id,
                    "source_revision": dataset.revision,
                    "source_episode_index": str(episode_index),
                    "converter_version": CONVERTER_VERSION,
                },
            )
            writer.add_metadata(
                name="source-provenance/v1",
                data={
                    "converter_version": CONVERTER_VERSION,
                    "ffmpeg_version": _get_ffmpeg_version(),
                    "source_uri": source_uri,
                },
            )
            writer.finish()

        hflow.write_canonical_episode(
            source_episode_path,
            output_path,
            hflow.TransformConfig(gop_seconds=1.0),
            source_uri=source_uri,
        )

    print(f"wrote {output_path} ({output_path.stat().st_size / 1_000_000:.1f} MB)")
    return output_path


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face dataset repo")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="Dataset revision")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory"
    )
    parser.add_argument(
        "--camera-key",
        default=DEFAULT_CAMERA_KEY,
        help="Comma-separated camera keys (default: observation.image)",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=None,
        help="Episode index to convert, or None for all episodes",
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
