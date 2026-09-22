#!/usr/bin/env python3
"""Convert WebDataset-style tar archives into canonical HFlow MCAP episodes.

Real egocentric corpora such as builddotai/Egocentric-10K and Egocentric-100K
ship as WebDataset-style tar archives of video clips (e.g., ~180 s H.265/MP4),
per-clip JSON sidecars, and a per-worker intrinsics.json calibration file.

This converter reads the tar directly, pairs each video with its sidecar,
maps sidecar fields (factory, worker, duration, fps, codec) onto episode/v1
metadata records, attaches intrinsics.json as a calibration attachment, and
transcodes the video directly to Annex B H.264 without an intermediate JPEG
roundtrip.

Usage:
    uv run python examples/egocentric/convert.py <archive.tar> \\
        --output-dir data/egocentric/converted \\
        --canonical
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

from foxglove_schemas_protobuf.CompressedVideo_pb2 import CompressedVideo
from mcap.writer import Writer
from mcap_protobuf.schema import build_file_descriptor_set

from hflow.ffmpeg import ffmpeg_path, ffmpeg_version
from hflow.format import EPISODE_KEY_ROBOT_SOFTWARE_VERSION, METADATA_RECORD_EPISODE
from hflow.transform import TransformConfig, write_canonical_episode
from hflow.video import AccessUnit, split_annex_b_stream

logger = logging.getLogger(__name__)

CONVERTER_VERSION = "egocentric-webdataset-converter-v1"
DEFAULT_START_TIME_NS = 1_755_000_000_000_000_000
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}


@dataclass(frozen=True)
class ClipSidecar:
    """Metadata parsed from a per-clip JSON sidecar in the WebDataset archive."""

    factory_id: str
    worker_id: str
    duration_s: float | None = None
    fps: float | None = None
    codec: str | None = None
    task: str = "unlabeled"
    operator_id: str | None = None
    raw_data: dict[str, object] | None = None


def parse_sidecar(sidecar_json_bytes: bytes, member_name: str) -> ClipSidecar:
    """Parse and validate sidecar JSON content."""
    try:
        data = json.loads(sidecar_json_bytes.decode("utf-8"))
    except Exception as error:
        raise RuntimeError(f"sidecar {member_name!r} is not valid JSON: {error}") from error

    if not isinstance(data, dict):
        raise RuntimeError(f"sidecar {member_name!r} must be a JSON object")

    raw_factory = (
        data.get("factory_id") if isinstance(data.get("factory_id"), str) else data.get("factory")
    )
    raw_worker = (
        data.get("worker_id") if isinstance(data.get("worker_id"), str) else data.get("worker")
    )

    if (
        not isinstance(raw_factory, str)
        or not raw_factory.strip()
        or not isinstance(raw_worker, str)
        or not raw_worker.strip()
    ):
        raise RuntimeError(
            f"sidecar {member_name!r} is missing a usable 'factory_id' or 'worker_id'"
        )

    factory_id = raw_factory.strip()
    worker_id = raw_worker.strip()

    raw_operator = data.get("operator", data.get("operator_id"))
    operator_id = (
        str(raw_operator).strip()
        if isinstance(raw_operator, str) and raw_operator.strip()
        else None
    )

    raw_duration = data.get("duration_sec", data.get("duration_s", data.get("duration")))
    duration_s = float(raw_duration) if isinstance(raw_duration, (int, float)) else None

    raw_fps = data.get("fps")
    fps = float(raw_fps) if isinstance(raw_fps, (int, float)) else None

    raw_codec = data.get("codec")
    codec = str(raw_codec) if isinstance(raw_codec, str) and raw_codec else None

    raw_task = data.get("task", data.get("task_id", "unlabeled"))
    task = str(raw_task) if raw_task else "unlabeled"

    return ClipSidecar(
        factory_id=factory_id,
        worker_id=worker_id,
        duration_s=duration_s,
        fps=fps,
        codec=codec,
        task=task,
        operator_id=operator_id,
        raw_data=data,
    )


def transcode_video_to_annex_b_h264(
    video_path: Path,
    target_fps: float = 10.0,
    target_width: int = 640,
    target_height: int = 360,
    max_duration_s: float | None = None,
) -> list[AccessUnit]:
    """Transcode source video to Annex B H.264 access units with AUD and SPS/PPS on keyframes."""
    keyframe_interval = max(1, round(target_fps))
    x264_parameters = (
        f"keyint={keyframe_interval}:min-keyint={keyframe_interval}:"
        "scenecut=0:bframes=0:repeat-headers=1:aud=1"
    )
    filter_string = (
        f"fps={target_fps:g},"
        f"scale={target_width}:{target_height}:"
        "force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:black"
    )

    command = [
        str(ffmpeg_path()),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        filter_string,
        "-an",
    ]
    if max_duration_s is not None and max_duration_s > 0:
        frame_limit = round(max_duration_s * target_fps)
        command.extend(["-frames:v", str(frame_limit)])

    command.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-x264-params",
            x264_parameters,
            "-fps_mode",
            "passthrough",
            "-f",
            "h264",
            "pipe:1",
        ]
    )

    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed ({' '.join(command)}): {completed.stderr.decode(errors='replace')}"
        )

    access_units = split_annex_b_stream(completed.stdout)
    if not access_units:
        raise RuntimeError(f"ffmpeg produced no video frames for {video_path}")

    return access_units


def write_webdataset_episode(
    video_path: Path,
    output_path: Path,
    sidecar: ClipSidecar,
    source_member: str,
    tar_path: Path,
    intrinsics_bytes: bytes | None = None,
    episode_index: int = 0,
    target_fps: float = 10.0,
    target_width: int = 640,
    target_height: int = 360,
    max_duration_s: float | None = None,
) -> None:
    """Write an input-shaped MCAP episode for a single WebDataset clip."""
    effective_fps = sidecar.fps if sidecar.fps and sidecar.fps > 0 else target_fps
    access_units = transcode_video_to_annex_b_h264(
        video_path,
        target_fps=effective_fps,
        target_width=target_width,
        target_height=target_height,
        max_duration_s=max_duration_s or sidecar.duration_s,
    )

    episode_start_time_ns = DEFAULT_START_TIME_NS + episode_index * 60_000_000_000
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as output_stream:
        writer = Writer(output_stream)
        writer.start(profile="", library="hflow egocentric webdataset converter")

        schema_id = writer.register_schema(
            name="foxglove.CompressedVideo",
            encoding="protobuf",
            data=build_file_descriptor_set(CompressedVideo).SerializeToString(),
        )
        channel_id = writer.register_channel(
            topic="/head_camera/compressed",
            message_encoding="protobuf",
            schema_id=schema_id,
        )

        # 1. episode/v1 metadata record
        episode_metadata = {
            "task": sidecar.task,
            "factory": sidecar.factory_id,
            "worker": sidecar.worker_id,
            "operator": sidecar.operator_id or f"{sidecar.factory_id}_{sidecar.worker_id}",
            EPISODE_KEY_ROBOT_SOFTWARE_VERSION: "build-ai-gen-1",
            "source_archive": tar_path.name,
            "source_member": source_member,
            "task_completion": "unlabeled",
        }
        if sidecar.duration_s is not None:
            episode_metadata["duration"] = f"{sidecar.duration_s:g}"
        if sidecar.fps is not None:
            episode_metadata["fps"] = f"{sidecar.fps:g}"
        if sidecar.codec is not None:
            episode_metadata["codec"] = sidecar.codec

        writer.add_metadata(
            name=METADATA_RECORD_EPISODE,
            data=episode_metadata,
        )

        # 2. source-provenance/v1 metadata record
        writer.add_metadata(
            name="source-provenance/v1",
            data={
                "converter_version": CONVERTER_VERSION,
                "ffmpeg_version": ffmpeg_version(),
                "source_archive": str(tar_path),
                "source_member": source_member,
            },
        )

        # 3. Calibration attachment (intrinsics.json)
        if intrinsics_bytes is not None:
            writer.add_attachment(
                name="intrinsics.json",
                media_type="application/json",
                data=intrinsics_bytes,
                log_time=episode_start_time_ns,
                create_time=episode_start_time_ns,
            )

        # 4. Write video messages
        frame_interval_ns = round(1_000_000_000 / effective_fps)
        for frame_index, access_unit in enumerate(access_units):
            log_time_ns = episode_start_time_ns + frame_index * frame_interval_ns
            message = CompressedVideo()
            message.timestamp.FromNanoseconds(log_time_ns)
            message.frame_id = "head_camera"
            message.data = access_unit.data
            message.format = "h264"
            writer.add_message(
                channel_id=channel_id,
                log_time=log_time_ns,
                data=message.SerializeToString(),
                publish_time=log_time_ns,
                sequence=frame_index,
            )

        writer.finish()


def convert_webdataset_tar(
    tar_path: Path,
    output_dir: Path,
    *,
    canonical: bool = False,
    target_fps: float = 10.0,
    target_width: int = 640,
    target_height: int = 360,
    max_duration_s: float | None = None,
) -> list[Path]:
    """Convert all clips in a WebDataset tar archive to HFlow MCAPs."""
    if not tar_path.is_file():
        raise FileNotFoundError(f"tar archive not found: {tar_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    converted_paths: list[Path] = []

    with tempfile.TemporaryDirectory(prefix="hflow_webdataset_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        with tarfile.open(tar_path, mode="r") as tar:
            members = tar.getmembers()
            members_by_name = {m.name: m for m in members if m.isfile()}

            # Locate intrinsics.json if present anywhere in the tar
            intrinsics_bytes: bytes | None = None
            for name, member in members_by_name.items():
                if Path(name).name == "intrinsics.json":
                    extracted = tar.extractfile(member)
                    if extracted is not None:
                        with extracted:
                            intrinsics_bytes = extracted.read()
                    break

            # Find matching video and sidecar pairs
            video_members: list[tarfile.TarInfo] = []
            for name, member in members_by_name.items():
                suffix = Path(name).suffix.lower()
                if suffix in VIDEO_EXTENSIONS:
                    video_members.append(member)

            video_members.sort(key=lambda m: m.name)

            if not video_members:
                raise RuntimeError(f"archive {tar_path.name!r} contains no video members")

            # Phase 1: Pre-validate all video members and their sidecars ahead of writes
            planned_clips: list[tuple[tarfile.TarInfo, ClipSidecar]] = []
            seen_stems: set[str] = set()

            for video_member in video_members:
                video_stem = Path(video_member.name).stem
                if video_stem in seen_stems:
                    raise RuntimeError(
                        f"duplicate video stem {video_stem!r} in archive {tar_path.name!r}"
                    )
                seen_stems.add(video_stem)

                sidecar_candidates = [
                    str(Path(video_member.name).with_suffix(".json").as_posix()),
                    f"{video_stem}.json",
                ]

                sidecar_info: tarfile.TarInfo | None = None
                for candidate in sidecar_candidates:
                    if candidate in members_by_name:
                        sidecar_info = members_by_name[candidate]
                        break

                if sidecar_info is None:
                    raise RuntimeError(
                        f"missing sidecar for source video {video_member.name!r} in {tar_path.name}"
                    )

                sidecar_stream = tar.extractfile(sidecar_info)
                if sidecar_stream is None:
                    raise RuntimeError(f"could not read sidecar {sidecar_info.name!r}")
                with sidecar_stream:
                    sidecar = parse_sidecar(sidecar_stream.read(), sidecar_info.name)

                planned_clips.append((video_member, sidecar))

            # Phase 2: Transcode and write episodes
            for episode_index, (video_member, sidecar) in enumerate(planned_clips):
                video_stem = Path(video_member.name).stem

                # Extract video to temporary file
                video_stream = tar.extractfile(video_member)
                if video_stream is None:
                    raise RuntimeError(f"could not read video member {video_member.name!r}")

                temp_video_path = temp_dir / Path(video_member.name).name
                with temp_video_path.open("wb") as dest:
                    shutil.copyfileobj(video_stream, dest)

                landing_mcap_path = output_dir / f"{video_stem}.mcap"
                write_webdataset_episode(
                    video_path=temp_video_path,
                    output_path=landing_mcap_path,
                    sidecar=sidecar,
                    source_member=video_member.name,
                    tar_path=tar_path,
                    intrinsics_bytes=intrinsics_bytes,
                    episode_index=episode_index,
                    target_fps=target_fps,
                    target_width=target_width,
                    target_height=target_height,
                    max_duration_s=max_duration_s,
                )

                if canonical:
                    canonical_path = output_dir / f"{video_stem}.canonical.mcap"
                    write_canonical_episode(landing_mcap_path, canonical_path, TransformConfig())
                    landing_mcap_path.unlink()
                    converted_paths.append(canonical_path)
                else:
                    converted_paths.append(landing_mcap_path)

                temp_video_path.unlink(missing_ok=True)

    return converted_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("tar_path", type=Path, help="path to WebDataset-style tar archive")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/egocentric/converted"),
        help="output directory for converted MCAP episodes (default: data/egocentric/converted)",
    )
    parser.add_argument(
        "--canonical",
        action="store_true",
        help="transform landing episodes to canonical MCAPs immediately",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="fallback video frame rate if not declared in sidecar (default: 10.0)",
    )
    arguments = parser.parse_args()

    results = convert_webdataset_tar(
        tar_path=arguments.tar_path,
        output_dir=arguments.output_dir,
        canonical=arguments.canonical,
        target_fps=arguments.fps,
    )
    print(f"Successfully converted {len(results)} episode(s) into {arguments.output_dir}")
    for path in results:
        print(f"  - {path.name} ({path.stat().st_size / 1_000_000:.2f} MB)")


if __name__ == "__main__":
    main()
