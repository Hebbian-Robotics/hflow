"""Download and prepare the pinned egocentric factory corpus.

The source archive stays under the ignored ``data/`` tree. The output is a
deterministically generated set of input-shaped MCAP episodes, including a
small number of camera faults declared in ``manifest.json``. Run the normal
hflow transform and checks on those files; this script does not precompute
any pipeline result.
"""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from foxglove_schemas_protobuf.CompressedVideo_pb2 import CompressedVideo
from mcap.writer import Writer
from mcap_protobuf.schema import build_file_descriptor_set

from hflow.ffmpeg import ffmpeg_path, ffmpeg_version
from hflow.format import EPISODE_KEY_ROBOT_SOFTWARE_VERSION, METADATA_RECORD_EPISODE
from hflow.video import AccessUnit, split_annex_b_stream

DEFAULT_MANIFEST_PATH = Path(__file__).with_name("manifest.json")
DEFAULT_SOURCE_ROOT = Path("data/egocentric")
DEFAULT_OUTPUT_ROOT = Path("data/egocentric")
EPISODE_IMAGE_HZ = 10.0
EPISODE_IMAGE_WIDTH = 640
EPISODE_IMAGE_HEIGHT = 360
EPISODE_START_TIME_NS = 1_755_000_000_000_000_000


class FaultKind(StrEnum):
    NONE = "none"
    BLACKOUT = "blackout"
    FREEZE = "freeze"


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
class PlannedFault:
    episode_number: int
    fault: FaultKind
    fault_segment_s: tuple[float, float]


@dataclass(frozen=True)
class EpisodePlan:
    total_episodes: int
    duration_s: float
    first_source_start_s: float
    source_stride_s: float
    faults: tuple[PlannedFault, ...]


@dataclass(frozen=True)
class PlannedEpisode:
    episode_id: str
    source_member: str
    source_start_s: float
    duration_s: float
    task: str
    fault: FaultKind
    fault_segment_s: tuple[float, float] | None


@dataclass(frozen=True)
class CorpusManifest:
    schema_version: int
    dataset: DatasetSource
    archive: SourceArchive
    sources: tuple[SourceVideo, ...]
    episode_plan: EpisodePlan
    episodes: tuple[PlannedEpisode, ...]


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


def _parse_fault_segment(value: object, context: str) -> tuple[float, float]:
    array = _require_array(value, context)
    if len(array) != 2:
        raise ValueError(f"{context} must contain [start_s, end_s]")
    return (
        _require_number(array[0], f"{context}[0]"),
        _require_number(array[1], f"{context}[1]"),
    )


def _expand_episode_plan(
    sources: tuple[SourceVideo, ...],
    episode_plan: EpisodePlan,
) -> tuple[PlannedEpisode, ...]:
    planned_faults_by_episode_number = {
        planned_fault.episode_number: planned_fault for planned_fault in episode_plan.faults
    }
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

        planned_fault = planned_faults_by_episode_number.get(episode_number)
        episodes.append(
            PlannedEpisode(
                episode_id=f"factory_051_episode_{episode_number:04d}",
                source_member=source_video.member,
                source_start_s=source_start_s,
                duration_s=episode_plan.duration_s,
                task=source_video.task,
                fault=planned_fault.fault if planned_fault is not None else FaultKind.NONE,
                fault_segment_s=(
                    planned_fault.fault_segment_s if planned_fault is not None else None
                ),
            )
        )
    return tuple(episodes)


def _load_manifest(manifest_path: Path) -> CorpusManifest:
    root = _require_object(json.loads(manifest_path.read_text()), "manifest")
    schema_version_value = root.get("schema_version")
    if schema_version_value != 2:
        raise ValueError(f"unsupported manifest schema_version {schema_version_value!r}")

    dataset_object = _require_object(root.get("dataset"), "dataset")
    archive_object = _require_object(root.get("archive"), "archive")
    source_objects = _require_array(root.get("sources"), "sources")
    episode_plan_object = _require_object(root.get("episode_plan"), "episode_plan")
    fault_objects = _require_array(episode_plan_object.get("faults"), "episode_plan.faults")

    dataset = DatasetSource(
        repo_id=_require_string(dataset_object.get("repo_id"), "dataset.repo_id"),
        revision=_require_string(dataset_object.get("revision"), "dataset.revision"),
        license=_require_string(dataset_object.get("license"), "dataset.license"),
    )
    archive = SourceArchive(
        path=_require_string(archive_object.get("path"), "archive.path"),
        sha256=_require_string(archive_object.get("sha256"), "archive.sha256"),
    )
    sources = tuple(
        SourceVideo(
            member=_require_string(source_object.get("member"), f"sources[{source_index}].member"),
            sha256=_require_string(source_object.get("sha256"), f"sources[{source_index}].sha256"),
            duration_s=_require_number(
                source_object.get("duration_s"), f"sources[{source_index}].duration_s"
            ),
            task=_require_string(source_object.get("task"), f"sources[{source_index}].task"),
        )
        for source_index, source_value in enumerate(source_objects)
        for source_object in [_require_object(source_value, f"sources[{source_index}]")]
    )

    planned_faults: list[PlannedFault] = []
    for fault_index, fault_value in enumerate(fault_objects):
        fault_object = _require_object(fault_value, f"episode_plan.faults[{fault_index}]")
        fault = FaultKind(
            _require_string(fault_object.get("fault"), f"episode_plan.faults[{fault_index}].fault")
        )
        if fault is FaultKind.NONE:
            raise ValueError(f"episode_plan.faults[{fault_index}].fault cannot be none")
        planned_faults.append(
            PlannedFault(
                episode_number=_require_integer(
                    fault_object.get("episode_number"),
                    f"episode_plan.faults[{fault_index}].episode_number",
                ),
                fault=fault,
                fault_segment_s=_parse_fault_segment(
                    fault_object.get("fault_segment_s"),
                    f"episode_plan.faults[{fault_index}].fault_segment_s",
                ),
            )
        )

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
        faults=tuple(planned_faults),
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
    fault_episode_numbers = [fault.episode_number for fault in episode_plan.faults]
    if len(fault_episode_numbers) != len(set(fault_episode_numbers)):
        raise ValueError("episode_plan.faults contains duplicate episode numbers")
    for planned_fault in episode_plan.faults:
        fault_start_s, fault_end_s = planned_fault.fault_segment_s
        if not 1 <= planned_fault.episode_number <= episode_plan.total_episodes:
            raise ValueError(
                f"fault episode_number {planned_fault.episode_number} is outside the episode plan"
            )
        if fault_start_s < 0 or fault_start_s >= fault_end_s:
            raise ValueError(
                f"fault segment for episode {planned_fault.episode_number} must have "
                "0 <= start < end"
            )
        if fault_end_s > episode_plan.duration_s:
            raise ValueError(
                f"fault segment for episode {planned_fault.episode_number} ends after the episode"
            )

    episodes = _expand_episode_plan(sources, episode_plan)

    return CorpusManifest(
        schema_version=2,
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
    download_root = data_root / "huggingface"
    archive_path = download_root / manifest.archive.path
    if not archive_path.is_file():
        if shutil.which("hf") is None:
            raise RuntimeError(
                "the `hf` CLI is required to download this dataset but is not on PATH. "
                "Install it with `uv tool install -U huggingface_hub`, then "
                "`hf auth login`."
            )
        command = [
            "hf",
            "download",
            manifest.dataset.repo_id,
            manifest.archive.path,
            "--repo-type",
            "dataset",
            "--revision",
            manifest.dataset.revision,
            "--local-dir",
            str(download_root),
            "--max-workers",
            "2",
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                "Hugging Face download failed. Run `hf auth whoami` and accept the "
                f"access terms for {manifest.dataset.repo_id}."
            )
    _verify_sha256(archive_path, manifest.archive.sha256)
    return archive_path


def _extract_source_videos(
    manifest: CorpusManifest,
    archive_path: Path,
    data_root: Path,
) -> dict[str, Path]:
    source_root = data_root / "source"
    source_root.mkdir(parents=True, exist_ok=True)
    source_paths: dict[str, Path] = {}
    with tarfile.open(archive_path, mode="r") as source_archive:
        for source_video in manifest.sources:
            destination_path = source_root / Path(source_video.member).name
            if destination_path.is_file():
                _verify_sha256(destination_path, source_video.sha256)
                source_paths[source_video.member] = destination_path
                continue
            archive_member = source_archive.getmember(source_video.member)
            if not archive_member.isfile() or archive_member.name != Path(archive_member.name).name:
                raise RuntimeError(f"unsafe or non-file archive member {archive_member.name!r}")
            source_stream = source_archive.extractfile(archive_member)
            if source_stream is None:
                raise RuntimeError(f"could not read archive member {archive_member.name!r}")
            with tempfile.NamedTemporaryFile(
                prefix=f".{destination_path.name}.", dir=source_root, delete=False
            ) as temporary_stream:
                temporary_path = Path(temporary_stream.name)
                shutil.copyfileobj(source_stream, temporary_stream)
            temporary_path.replace(destination_path)
            _verify_sha256(destination_path, source_video.sha256)
            source_paths[source_video.member] = destination_path
    return source_paths


def _fault_frame_range(episode: PlannedEpisode) -> tuple[int, int] | None:
    if episode.fault_segment_s is None:
        return None
    fault_start_s, fault_end_s = episode.fault_segment_s
    first_frame = math.ceil(fault_start_s * EPISODE_IMAGE_HZ)
    last_frame = math.ceil(fault_end_s * EPISODE_IMAGE_HZ) - 1
    return first_frame, last_frame


def _video_filter_arguments(episode: PlannedEpisode) -> list[str]:
    base_filter = (
        f"fps={EPISODE_IMAGE_HZ:g},"
        f"scale={EPISODE_IMAGE_WIDTH}:{EPISODE_IMAGE_HEIGHT}:"
        "force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={EPISODE_IMAGE_WIDTH}:{EPISODE_IMAGE_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black"
    )
    fault_frame_range = _fault_frame_range(episode)
    if episode.fault is FaultKind.NONE:
        return ["-vf", base_filter]
    if fault_frame_range is None:
        raise ValueError(
            f"episode {episode.episode_id} declares {episode.fault.value} without a segment"
        )

    first_fault_frame, last_fault_frame = fault_frame_range
    if episode.fault is FaultKind.BLACKOUT:
        # drawbox runs before H.264 encoding, so the source is decoded and encoded
        # exactly once while every selected pixel in the segment becomes black.
        blackout_filter = (
            "drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:"
            f"enable=between(n\\,{first_fault_frame}\\,{last_fault_frame})"
        )
        return ["-vf", f"{base_filter},{blackout_filter}"]
    if episode.fault is FaultKind.FREEZE:
        # freezeframes takes a main and replacement input. Splitting the filtered
        # source lets it replace the interval with its first frame in the same
        # decode/filter/encode invocation.
        filter_graph = (
            f"[0:v]{base_filter},split=2[main][replacement];"
            "[main][replacement]freezeframes="
            f"first={first_fault_frame}:last={last_fault_frame}:replace={first_fault_frame}[video]"
        )
        return ["-filter_complex", filter_graph, "-map", "[video]"]
    raise AssertionError(f"unhandled fault kind {episode.fault!r}")


def _transcode_episode_to_h264(
    source_video_path: Path, episode: PlannedEpisode
) -> list[AccessUnit]:
    frame_count = round(episode.duration_s * EPISODE_IMAGE_HZ)
    keyframe_interval = max(1, round(EPISODE_IMAGE_HZ))
    x264_parameters = (
        f"keyint={keyframe_interval}:min-keyint={keyframe_interval}:"
        "scenecut=0:bframes=0:repeat-headers=1:aud=1"
    )
    command = [
        str(ffmpeg_path()),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{episode.source_start_s:g}",
        "-i",
        str(source_video_path),
        *_video_filter_arguments(episode),
        "-an",
        "-frames:v",
        str(frame_count),
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
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed ({' '.join(command)}): {completed.stderr.decode(errors='replace')}"
        )
    try:
        access_units = split_annex_b_stream(completed.stdout)
    except ValueError as error:
        raise RuntimeError(
            f"ffmpeg produced invalid H.264 for {episode.episode_id}: {error}"
        ) from error
    if len(access_units) != frame_count:
        raise RuntimeError(
            f"expected {frame_count} frames from {source_video_path}, got {len(access_units)}; "
            "the requested excerpt may extend past the source video"
        )
    for frame_index, access_unit in enumerate(access_units):
        keyframe_expected = frame_index % keyframe_interval == 0
        if access_unit.is_keyframe != keyframe_expected:
            raise RuntimeError(
                f"frame {frame_index} from {source_video_path} has unexpected keyframe state"
            )
        if access_unit.is_keyframe and not access_unit.has_parameter_sets:
            raise RuntimeError(f"keyframe {frame_index} from {source_video_path} lacks SPS/PPS")
    return access_units


def _episode_metadata(manifest: CorpusManifest, episode: PlannedEpisode) -> dict[str, str]:
    return {
        "task": episode.task,
        "operator": "factory_051_worker_001",
        EPISODE_KEY_ROBOT_SOFTWARE_VERSION: "build-ai-gen-1",
        "source_dataset": manifest.dataset.repo_id,
        "source_revision": manifest.dataset.revision,
        "source_member": episode.source_member,
        "source_start_s": f"{episode.source_start_s:g}",
        "source_license": manifest.dataset.license,
        "injected_fault": episode.fault.value,
        "task_completion": "unlabeled",
    }


def _write_video_episode(
    source_video_path: Path,
    output_path: Path,
    manifest: CorpusManifest,
    episode: PlannedEpisode,
    episode_index: int,
) -> None:
    access_units = _transcode_episode_to_h264(source_video_path, episode)
    episode_start_time_ns = EPISODE_START_TIME_NS + episode_index * 60_000_000_000
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output_stream:
        writer = Writer(output_stream)
        writer.start(profile="", library="hflow egocentric source adapter")
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
        writer.add_metadata(name=METADATA_RECORD_EPISODE, data=_episode_metadata(manifest, episode))
        writer.add_metadata(
            name="source-provenance/v1",
            data={
                "converter_version": "1",
                "ffmpeg_version": ffmpeg_version(),
                "source_uri": (
                    f"hf://datasets/{manifest.dataset.repo_id}@{manifest.dataset.revision}/"
                    f"{episode.source_member}"
                ),
            },
        )
        for frame_index, access_unit in enumerate(access_units):
            log_time_ns = episode_start_time_ns + round(
                frame_index * 1_000_000_000 / EPISODE_IMAGE_HZ
            )
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


def _write_prepared_manifest(
    manifest: CorpusManifest,
    prepared_episode_paths: list[Path],
    output_path: Path,
) -> None:
    prepared_manifest = {
        "schema_version": manifest.schema_version,
        "source": {
            "repo_id": manifest.dataset.repo_id,
            "revision": manifest.dataset.revision,
            "license": manifest.dataset.license,
            "archive_path": manifest.archive.path,
            "archive_sha256": manifest.archive.sha256,
        },
        "generation": {
            "total_episodes": manifest.episode_plan.total_episodes,
            "duration_s": manifest.episode_plan.duration_s,
            "injected_faults": len(manifest.episode_plan.faults),
        },
        "episodes": [
            {
                "episode_id": episode.episode_id,
                "path": str(prepared_path),
                "sha256": _sha256_file(prepared_path),
                "size_bytes": prepared_path.stat().st_size,
                "task": episode.task,
                "injected_fault": episode.fault.value,
            }
            for episode, prepared_path in zip(
                manifest.episodes, prepared_episode_paths, strict=True
            )
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output_path = output_path.with_suffix(".json.tmp")
    temporary_output_path.write_text(json.dumps(prepared_manifest, indent=2) + "\n")
    temporary_output_path.replace(output_path)


def prepare_corpus(manifest_path: Path, source_root: Path, output_root: Path) -> list[Path]:
    manifest = _load_manifest(manifest_path)
    archive_path = _ensure_source_archive(manifest, source_root)
    source_paths = _extract_source_videos(manifest, archive_path, source_root)
    landing_root = output_root / "landing"
    landing_root.mkdir(parents=True, exist_ok=True)

    prepared_episode_paths: list[Path] = []
    for episode_index, episode in enumerate(manifest.episodes):
        output_path = landing_root / f"{episode.episode_id}.mcap"
        _write_video_episode(
            source_paths[episode.source_member],
            output_path,
            manifest,
            episode,
            episode_index,
        )
        prepared_episode_paths.append(output_path)
        prepared_count = episode_index + 1
        if (
            prepared_count % 12 == 0
            or prepared_count == len(manifest.episodes)
            or episode.fault is not FaultKind.NONE
        ):
            print(
                f"prepared {prepared_count}/{len(manifest.episodes)}: {output_path} "
                f"({output_path.stat().st_size / 1_000_000:.1f} MB, "
                f"fault={episode.fault.value})"
            )

    prepared_manifest_path = output_root / "prepared-manifest.json"
    _write_prepared_manifest(manifest, prepared_episode_paths, prepared_manifest_path)
    print(f"wrote {prepared_manifest_path}")
    return prepared_episode_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args()
    prepare_corpus(arguments.manifest, arguments.source_root, arguments.output_root)


if __name__ == "__main__":
    main()
