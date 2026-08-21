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
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import hflow

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


def _video_episode_spec(
    manifest: CorpusManifest,
    episode: PlannedEpisode,
    episode_index: int,
) -> hflow.testing.VideoEpisodeSpec:
    return hflow.testing.VideoEpisodeSpec(
        source_start_s=episode.source_start_s,
        duration_s=episode.duration_s,
        image_hz=EPISODE_IMAGE_HZ,
        image_width=EPISODE_IMAGE_WIDTH,
        image_height=EPISODE_IMAGE_HEIGHT,
        camera_name="head_camera",
        black_segment=(episode.fault_segment_s if episode.fault is FaultKind.BLACKOUT else None),
        freeze_segment=episode.fault_segment_s if episode.fault is FaultKind.FREEZE else None,
        start_time_ns=EPISODE_START_TIME_NS + episode_index * 60_000_000_000,
        task=episode.task,
        operator="factory_051_worker_001",
        success=None,
        robot_software_version="build-ai-gen-1",
        metadata=(
            ("source_dataset", manifest.dataset.repo_id),
            ("source_revision", manifest.dataset.revision),
            ("source_member", episode.source_member),
            ("source_start_s", f"{episode.source_start_s:g}"),
            ("source_license", manifest.dataset.license),
            ("injected_fault", episode.fault.value),
            ("task_completion", "unlabeled"),
        ),
    )


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
        hflow.testing.write_video_episode(
            source_paths[episode.source_member],
            output_path,
            _video_episode_spec(manifest, episode, episode_index),
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
