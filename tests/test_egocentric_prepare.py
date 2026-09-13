"""The egocentric converter lands H.264 directly and preserves its planted faults."""

import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from types import ModuleType

import pytest
from foxglove_schemas_protobuf.CompressedVideo_pb2 import CompressedVideo
from mcap.reader import make_reader

from hflow import Episode, TransformConfig, write_canonical_episode
from hflow.checks import camera_frame_stats
from hflow.ffmpeg import ffmpeg_path


def _load_prepare_module() -> ModuleType:
    module_path = Path(__file__).parents[1] / "examples" / "egocentric" / "prepare.py"
    module_spec = importlib.util.spec_from_file_location("egocentric_prepare", module_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"could not load {module_path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


PREPARE = _load_prepare_module()


@pytest.fixture(scope="module")
def moving_hevc_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output_path = tmp_path_factory.mktemp("egocentric-source") / "source.mp4"
    completed = subprocess.run(
        [
            str(ffmpeg_path()),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=10:duration=24",
            "-c:v",
            "libx265",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.decode(errors="replace"))
    return output_path


def _manifest(episode: object) -> object:
    dataset = PREPARE.DatasetSource(
        repo_id="example/corpus", revision="abc123", license="apache-2.0"
    )
    source = PREPARE.SourceVideo(
        member="source.mp4", sha256="unused", duration_s=24.0, task="factory_task"
    )
    episode_plan = PREPARE.EpisodePlan(
        total_episodes=1,
        duration_s=20.0,
        first_source_start_s=1.0,
        source_stride_s=0.0,
        faults=(),
    )
    return PREPARE.CorpusManifest(
        schema_version=2,
        dataset=dataset,
        archive=PREPARE.SourceArchive(path="source.tar", sha256="unused"),
        sources=(source,),
        episode_plan=episode_plan,
        episodes=(episode,),
    )


def _video_payloads(path: Path) -> tuple[list[str], list[bytes]]:
    schema_names: list[str] = []
    payloads: list[bytes] = []
    with path.open("rb") as stream:
        for schema, _channel, message in make_reader(stream).iter_messages():
            assert schema is not None
            schema_names.append(schema.name)
            decoded = CompressedVideo.FromString(message.data)
            assert decoded.format == "h264"
            payloads.append(bytes(decoded.data))
    return schema_names, payloads


@pytest.mark.parametrize(
    ("fault", "fault_segment_s", "expected_black_frame_pct"),
    [
        ("blackout", (7.0, 10.0), 15.0),
        ("freeze", (7.0, 11.0), 0.0),
    ],
)
def test_egocentric_h264_lands_once_and_faults_survive_transform(
    tmp_path: Path,
    moving_hevc_video: Path,
    fault: str,
    fault_segment_s: tuple[float, float],
    expected_black_frame_pct: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(PREPARE, "EPISODE_IMAGE_WIDTH", 160)
    monkeypatch.setattr(PREPARE, "EPISODE_IMAGE_HEIGHT", 90)
    fault_kind = PREPARE.FaultKind(fault)
    episode = PREPARE.PlannedEpisode(
        episode_id=f"episode_{fault}",
        source_member="source.mp4",
        source_start_s=1.0,
        duration_s=20.0,
        task="factory_task",
        fault=fault_kind,
        fault_segment_s=fault_segment_s,
    )
    landing_path = tmp_path / f"{fault}.mcap"
    canonical_path = tmp_path / f"{fault}.canonical.mcap"

    identity = PREPARE.SourceIdentity(factory_id="factory_002", worker_id="worker_001")
    PREPARE._write_video_episode(
        moving_hevc_video, landing_path, _manifest(episode), episode, 0, identity
    )
    write_canonical_episode(landing_path, canonical_path, TransformConfig())

    landing_schemas, landing_payloads = _video_payloads(landing_path)
    canonical_schemas, canonical_payloads = _video_payloads(canonical_path)
    assert set(landing_schemas) == {"foxglove.CompressedVideo"}
    assert "sensor_msgs/msg/CompressedImage" not in landing_schemas
    assert set(canonical_schemas) == {"foxglove.CompressedVideo"}
    assert canonical_payloads == landing_payloads

    with Episode(canonical_path) as canonical_episode:
        evidence = camera_frame_stats(canonical_episode)
    camera_topic = "/head_camera/compressed"
    assert evidence.measurements[f"{camera_topic}/black_frame_pct"] == pytest.approx(
        expected_black_frame_pct, abs=0.6
    )
    freeze_total_seconds = evidence.measurements[f"{camera_topic}/freeze_total_s"]
    decoded_frame_count = evidence.measurements[f"{camera_topic}/decoded_frame_count"]
    assert isinstance(freeze_total_seconds, float)
    assert freeze_total_seconds >= 2.0
    assert decoded_frame_count == 200


def _write_shard_tar(
    tar_path: Path,
    member_stem: str,
    video_source: Path,
    factory_id: str,
    worker_id: str,
) -> str:
    """One pinned shard tar: a single video plus its sidecar.

    Returns the archive sha256 so the manifest can pin it.
    """
    video_member = f"{member_stem}.mp4"
    sidecar_member = f"{member_stem}.json"
    video_bytes = video_source.read_bytes()
    sidecar = json.dumps(
        {
            "factory_id": factory_id,
            "worker_id": worker_id,
            "video_index": 0,
            "duration_sec": 24.0,
            "width": 160,
            "height": 90,
            "fps": 10.0,
            "size_bytes": len(video_bytes),
            "codec": "h265",
        }
    )
    with tarfile.open(tar_path, "w") as tar:
        video_info = tarfile.TarInfo(video_member)
        video_info.size = len(video_bytes)
        tar.addfile(video_info, io.BytesIO(video_bytes))
        sidecar_info = tarfile.TarInfo(sidecar_member)
        sidecar_info.size = len(sidecar.encode())
        tar.addfile(sidecar_info, io.BytesIO(sidecar.encode()))
    return (
        video_member,
        hashlib.sha256(video_bytes).hexdigest(),
        hashlib.sha256(tar_path.read_bytes()).hexdigest(),
    )


def _manifest_json(
    archive_path: str,
    archive_sha256: str,
    member: str,
    member_sha256: str,
    task: str,
) -> str:
    manifest = {
        "schema_version": 2,
        "dataset": {"repo_id": "e/c", "revision": "abc123", "license": "apache-2.0"},
        "archive": {"path": archive_path, "sha256": archive_sha256},
        "sources": [
            {
                "member": member,
                "sha256": member_sha256,
                "duration_s": 24.0,
                "task": task,
            }
        ],
        "episode_plan": {
            "total_episodes": 1,
            "duration_s": 20.0,
            "first_source_start_s": 1.0,
            "source_stride_s": 0.0,
            "faults": [],
        },
    }
    return json.dumps(manifest, indent=2)


def _episode_provenance(landing_path: Path) -> dict[str, str]:
    with Episode(landing_path) as episode:
        return episode.metadata_records["episode/v1"]


def test_two_shards_coexist_in_one_output_root(tmp_path: Path, moving_hevc_video: Path) -> None:
    """#519: two factories' shards into one output root must coexist. With the
    old hardcoded ids the second prepare silently overwrote the first."""
    source_root = tmp_path / "source"
    output_root = tmp_path / "corpus"
    shard_specs = [
        ("factory002_worker001_00000", "factory_002", "worker_001"),
        ("factory012_worker003_00000", "factory_012", "worker_003"),
    ]
    for index, (stem, factory_id, worker_id) in enumerate(shard_specs):
        tar_path = source_root / "huggingface" / f"shard{index}.tar"
        tar_path.parent.mkdir(parents=True, exist_ok=True)
        member, member_sha, archive_sha = _write_shard_tar(
            tar_path, stem, moving_hevc_video, factory_id, worker_id
        )
        manifest_path = tmp_path / f"manifest-{index}.json"
        manifest_path.write_text(
            _manifest_json(
                f"shard{index}.tar", archive_sha, member, member_sha, f"{factory_id} task"
            ),
            encoding="utf-8",
        )
        PREPARE.prepare_corpus(manifest_path, source_root, output_root)

    landing_paths = sorted((output_root / "landing").glob("*.mcap"))
    assert len(landing_paths) == len({p.name for p in landing_paths}) == 2, landing_paths
    for landing_path in (output_root / "landing").glob("*.mcap"):
        with Episode(landing_path) as episode:
            metadata = episode.metadata_records["episode/v1"]
        expected_operator = {
            "factory002_worker001_00000": "factory_002_worker_001",
            "factory012_worker003_00000": "factory_012_worker_003",
        }[metadata["source_member"].rsplit(".", 1)[0]]
        assert metadata["operator"] == expected_operator, metadata["operator"]
        assert (
            metadata["factory"]
            == expected_operator.split("_")[0] + "_" + expected_operator.split("_")[1]
        )


def test_single_shard_provenance_names_the_real_source(
    tmp_path: Path, moving_hevc_video: Path
) -> None:
    """One shard, one episode: operator and factory come from the sidecar."""
    source_root = tmp_path / "source"
    output_root = tmp_path / "corpus"
    tar_path = source_root / "huggingface" / "shard.tar"
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    member, member_sha, archive_sha = _write_shard_tar(
        tar_path, "factory002_worker001_00000", moving_hevc_video, "factory_002", "worker_001"
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        _manifest_json("shard.tar", archive_sha, member, member_sha, "factory_002 task"),
        encoding="utf-8",
    )

    report = PREPARE.prepare_corpus(manifest_path, source_root, output_root)

    assert len(report) == 1
    with Episode(report[0]) as episode:
        metadata = episode.metadata_records["episode/v1"]
    assert metadata["operator"] == "factory_002_worker_001"
    assert metadata["factory"] == "factory_002"
    assert metadata["source_member"] == member
