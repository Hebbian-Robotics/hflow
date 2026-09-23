"""The egocentric converter lands H.264 directly and preserves its planted faults."""

import asyncio
import hashlib
import importlib.util
import io
import json
import re
import subprocess
import sys
import tarfile
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

import pytest
from foxglove_schemas_protobuf.CompressedVideo_pb2 import CompressedVideo
from mcap.reader import make_reader

from hflow import Episode, TransformConfig, write_canonical_episode
from hflow.checks import camera_frame_stats
from hflow.ffmpeg import ffmpeg_path


def _load_egocentric_example_module(file_name: str, module_name: str) -> ModuleType:
    module_path = Path(__file__).parents[1] / "examples" / "egocentric" / file_name
    module_spec = importlib.util.spec_from_file_location(module_name, module_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"could not load {module_path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


PREPARE = _load_egocentric_example_module("prepare.py", "egocentric_prepare")
CONVERT = _load_egocentric_example_module("convert.py", "egocentric_convert")


def _write_tar(tar_path: Path, members: Mapping[str, bytes]) -> None:
    """Write ``members`` into a plain tar, in insertion order."""
    with tarfile.open(tar_path, "w") as tar:
        for member_name, member_bytes in members.items():
            member_info = tarfile.TarInfo(member_name)
            member_info.size = len(member_bytes)
            tar.addfile(member_info, io.BytesIO(member_bytes))


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
        evidence = asyncio.run(camera_frame_stats(canonical_episode))
    camera_topic = "/head_camera/compressed"
    assert evidence.measurements[f"{camera_topic}/black_frame_pct"] == pytest.approx(
        expected_black_frame_pct, abs=0.6
    )
    freeze_total_seconds = evidence.measurements[f"{camera_topic}/freeze_total_s"]
    decoded_frame_count = evidence.measurements[f"{camera_topic}/decoded_frame_count"]
    assert isinstance(freeze_total_seconds, float)
    assert freeze_total_seconds >= 2.0
    assert decoded_frame_count == 200


def _exactly(message: str) -> str:
    """A ``match=`` pattern pinning the whole message, metacharacters and all."""
    return rf"^{re.escape(message)}$"


def _write_shard_tar(
    tar_path: Path,
    member_stem: str,
    video_source: Path,
    factory_id: str,
    worker_id: str,
    *,
    sidecar_fields: Mapping[str, object] | None = None,
    include_sidecar: bool = True,
    intrinsics_fields: Mapping[str, object] | None = None,
) -> tuple[str, str, str]:
    """One pinned shard tar: a single video plus its sidecar.

    Returns the archive sha256 so the manifest can pin it.
    """
    video_member = f"{member_stem}.mp4"
    sidecar_member = f"{member_stem}.json"
    video_bytes = video_source.read_bytes()
    if sidecar_fields is None:
        sidecar_fields = {
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
    sidecar = json.dumps(sidecar_fields)
    members = {video_member: video_bytes}
    if include_sidecar:
        members[sidecar_member] = sidecar.encode()
    if intrinsics_fields is not None:
        members["intrinsics.json"] = json.dumps(intrinsics_fields).encode("utf-8")
    _write_tar(tar_path, members)
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


def test_same_member_stem_from_two_shards_never_collides(
    tmp_path: Path, moving_hevc_video: Path
) -> None:
    """The digest in the episode id is load-bearing: source basenames are not
    identities, so two different shards whose members share one stem must land
    as two episodes instead of the second overwriting the first."""
    source_root = tmp_path / "source"
    output_root = tmp_path / "corpus"
    shared_stem = "factory002_worker001_00000"
    shard_factories = ["factory_002", "factory_012"]
    for index, factory_id in enumerate(shard_factories):
        tar_path = source_root / "huggingface" / f"shard{index}.tar"
        tar_path.parent.mkdir(parents=True, exist_ok=True)
        member, member_sha, archive_sha = _write_shard_tar(
            tar_path, shared_stem, moving_hevc_video, factory_id, "worker_001"
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
    assert len(landing_paths) == 2, [path.name for path in landing_paths]
    assert len({path.name for path in landing_paths}) == len(landing_paths)
    observed = {
        landing_path.name: _episode_provenance(landing_path)["factory"]
        for landing_path in landing_paths
    }
    assert set(observed.values()) == set(shard_factories), observed


@pytest.mark.parametrize(
    ("sidecar_case", "expected_message"),
    [
        (
            "absent",
            "missing sidecar 'factory002_worker001_00000.json' for source video "
            "'factory002_worker001_00000.mp4' in the source archive",
        ),
        (
            "empty_worker_id",
            "sidecar 'factory002_worker001_00000.json' is missing a usable 'worker_id'",
        ),
    ],
)
def test_unusable_sidecar_refuses_the_source(
    tmp_path: Path,
    moving_hevc_video: Path,
    sidecar_case: str,
    expected_message: str,
) -> None:
    """A missing or malformed sidecar must fail the prepare loudly, naming the
    member and the unusable field, instead of stamping false provenance."""
    source_root = tmp_path / "source"
    output_root = tmp_path / "corpus"
    tar_path = source_root / "huggingface" / "shard.tar"
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    stem = "factory002_worker001_00000"
    if sidecar_case == "absent":
        member, member_sha, archive_sha = _write_shard_tar(
            tar_path,
            stem,
            moving_hevc_video,
            "factory_002",
            "worker_001",
            include_sidecar=False,
        )
    else:
        member, member_sha, archive_sha = _write_shard_tar(
            tar_path,
            stem,
            moving_hevc_video,
            "factory_002",
            "worker_001",
            sidecar_fields={"factory_id": "factory_002", "worker_id": ""},
        )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        _manifest_json("shard.tar", archive_sha, member, member_sha, "factory_002 task"),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match=_exactly(expected_message)):
        PREPARE.prepare_corpus(manifest_path, source_root, output_root)
    assert list((output_root / "landing").glob("*.mcap")) == []


def test_sidecar_fields_map_to_episode_metadata_and_intrinsics_attached(
    tmp_path: Path, moving_hevc_video: Path
) -> None:
    """#584: Sidecars map cleanly onto episode/v1 metadata (factory, worker,
    duration, fps, codec) and intrinsics.json maps onto a calibration attachment."""
    source_root = tmp_path / "source"
    output_root = tmp_path / "corpus"
    tar_path = source_root / "huggingface" / "shard.tar"
    tar_path.parent.mkdir(parents=True, exist_ok=True)

    intrinsics = {"fx": 525.0, "fy": 525.0, "cx": 320.0, "cy": 180.0, "distortion": [0.0, 0.0]}
    sidecar_data = {
        "factory_id": "factory_051",
        "worker_id": "worker_007",
        "video_index": 0,
        "duration_sec": 24.0,
        "width": 160,
        "height": 90,
        "fps": 10.0,
        "codec": "h265",
    }
    member, member_sha, archive_sha = _write_shard_tar(
        tar_path,
        "factory051_worker007_00000",
        moving_hevc_video,
        "factory_051",
        "worker_007",
        sidecar_fields=sidecar_data,
        intrinsics_fields=intrinsics,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        _manifest_json("shard.tar", archive_sha, member, member_sha, "component_sorting"),
        encoding="utf-8",
    )

    report = PREPARE.prepare_corpus(manifest_path, source_root, output_root)
    assert len(report) == 1
    landing_path = report[0]

    # Verify landing MCAP
    with Episode(landing_path) as episode:
        metadata = episode.metadata_records["episode/v1"]
        assert metadata["factory"] == "factory_051"
        assert metadata["worker"] == "worker_007"
        assert metadata["operator"] == "factory_051_worker_007"
        assert metadata["duration"] == "24"
        assert metadata["fps"] == "10"
        assert metadata["codec"] == "h265"
        assert metadata["task"] == "component_sorting"
        assert metadata["source_member"] == member

        attachments = episode.attachments
        assert len(attachments) == 1
        assert attachments[0].name == "intrinsics.json"
        assert attachments[0].media_type == "application/json"
        assert json.loads(attachments[0].data.decode("utf-8")) == intrinsics

    # Verify canonical MCAP preserves metadata and attachment
    canonical_path = tmp_path / "canonical.mcap"
    write_canonical_episode(landing_path, canonical_path, TransformConfig())

    with Episode(canonical_path) as canonical_episode:
        canonical_metadata = canonical_episode.metadata_records["episode/v1"]
        assert canonical_metadata["factory"] == "factory_051"
        assert canonical_metadata["worker"] == "worker_007"
        assert canonical_metadata["operator"] == "factory_051_worker_007"
        assert canonical_metadata["duration"] == "24"
        assert canonical_metadata["fps"] == "10"
        assert canonical_metadata["codec"] == "h265"

        canonical_attachments = canonical_episode.attachments
        assert len(canonical_attachments) == 1
        assert canonical_attachments[0].name == "intrinsics.json"
        assert json.loads(canonical_attachments[0].data.decode("utf-8")) == intrinsics


def test_convert_webdataset_tar_converts_clips_with_sidecars_and_intrinsics(
    tmp_path: Path, moving_hevc_video: Path
) -> None:
    """#584: Worked converter example reads a WebDataset tar, maps sidecars
    onto episode/v1, attaches intrinsics.json, and writes canonical episodes."""
    tar_path = tmp_path / "corpus_shard.tar"
    output_dir = tmp_path / "converted"

    video_bytes = moving_hevc_video.read_bytes()
    intrinsics = {"fx": 450.0, "fy": 450.0, "cx": 228.0, "cy": 128.0}
    intrinsics_json = json.dumps(intrinsics).encode("utf-8")

    clips = [
        ("factory010_worker002_00000", "factory_010", "worker_002", "material_handling"),
        ("factory010_worker002_00001", "factory_010", "worker_002", "tool_setup"),
    ]

    members = {"intrinsics.json": intrinsics_json}
    for stem, factory_id, worker_id, task in clips:
        members[f"{stem}.mp4"] = video_bytes
        sidecar_dict = {
            "factory_id": factory_id,
            "worker_id": worker_id,
            "task": task,
            "duration_sec": 24.0,
            "fps": 10.0,
            "codec": "h265",
        }
        members[f"{stem}.json"] = json.dumps(sidecar_dict).encode("utf-8")
    _write_tar(tar_path, members)

    results = CONVERT.convert_webdataset_tar(
        tar_path=tar_path,
        output_dir=output_dir,
        canonical=True,
        target_fps=10.0,
        max_duration_s=2.0,
    )

    assert len(results) == 2
    for path, (stem, factory_id, worker_id, task) in zip(results, clips, strict=True):
        assert path.is_file()
        assert path.name == f"{stem}.canonical.mcap"

        with Episode(path) as episode:
            metadata = episode.metadata_records["episode/v1"]
            assert metadata["factory"] == factory_id
            assert metadata["worker"] == worker_id
            assert metadata["operator"] == f"{factory_id}_{worker_id}"
            assert metadata["task"] == task
            assert metadata["duration"] == "24"
            assert metadata["fps"] == "10"
            assert metadata["codec"] == "h265"

            attachments = episode.attachments
            assert len(attachments) == 1
            assert attachments[0].name == "intrinsics.json"
            assert json.loads(attachments[0].data.decode("utf-8")) == intrinsics
            assert len(episode.cameras) == 1
            assert episode.cameras[0] == "/head_camera/compressed"


def test_convert_webdataset_tar_missing_sidecar_fails(
    tmp_path: Path, moving_hevc_video: Path
) -> None:
    """Missing sidecar in WebDataset tar fails with descriptive error."""
    tar_path = tmp_path / "broken.tar"
    output_dir = tmp_path / "converted"
    _write_tar(tar_path, {"clip.mp4": moving_hevc_video.read_bytes()})

    with pytest.raises(
        RuntimeError, match=_exactly("missing sidecar for source video 'clip.mp4' in broken.tar")
    ):
        CONVERT.convert_webdataset_tar(tar_path, output_dir)


def test_convert_webdataset_tar_preserves_distinct_operator(
    tmp_path: Path, moving_hevc_video: Path
) -> None:
    """When sidecar specifies an operator, it is preserved instead of generated."""
    tar_path = tmp_path / "operator_test.tar"
    output_dir = tmp_path / "converted"
    video_bytes = moving_hevc_video.read_bytes()

    sidecar_dict = {
        "factory_id": "factory_010",
        "worker_id": "worker_002",
        "operator": "lead_operator_99",
        "duration_sec": 24.0,
        "fps": 10.0,
        "codec": "h265",
    }

    _write_tar(
        tar_path,
        {"clip.mp4": video_bytes, "clip.json": json.dumps(sidecar_dict).encode("utf-8")},
    )

    results = CONVERT.convert_webdataset_tar(
        tar_path=tar_path,
        output_dir=output_dir,
        canonical=False,
        max_duration_s=1.0,
    )
    assert len(results) == 1
    with Episode(results[0]) as episode:
        metadata = episode.metadata_records["episode/v1"]
        assert metadata["operator"] == "lead_operator_99"


def test_convert_webdataset_tar_staged_validation_leaves_no_partial_mcap(
    tmp_path: Path, moving_hevc_video: Path
) -> None:
    """If a subsequent clip is missing its sidecar, pre-validation fails before any MCAP is written."""
    tar_path = tmp_path / "partial_test.tar"
    output_dir = tmp_path / "converted"
    output_dir.mkdir(parents=True, exist_ok=True)
    video_bytes = moving_hevc_video.read_bytes()

    _write_tar(
        tar_path,
        {
            # First clip has valid sidecar
            "clip1.mp4": video_bytes,
            "clip1.json": json.dumps({"factory_id": "f1", "worker_id": "w1"}).encode("utf-8"),
            # Second clip is missing sidecar
            "clip2.mp4": video_bytes,
        },
    )

    with pytest.raises(
        RuntimeError,
        match=_exactly(f"missing sidecar for source video 'clip2.mp4' in {tar_path.name}"),
    ):
        CONVERT.convert_webdataset_tar(tar_path, output_dir)

    # Assert no partial MCAPs were written to output_dir
    assert list(output_dir.glob("*.mcap")) == []


def test_convert_webdataset_tar_invalid_identity_fails(
    tmp_path: Path, moving_hevc_video: Path
) -> None:
    """Non-string or null factory_id or worker_id fails loudly."""
    tar_path = tmp_path / "invalid_id.tar"
    output_dir = tmp_path / "converted"
    video_bytes = moving_hevc_video.read_bytes()

    _write_tar(
        tar_path,
        {
            "clip1.mp4": video_bytes,
            "clip1.json": json.dumps({"factory_id": "f1", "worker_id": None}).encode("utf-8"),
        },
    )

    with pytest.raises(
        RuntimeError,
        match=_exactly("sidecar 'clip1.json' is missing a usable 'factory_id' or 'worker_id'"),
    ):
        CONVERT.convert_webdataset_tar(tar_path, output_dir)
