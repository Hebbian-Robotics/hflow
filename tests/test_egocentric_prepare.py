"""The egocentric converter lands H.264 directly and preserves its planted faults."""

import importlib.util
import subprocess
import sys
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

    PREPARE._write_video_episode(moving_hevc_video, landing_path, _manifest(episode), episode, 0)
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
