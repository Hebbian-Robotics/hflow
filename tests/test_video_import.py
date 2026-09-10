"""Video import publishes complete, correctly sampled source episodes."""

import hashlib
import subprocess
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

import hflow
from hflow.ffmpeg import ffmpeg_path
from hflow.importers.video import VideoImportConfig, import_video_episode


@pytest.fixture
def source_video(tmp_path: Path) -> Path:
    source_path = tmp_path / "source.mp4"
    subprocess.run(
        [
            str(ffmpeg_path()),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=red:size=160x90:rate=4:duration=1",
            "-f",
            "lavfi",
            "-i",
            "color=blue:size=160x90:rate=4:duration=1",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(source_path),
        ],
        capture_output=True,
        check=True,
    )
    return source_path


def test_imported_excerpt_preserves_content_time_and_known_metadata(
    source_video: Path, tmp_path: Path
) -> None:
    config = VideoImportConfig(
        source_start_s=0.5,
        duration_s=1,
        image_hz=4,
        image_width=80,
        image_height=80,
        camera_name="head",
        start_time_ns=1_234_567_890,
        metadata=(("task", "inspection"), ("source_dataset", "example")),
    )
    output_path = tmp_path / "imported.mcap"
    assert import_video_episode(source_video, output_path, config) == output_path
    with output_path.open("rb") as input_stream:
        reader = make_reader(input_stream, decoder_factories=[DecoderFactory()])
        messages = list(reader.iter_decoded_messages())
        metadata = {record.name: record.metadata for record in reader.iter_metadata()}

    assert len(messages) == 4
    for sample_index, (schema, channel, message, decoded) in enumerate(messages):
        assert schema is not None
        assert schema.name == "foxglove.CompressedImage"
        assert channel.topic == "/head/compressed"
        expected_timestamp = config.start_time_ns + sample_index * 250_000_000
        assert message.log_time == message.publish_time == expected_timestamp
        assert (
            decoded.timestamp.seconds * 1_000_000_000 + decoded.timestamp.nanos
            == expected_timestamp
        )
        assert decoded.frame_id == "head"
        pixels = cv2.imdecode(np.frombuffer(decoded.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert pixels is not None
        assert pixels.shape == (80, 80, 3)
        assert pixels[0].max() < 20  # Letterboxing, not stretched source content.
        expected_channel = 2 if sample_index < 2 else 0  # OpenCV uses BGR.
        assert pixels[40, 40, expected_channel] > 230
        assert np.delete(pixels[40, 40], expected_channel).max() < 20
    assert metadata["episode/v1"] == dict(config.metadata)
    assert (
        metadata["video_import/v1"]["source_sha256"]
        == hashlib.sha256(source_video.read_bytes()).hexdigest()
    )
    assert metadata["video_import/v1"]["frame_count"] == "4"
    assert metadata["video_import/v1"]["source_start_s"] == "0.5"

    app = hflow.App("video-import", data_root=tmp_path / "worker", default_checks=())

    @app.check(version="1")
    def sampled_images(episode: hflow.Episode) -> hflow.CheckResult:
        assert episode.metadata_records["episode/v1"] == dict(config.metadata)
        assert episode.metadata_records["video_import/v1"] == metadata["video_import/v1"]
        assert episode.cameras == ["/head/compressed"]
        return hflow.CheckResult(measurements={"frames": len(episode.frames(fps=4))})

    report = app.process(output_path, record=False, stages={hflow.Stage.SYNC, hflow.Stage.META})
    assert not report.has_errors, report.summary()
    result = report.check("sampled_images").result
    assert result is not None
    assert result.measurements == {"frames": 4}
    assert not tuple(tmp_path.glob(".video-import-*"))


@pytest.mark.parametrize(
    ("duration_s", "image_hz", "expected_count"),
    [(0.6, 2.5, 2), (0.1, 0.1, 1), (0.14, 100.0, 14)],
)
def test_sampling_includes_every_grid_point_before_excerpt_end(
    source_video: Path,
    tmp_path: Path,
    duration_s: float,
    image_hz: float,
    expected_count: int,
) -> None:
    output_path = import_video_episode(
        source_video,
        tmp_path / "fractional.mcap",
        VideoImportConfig(
            duration_s=duration_s, image_hz=image_hz, image_width=80, image_height=46
        ),
    )
    with output_path.open("rb") as input_stream:
        reader = make_reader(input_stream)
        messages = list(reader.iter_messages())
        metadata = {record.name: record.metadata for record in reader.iter_metadata()}
    assert metadata["episode/v1"] == {}
    assert [message.log_time for _schema, _channel, message in messages] == [
        round(sample_index * 1_000_000_000 / image_hz) for sample_index in range(expected_count)
    ]


def test_existing_paths_are_never_overwritten(source_video: Path, tmp_path: Path) -> None:
    existing_path = tmp_path / "existing.mcap"
    existing_path.write_bytes(b"existing episode")
    missing_target = tmp_path / "missing.mcap"
    symbolic_path = tmp_path / "symbolic.mcap"
    symbolic_path.symlink_to(missing_target)
    source_bytes = source_video.read_bytes()
    for output_path in (existing_path, source_video, symbolic_path):
        with pytest.raises(FileExistsError):
            import_video_episode(source_video, output_path, VideoImportConfig(duration_s=1))
    assert existing_path.read_bytes() == b"existing episode"
    assert source_video.read_bytes() == source_bytes
    assert symbolic_path.is_symlink()
    assert not missing_target.exists()
    assert not tuple(tmp_path.glob(".video-import-*"))


@pytest.mark.parametrize(
    ("source_start_s", "duration_s", "expected_color_channel"),
    [(0.9, 0.05, 2), (1.9, 0.1, 0)],
)
def test_subframe_excerpts_sample_the_frame_covering_their_start(
    source_video: Path,
    tmp_path: Path,
    source_start_s: float,
    duration_s: float,
    expected_color_channel: int,
) -> None:
    output_path = import_video_episode(
        source_video,
        tmp_path / "subframe.mcap",
        VideoImportConfig(
            source_start_s=source_start_s,
            duration_s=duration_s,
            image_hz=1,
            image_width=160,
            image_height=90,
        ),
    )
    with output_path.open("rb") as input_stream:
        reader = make_reader(input_stream, decoder_factories=[DecoderFactory()])
        messages = list(reader.iter_decoded_messages())
    assert len(messages) == 1
    _schema, _channel, message, decoded = messages[0]
    assert message.log_time == 0
    pixels = cv2.imdecode(np.frombuffer(decoded.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert pixels is not None
    assert pixels[45, 80, expected_color_channel] > 230
    assert np.delete(pixels[45, 80], expected_color_channel).max() < 20


def test_invalid_sources_and_incomplete_excerpts_publish_nothing(
    source_video: Path, tmp_path: Path
) -> None:
    output_path = tmp_path / "unpublished.mcap"
    config = VideoImportConfig(duration_s=1)
    with pytest.raises(FileNotFoundError):
        import_video_episode(tmp_path / "absent.mp4", output_path, config)
    with pytest.raises(ValueError, match="extends past"):
        import_video_episode(source_video, output_path, replace(config, source_start_s=1.1))
    broken_source = tmp_path / "broken.mp4"
    source_bytes = source_video.read_bytes()
    broken_source.write_bytes(source_bytes[: int(len(source_bytes) * 0.85)])
    with pytest.raises(RuntimeError, match=r"source video|video samples"):
        import_video_episode(broken_source, output_path, config)
    assert not output_path.exists()
    assert not tuple(tmp_path.glob(".video-import-*"))


@pytest.mark.parametrize(
    "config",
    [
        {"duration_s": 0},
        {"duration_s": float("nan")},
        {"source_start_s": -1},
        {"image_hz": 0},
        {"image_hz": float("inf")},
        {"image_width": 3},
        {"image_height": True},
        {"start_time_ns": -1},
        {"start_time_ns": (1 << 64) - 1},
        {"camera_name": ""},
        {"metadata": (("task", "one"), ("task", "two"))},
    ],
)
def test_invalid_import_configuration_is_rejected(config: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(VideoImportConfig(duration_s=1), **config)
