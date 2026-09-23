"""Video import publishes complete, correctly sampled source episodes."""

import asyncio
import hashlib
import json
import re
import subprocess
import tracemalloc
from dataclasses import replace
from pathlib import Path

import av
import numpy as np
import pytest
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory
from media_test_helpers import render_lavfi, run_ffmpeg, run_ffprobe

import hflow
from hflow.format import GopPreset
from hflow.importers.video import VideoImportConfig, import_video_episode
from hflow.media import VideoLimits
from hflow.transform import TransformConfig, write_canonical_episode


@pytest.fixture
def source_video(tmp_path: Path) -> Path:
    return render_lavfi(
        tmp_path / "source.mp4",
        "color=red:size=160x90:rate=4:duration=1",
        "color=blue:size=160x90:rate=4:duration=1",
        output_arguments=(
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ),
    )


def _bgr_frame_from_h264(access_unit: bytes, decoder: av.CodecContext) -> np.ndarray:
    assert isinstance(decoder, av.VideoCodecContext)
    frames = list(decoder.decode(av.Packet(access_unit)))
    assert len(frames) == 1, "expected exactly one decoded H.264 picture per access unit"
    return frames[0].to_ndarray(format="bgr24")


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
    decoder = av.CodecContext.create("h264", "r")
    for sample_index, (schema, channel, message, decoded) in enumerate(messages):
        assert schema is not None
        assert schema.name == "foxglove.CompressedVideo"
        assert channel.topic == "/head/compressed"
        expected_timestamp = config.start_time_ns + sample_index * 250_000_000
        assert message.log_time == message.publish_time == expected_timestamp
        assert (
            decoded.timestamp.seconds * 1_000_000_000 + decoded.timestamp.nanos
            == expected_timestamp
        )
        assert decoded.frame_id == "head"
        assert decoded.format == "h264"
        pixels = _bgr_frame_from_h264(decoded.data, decoder)
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
    async def sampled_images(episode: hflow.Episode) -> hflow.CheckResult:
        assert episode.metadata_records["episode/v1"] == dict(config.metadata)
        assert episode.metadata_records["video_import/v1"] == metadata["video_import/v1"]
        assert episode.cameras == ["/head/compressed"]
        return hflow.CheckResult(measurements={"frames": len(episode.frames(fps=4))})

    report = asyncio.run(
        app.process(output_path, record=False, stages={hflow.Stage.SYNC, hflow.Stage.META})
    )
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
    assert _schema is not None and _schema.name == "foxglove.CompressedVideo"
    assert decoded.format == "h264"
    decoder = av.CodecContext.create("h264", "r")
    pixels = _bgr_frame_from_h264(decoded.data, decoder)
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
        {"source_start_s": -1},
        {"image_hz": 0},
        {"image_height": True},
        {"start_time_ns": -1},
        {"start_time_ns": True},
        # Inside the field's range but the final frame's timestamp overflows.
        {"start_time_ns": (1 << 64) - 1},
        {"camera_name": ""},
        {"metadata": (("task", "one"), ("task", "two"))},
        {"maximum_encoded_bytes": 0},
        {"maximum_encoded_bytes": True},
        {"maximum_encoded_bytes": 1.5},
    ],
)
def test_invalid_import_configuration_is_rejected(config: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(VideoImportConfig(duration_s=1), **config)


def test_preparation_distinguishes_rejected_media_from_tool_failures(
    source_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hflow.media as media
    from hflow.importers.video import ImportedVideoEpisode, prepare_video_episode

    output = tmp_path / "episode.mcap"
    unreadable = tmp_path / "invalid.mp4"
    unreadable.write_bytes(b"not a recording")
    assert isinstance(
        prepare_video_episode(unreadable, output, VideoImportConfig(duration_s=1)),
        media.UnreadableVideo,
    )
    assert isinstance(
        prepare_video_episode(
            source_video,
            output,
            VideoImportConfig(duration_s=1),
            limits=media.VideoLimits(maximum_frame_pixels=10),
        ),
        media.UnsupportedVideo,
    )
    assert not output.exists()
    assert isinstance(
        prepare_video_episode(source_video, output, VideoImportConfig(duration_s=1)),
        ImportedVideoEpisode,
    )
    missing_tool = tmp_path / "missing-ffprobe"
    monkeypatch.setattr(media, "ffprobe_path", lambda: missing_tool)
    with pytest.raises(media.MediaToolError):
        prepare_video_episode(
            source_video, tmp_path / "absent.mcap", VideoImportConfig(duration_s=1)
        )
    assert not (tmp_path / "absent.mcap").exists()


@pytest.mark.parametrize(
    "limits",
    [VideoLimits(maximum_frame_pixels=160 * 90), VideoLimits(maximum_frames_per_second=4)],
)
def test_both_import_entrypoints_reject_output_exceeding_limits(
    source_video: Path, tmp_path: Path, limits: VideoLimits
) -> None:
    from hflow.importers.video import prepare_video_episode
    from hflow.media import UnsupportedVideo

    config = VideoImportConfig(duration_s=1)
    output_path = tmp_path / "unsupported.mcap"
    with pytest.raises(ValueError, match="output exceeds supported limits"):
        import_video_episode(source_video, output_path, config, limits=limits)
    assert isinstance(
        prepare_video_episode(source_video, output_path, config, limits=limits), UnsupportedVideo
    )
    assert not output_path.exists()
    assert not tuple(tmp_path.glob(".video-import-*"))


def test_window_preparation_preserves_requested_sampling_and_first_video_stream(
    source_video: Path, tmp_path: Path
) -> None:
    from hflow.media import PreparedVideoWindow, VideoWindow, prepare_video_window

    multiple_streams = tmp_path / "multiple.mp4"
    run_ffmpeg(
        "-v",
        "error",
        "-i",
        str(source_video),
        "-f",
        "lavfi",
        "-i",
        "color=green:size=320x180:rate=4:duration=2",
        "-map",
        "0:v:0",
        "-map",
        "1:v:0",
        "-c:v",
        "libx264",
        str(multiple_streams),
    )
    output = tmp_path / "window.mp4"
    prepared = prepare_video_window(multiple_streams, output, VideoWindow(0.5, 1.0, 4.0))
    assert isinstance(prepared, PreparedVideoWindow)
    assert prepared.properties.width == 160
    assert prepared.properties.height == 90
    assert prepared.properties.duration_seconds == 1
    assert prepared.properties.frames_per_second == 4
    assert prepared.source_window == VideoWindow(0.5, 1.0, 4.0)
    with pytest.raises(FileExistsError):
        prepare_video_window(multiple_streams, output, VideoWindow(0.5, 1, 4))
    assert not tuple(tmp_path.glob(".video-window-*"))


def test_prepared_window_preserves_source_duration_despite_frame_padding(
    source_video: Path, tmp_path: Path
) -> None:
    from hflow.media import PreparedVideoWindow, VideoWindow, prepare_video_window

    window = VideoWindow(0, 0.65, 4)
    prepared = prepare_video_window(source_video, tmp_path / "fractional.mp4", window)
    assert isinstance(prepared, PreparedVideoWindow)
    assert prepared.source_window == window
    assert float(prepared.properties.duration_seconds) > window.duration_seconds


@pytest.mark.parametrize(
    ("start_seconds", "requested_seconds", "covered_seconds"),
    [(1.5, 1, 0.5), (1.75, 10, 0.25)],
)
def test_prepared_source_interval_stops_at_eof(
    source_video: Path,
    tmp_path: Path,
    start_seconds: float,
    requested_seconds: float,
    covered_seconds: float,
) -> None:
    from hflow.media import PreparedVideoWindow, UnreadableVideo, VideoWindow, prepare_video_window

    prepared = prepare_video_window(
        source_video, tmp_path / "last-window.mp4", VideoWindow(start_seconds, requested_seconds, 4)
    )
    assert isinstance(prepared, PreparedVideoWindow)
    assert prepared.source_window == VideoWindow(start_seconds, covered_seconds, 4)
    beyond_source = tmp_path / "beyond-source.mp4"
    assert isinstance(
        prepare_video_window(source_video, beyond_source, VideoWindow(2, 1, 4)), UnreadableVideo
    )
    assert not beyond_source.exists()


def test_tagged_video_duration_is_shared_by_probe_and_import(
    source_video: Path, tmp_path: Path
) -> None:
    from hflow.importers.video import ImportedVideoEpisode, prepare_video_episode

    matroska = tmp_path / "source.mkv"
    run_ffmpeg("-v", "error", "-i", str(source_video), "-c", "copy", str(matroska))
    outcome = prepare_video_episode(
        matroska, tmp_path / "tagged.mcap", VideoImportConfig(duration_s=1, image_hz=4)
    )
    assert isinstance(outcome, ImportedVideoEpisode)
    assert len(hflow.Episode(outcome.path).channel("/camera/compressed")) == 4


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        # Positivity comes from the shared guard; evenness keeps its own message.
        ("image_width", 0, "image_width must be > 0, got 0"),
        ("image_width", 3, "image_width must be an even integer, got 3"),
        ("image_height", -4, "image_height must be > 0, got -4"),
        ("image_height", 3, "image_height must be an even integer, got 3"),
        # The shared guard splits the old blanket message into type vs finiteness.
        ("duration_s", "fast", "duration_s must be an int or float, got str"),
        ("duration_s", float("nan"), "duration_s must be finite, got nan"),
        ("source_start_s", False, "source_start_s must be an int or float, got bool"),
        ("image_hz", float("inf"), "image_hz must be finite, got inf"),
        # The field guard owns the start_time_ns upper-bound refusal.
        ("start_time_ns", 1 << 64, f"start_time_ns must be in [0, {(1 << 64) - 1}], got {1 << 64}"),
    ],
)
def test_a_refused_field_names_itself_and_the_defect(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        replace(VideoImportConfig(duration_s=1), **{field: value})


@pytest.mark.parametrize("duration_s,image_hz", [(1.0, 4.0), (0.1, 1.0)])
def test_direct_model_video_matches_canonical_decoded_pixels(
    source_video: Path, tmp_path: Path, duration_s: float, image_hz: float
) -> None:
    from hflow.importers.video import prepare_model_video

    configuration = VideoImportConfig(
        duration_s=duration_s, image_hz=image_hz, image_width=80, image_height=80
    )
    imported = import_video_episode(source_video, tmp_path / "import.mcap", configuration)
    application = hflow.App("model-parity", data_root=tmp_path / "workspace", default_checks=())
    report = asyncio.run(application.process(imported, record=False, stages={hflow.Stage.SYNC}))
    assert not report.has_errors, report.summary()
    output = tmp_path / "direct.mp4"
    assert prepare_model_video(source_video, output, configuration) == output
    from hflow.video import write_access_units_to_mp4

    with report.canonical_path.open("rb") as canonical_stream:
        canonical_messages = list(
            make_reader(
                canonical_stream, decoder_factories=[DecoderFactory()]
            ).iter_decoded_messages()
        )
        reference_video = write_access_units_to_mp4(
            (decoded.data for _schema, _channel, _message, decoded in canonical_messages),
            fps=image_hz if configuration.frame_count > 1 else 1.0,
            output=tmp_path / "reference.mp4",
        )
        fingerprints = [
            run_ffmpeg(
                "-v",
                "error",
                "-i",
                str(video),
                "-map",
                "0:v:0",
                "-f",
                "framemd5",
                "-",
                timeout_seconds=30,
            )
            for video in (reference_video, output)
        ]
    assert fingerprints[0] == fingerprints[1]
    original_output = output.read_bytes()
    with pytest.raises(FileExistsError):
        prepare_model_video(source_video, output, configuration)
    assert output.read_bytes() == original_output


def test_import_lands_h264_without_jpeg_and_canonical_passthrough_shrinks_ratio(
    moving_video: Path, tmp_path: Path
) -> None:
    """Landing stores in-band H.264; canonical pass-through no longer grows from JPEG."""
    from hflow.transform import write_canonical_episode
    from hflow.video import split_annex_b_stream

    config = VideoImportConfig(
        duration_s=3,
        image_hz=10,
        image_width=320,
        image_height=240,
        camera_name="cam",
    )
    landing_path = tmp_path / "landing.mcap"
    import_video_episode(moving_video, landing_path, config)
    with landing_path.open("rb") as input_stream:
        reader = make_reader(input_stream, decoder_factories=[DecoderFactory()])
        messages = list(reader.iter_decoded_messages())
        metadata = {record.name: record.metadata for record in reader.iter_metadata()}
    assert metadata["video_import/v1"]["landing_format"] == "h264"
    assert metadata["video_import/v1"]["importer_version"] == "2"
    assert len(messages) == 30
    for schema, _channel, _message, decoded in messages:
        assert schema is not None
        assert schema.name == "foxglove.CompressedVideo"
        assert decoded.format == "h264"

        units = split_annex_b_stream(decoded.data)
        assert len(units) == 1

    canonical_path = tmp_path / "canonical.mcap"
    write_canonical_episode(landing_path, canonical_path)
    payloads = _video_payloads(landing_path)
    assert payloads == _video_payloads(canonical_path)
    media_bytes = sum(map(len, payloads))
    assert landing_path.stat().st_size <= media_bytes * 1.15
    landing_bytes = landing_path.stat().st_size
    canonical_bytes = canonical_path.stat().st_size
    # H.264 landing should stay close to the lossless canonical copy's size.
    assert landing_bytes <= canonical_bytes * 1.5


@pytest.fixture
def moving_video(tmp_path: Path) -> Path:
    return render_lavfi(
        tmp_path / "moving.mkv",
        "testsrc2=size=320x240:rate=10:duration=3",
        output_arguments=("-c:v", "ffv1"),
    )


def _video_payloads(path: Path) -> list[bytes]:
    with path.open("rb") as stream:
        return [
            bytes(decoded.data)
            for _schema, _channel, _message, decoded in make_reader(
                stream, decoder_factories=[DecoderFactory()]
            ).iter_decoded_messages()
        ]


def _decoded_yuv(path: Path) -> np.ndarray:
    with av.open(str(path)) as container:
        return np.stack([frame.to_ndarray(format="yuv420p") for frame in container.decode(video=0)])


@pytest.mark.parametrize(
    "settings,gop_frames",
    [
        (TransformConfig(), 10),
        (TransformConfig(crf=0), 10),
        (TransformConfig(crf=40), 10),
        (TransformConfig(gop_preset=GopPreset.WORLD_MODEL), 60),
        (TransformConfig(crf=18, gop_preset=GopPreset.WORLD_MODEL, gop_seconds=0.5), 5),
    ],
)
def test_custom_encoding_matches_independent_moving_video_reference(
    moving_video: Path, tmp_path: Path, settings: TransformConfig, gop_frames: int
) -> None:
    from hflow.importers.video import (
        ImportedVideoEpisode,
        prepare_model_video,
        prepare_video_episode,
    )
    from hflow.video import (
        scan_picture_coding_types,
        split_annex_b_stream,
        write_access_units_to_mp4,
    )

    config = VideoImportConfig(duration_s=3, image_hz=10, image_width=320, image_height=240)
    landing = import_video_episode(
        moving_video, tmp_path / "landing.mcap", config, transform_config=settings
    )
    prepared = prepare_video_episode(
        moving_video, tmp_path / "prepared.mcap", config, transform_config=settings
    )
    assert isinstance(prepared, ImportedVideoEpisode)
    canonical = tmp_path / "canonical.mcap"
    # Grouping is independent of the committed encoding settings.
    stamps = write_canonical_episode(
        landing,
        canonical,
        replace(settings, compression="none", topic_groups={"/camera/compressed": "custom"}),
    )
    assert stamps.ffmpeg_version == "not-used"
    payloads = _video_payloads(landing)
    assert payloads == _video_payloads(canonical) == _video_payloads(prepared.path)
    with landing.open("rb") as stream:
        metadata = {record.name: record.metadata for record in make_reader(stream).iter_metadata()}
    assert "provenance/v1" not in metadata
    assert metadata["episode/v1"] == {}
    assert int(metadata["video_import/v1"]["crf"]) == settings.crf
    assert float(metadata["video_import/v1"]["gop_seconds"]) == gop_frames / 10
    units = split_annex_b_stream(b"".join(payloads))
    assert len(units) == 30
    assert [i for i, unit in enumerate(units) if unit.is_keyframe] == list(range(0, 30, gop_frames))
    assert all(unit.has_parameter_sets for unit in units if unit.is_keyframe)
    decoder = av.CodecContext.create("h264", "r")
    for payload in payloads:
        _bgr_frame_from_h264(payload, decoder)
    assert isinstance(decoder, av.VideoCodecContext)
    assert decoder.decode(None) == []  # No delayed picture/reorder tail.
    scan = scan_picture_coding_types(b"".join(payloads))
    assert scan.picture_count == 30 and scan.b_picture_count == 0
    exported = write_access_units_to_mp4(payloads, fps=10, output=tmp_path / "canonical.mp4")
    direct = tmp_path / "direct.mp4"
    assert prepare_model_video(moving_video, direct, config, transform_config=settings) == direct

    # Independent FFmpeg oracle: the fixture already has the requested rate,
    # dimensions and pixel format, so no importer filter/helper is involved.
    reference = tmp_path / "oracle.mp4"
    run_ffmpeg(
        "-v",
        "error",
        "-i",
        str(moving_video),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(settings.crf),
        "-pix_fmt",
        "yuv420p",
        "-x264-params",
        f"keyint={gop_frames}:min-keyint={gop_frames}:scenecut=0:bframes=0:repeat-headers=1:aud=1",
        str(reference),
    )
    expected = _decoded_yuv(reference)
    assert np.array_equal(_decoded_yuv(exported), expected)
    assert np.array_equal(_decoded_yuv(direct), expected)
    if settings.crf == 0:
        assert np.array_equal(expected, _decoded_yuv(moving_video))
    else:
        assert not np.array_equal(expected, _decoded_yuv(moving_video))


@pytest.mark.parametrize(
    "requested",
    [
        TransformConfig(crf=0),
        TransformConfig(crf=40),
        TransformConfig(gop_preset=GopPreset.WORLD_MODEL),
        TransformConfig(gop_seconds=0.5),
    ],
)
def test_incompatible_import_encoding_requires_explicit_reimport(
    source_video: Path, tmp_path: Path, requested: TransformConfig
) -> None:
    from hflow.transform import SourceNotConforming

    config = VideoImportConfig(duration_s=2, image_hz=4, image_width=160, image_height=90)
    landing = import_video_episode(source_video, tmp_path / "landing.mcap", config)
    output = tmp_path / "canonical.mcap"
    with pytest.raises(SourceNotConforming, match=r"re-import.*transform_config"):
        write_canonical_episode(landing, output, requested)
    assert not output.exists()
    reimported = import_video_episode(
        source_video, tmp_path / "reimported.mcap", config, transform_config=requested
    )
    write_canonical_episode(reimported, output, requested)
    assert _video_payloads(output) == _video_payloads(reimported)


@pytest.mark.parametrize("image_hz", [0.1, 1.0, 10.0, 30.0])
def test_single_frame_packet_and_container_duration_is_one_second(
    source_video: Path, tmp_path: Path, image_hz: float
) -> None:
    from hflow.importers.video import prepare_model_video
    from hflow.video import write_access_units_to_mp4

    config = VideoImportConfig(duration_s=0.01, image_hz=image_hz, image_width=160, image_height=90)
    landing = import_video_episode(source_video, tmp_path / "landing.mcap", config)
    canonical = tmp_path / "canonical.mcap"
    write_canonical_episode(landing, canonical)
    payloads = _video_payloads(canonical)
    assert len(payloads) == 1
    exported = write_access_units_to_mp4(payloads, fps=1, output=tmp_path / "canonical.mp4")
    direct = tmp_path / "direct.mp4"
    assert prepare_model_video(source_video, direct, config) == direct
    for path in (exported, direct):
        probe = json.loads(
            run_ffprobe(
                "-v",
                "error",
                "-show_packets",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
                timeout_seconds=30,
            )
        )
        assert len(probe["packets"]) == 1
        assert float(probe["packets"][0]["duration_time"]) == pytest.approx(1, abs=1e-6)
        assert float(probe["streams"][0]["duration"]) == pytest.approx(1, abs=1e-6)
        assert float(probe["format"]["duration"]) == pytest.approx(1, abs=1e-6)
        assert _decoded_yuv(path).shape[0] == 1


def test_model_frames_match_canonical_episode_at_selected_indices(
    source_video: Path, tmp_path: Path
) -> None:
    from hflow.importers import prepare_model_frames

    config = VideoImportConfig(duration_s=2, image_hz=4, image_width=160, image_height=90)
    selected_indices = (0, 3, 7)
    landing = import_video_episode(source_video, tmp_path / "landing.mcap", config)
    canonical = tmp_path / "canonical.mcap"
    write_canonical_episode(landing, canonical)
    prepared = prepare_model_frames(source_video, tmp_path / "selected", config, selected_indices)
    assert isinstance(prepared, tuple)
    with hflow.Episode(canonical) as episode:
        camera_topic = episode.cameras[0]
        reference = episode.frames_at_indices(camera_topic, frame_indices=list(selected_indices))
        assert [frame.read_bytes() for frame in prepared] == [
            frame.path.read_bytes() for frame in reference
        ]
    assert [path.name for path in prepared] == [
        "frame_000000.jpg",
        "frame_000001.jpg",
        "frame_000002.jpg",
    ]
    assert not tuple(tmp_path.glob(".model-frames-*"))


def test_model_frame_preparation_rejects_invalid_indices_without_output(
    source_video: Path, tmp_path: Path
) -> None:
    from hflow.importers import prepare_model_frames

    config = VideoImportConfig(duration_s=2, image_hz=4)
    output = tmp_path / "selected"
    for indices in ((), (3, 3), (7, 1), (8,)):
        with pytest.raises(ValueError, match="frame indices"):
            prepare_model_frames(source_video, output, config, indices)
        assert not output.exists()


def test_encoded_byte_budget_is_exclusive_and_all_entrypoints_clean_up(
    moving_video: Path, tmp_path: Path
) -> None:
    from hflow.importers.video import prepare_model_video, prepare_video_episode
    from hflow.media import UnsupportedVideo

    config = VideoImportConfig(duration_s=3, image_hz=10, image_width=320, image_height=240)
    landing = import_video_episode(moving_video, tmp_path / "sized.mcap", config)
    encoded_size = sum(map(len, _video_payloads(landing)))
    accepted = import_video_episode(
        moving_video,
        tmp_path / "accepted.mcap",
        replace(config, maximum_encoded_bytes=encoded_size + 1),
    )
    assert _video_payloads(accepted) == _video_payloads(landing)
    for limit in (encoded_size, 1024):
        bounded = replace(config, maximum_encoded_bytes=limit)
        output = tmp_path / "rejected"
        with pytest.raises(ValueError, match="maximum_encoded_bytes"):
            import_video_episode(moving_video, output, bounded)
        assert isinstance(prepare_video_episode(moving_video, output, bounded), UnsupportedVideo)
        assert isinstance(prepare_model_video(moving_video, output, bounded), UnsupportedVideo)
        assert not output.exists()
        assert not tuple(tmp_path.glob(".video-import-*"))
        assert not tuple(tmp_path.glob(".model-video-*"))


def test_oversized_encoded_output_is_rejected_before_memory_scales_with_file(
    source_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hflow.importers.video as importer
    from hflow.media import UnsupportedVideo

    # A process-boundary double simulates a tool ignoring -fs. Sparse files
    # exercise large size rejection without an expensive encode or allocation.
    encoded_size = 8 * 1024 * 1024

    def oversized_encode(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        with Path(command[-1]).open("wb") as stream:
            stream.truncate(encoded_size)
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(importer, "run_media_command", oversized_encode)
    config = VideoImportConfig(duration_s=1, maximum_encoded_bytes=1024 * 1024)
    peaks = []
    for encoded_size in (8 * 1024 * 1024, 256 * 1024 * 1024):
        tracemalloc.start()
        try:
            outcome = importer.prepare_video_episode(
                source_video, tmp_path / "rejected.mcap", config
            )
            _, peak = tracemalloc.get_traced_memory()
            peaks.append(peak)
        finally:
            tracemalloc.stop()
        assert isinstance(outcome, UnsupportedVideo), f"accepted {encoded_size} encoded bytes"
        assert not (tmp_path / "rejected.mcap").exists()
        assert not tuple(tmp_path.glob(".video-import-*"))
    assert max(peaks) < 4 * 1024 * 1024
