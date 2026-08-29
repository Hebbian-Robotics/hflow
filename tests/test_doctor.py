"""The canonical episode format as an executable conformance check."""

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from foxglove_schemas_protobuf.CompressedVideo_pb2 import CompressedVideo
from mcap.writer import Writer as StockWriter
from mcap_protobuf.schema import build_file_descriptor_set
from mcap_ros2.writer import Writer as Ros2Writer

from hflow.cli import main as cli_main
from hflow.doctor import DiagnosticLevel, diagnose
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode
from hflow.transform import write_canonical_episode

ROS2_COMPRESSED_VIDEO_SCHEMA = "\n".join(
    [
        "builtin_interfaces/Time timestamp",
        "string frame_id",
        "uint8[] data",
        "string format",
        "=" * 80,
        "MSG: builtin_interfaces/Time",
        "int32 sec",
        "uint32 nanosec",
    ]
)


def write_chunked_episode(
    path: Path,
    message_specs: list[tuple[str, int, int]],
    groups_by_topic: dict[str, str],
    *,
    chunk_size_bytes: int = 1_000,
) -> None:
    """Write an MCAP whose chunk layout is controlled by the caller.

    The stock ``mcap`` writer finalizes a chunk once the accumulated bytes
    exceed an explicit ``chunk_size``, so varying per-message payload sizes
    places message boundaries exactly where the test needs them. HFlow's own
    grouped writer cannot produce chunks out of time order, so the violation
    fixtures here are built with the stock writer instead.
    """
    with path.open("wb") as stream:
        writer = StockWriter(stream, chunk_size=chunk_size_bytes)
        writer.start(profile="", library="test")
        schema_id = writer.register_schema(name="test/v1", encoding="json", data=b"{}")
        channel_ids: dict[str, int] = {}
        for topic in sorted({topic for topic, _log_time, _size in message_specs}):
            channel_ids[topic] = writer.register_channel(
                topic=topic, message_encoding="json", schema_id=schema_id
            )
        for topic, log_time, payload_size in message_specs:
            writer.add_message(
                channel_ids[topic],
                log_time=log_time,
                data=b"\x00" * payload_size,
                publish_time=log_time,
            )
        provenance = {"schema_version": "1", "pipeline_version": "0123456789ab"}
        provenance.update({f"group/{topic}": group for topic, group in groups_by_topic.items()})
        writer.add_metadata("provenance/v1", provenance)
        writer.finish()


@pytest.fixture(scope="module")
def canonical_episode(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("doctor")
    source = synthesize_episode(root / "source.mcap", SyntheticEpisodeSpec(duration_s=4.0))
    output = root / "episode.canonical.mcap"
    write_canonical_episode(source, output)
    return output


def test_transform_output_is_conforming(canonical_episode: Path) -> None:
    report = diagnose(canonical_episode)
    assert report.conforming, report.summary()
    assert "CONFORMING" in report.summary()


def test_raw_input_recording_is_not_conforming(tmp_path: Path) -> None:
    source = synthesize_episode(tmp_path / "raw.mcap", SyntheticEpisodeSpec(duration_s=2.0))
    report = diagnose(source)
    assert not report.conforming
    codes = {finding.code for finding in report.findings}
    assert "missing-provenance" in codes


def test_nonconforming_video_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "bad_video.mcap"
    with path.open("wb") as stream:
        writer = StockWriter(stream)
        writer.start(profile="", library="test")
        schema_id = writer.register_schema(
            name="foxglove.CompressedVideo",
            encoding="protobuf",
            data=build_file_descriptor_set(CompressedVideo).SerializeToString(),
        )
        channel_id = writer.register_channel(
            topic="/cam", message_encoding="protobuf", schema_id=schema_id
        )
        message = CompressedVideo()
        message.timestamp.FromNanoseconds(10**9)
        message.frame_id = "cam"
        message.data = b"\x00\x00\x00\x01\x41not-aud-delimited"
        message.format = "h264"
        writer.add_message(
            channel_id, log_time=10**9, data=message.SerializeToString(), publish_time=10**9
        )
        writer.finish()
    report = diagnose(path)
    assert not report.conforming
    codes = {finding.code for finding in report.findings}
    assert "video-not-aud-delimited" in codes


def test_nonconforming_ros2_video_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "bad_ros2_video.mcap"
    writer = Ros2Writer(str(path))
    schema = writer.register_msgdef(
        "foxglove_msgs/msg/CompressedVideo", ROS2_COMPRESSED_VIDEO_SCHEMA
    )
    writer.write_message(
        "/cam",
        schema,
        SimpleNamespace(
            timestamp=SimpleNamespace(sec=1, nanosec=0),
            frame_id="cam",
            data=b"\x00\x00\x00\x01\x41not-aud-delimited",
            format="h264",
        ),
        log_time=10**9,
        publish_time=10**9,
    )
    writer.finish()

    report = diagnose(path)

    assert not report.conforming
    codes = {finding.code for finding in report.findings}
    assert "video-not-aud-delimited" in codes


def test_group_chunk_sequence_descending_is_reported_once(tmp_path: Path) -> None:
    # One group, three chunks with start times [10 s, 0 s, 5 s]: two descents
    # within the same group. Each channel's own messages still ascend, so this
    # is only a chunk-layout deviation, not a per-topic time-order error.
    path = tmp_path / "group_descends.mcap"
    write_chunked_episode(
        path,
        [
            ("/alpha", 10 * 10**9, 200),
            ("/alpha", 11 * 10**9, 900),
            ("/beta", 0 * 10**9, 200),
            ("/beta", 1 * 10**9, 900),
        ],
        {"/alpha": "cameras", "/beta": "cameras"},
    )

    report = diagnose(path)

    assert report.conforming  # a layout deviation is a warning, not an error
    findings = [
        finding for finding in report.findings if finding.code == "chunk-group-out-of-time-order"
    ]
    assert len(findings) == 1, report.summary()
    finding = findings[0]
    assert finding.level is DiagnosticLevel.WARNING
    assert "cameras" in finding.message
    assert "chunk" in finding.message


def test_group_chunk_sequence_ascending_is_not_reported(tmp_path: Path) -> None:
    path = tmp_path / "group_ascends.mcap"
    write_chunked_episode(
        path,
        [
            ("/alpha", 0 * 10**9, 200),
            ("/alpha", 1 * 10**9, 900),
            ("/beta", 2 * 10**9, 200),
            ("/beta", 3 * 10**9, 900),
        ],
        {"/alpha": "cameras", "/beta": "cameras"},
    )

    report = diagnose(path)

    codes = {finding.code for finding in report.findings}
    assert "chunk-group-out-of-time-order" not in codes


def test_group_chunks_may_interleave_between_groups(tmp_path: Path) -> None:
    # The state group's chunks surround a cameras chunk that starts earlier.
    # Out of global order, but each group's own sequence is ascending, so the
    # interleaving itself is not reported.
    path = tmp_path / "groups_interleave.mcap"
    write_chunked_episode(
        path,
        [
            ("/beta", 2 * 10**9, 5_000),
            ("/alpha", 0 * 10**9, 5_000),
            ("/beta", 4 * 10**9, 5_000),
        ],
        {"/alpha": "cameras", "/beta": "state"},
        chunk_size_bytes=1_024,
    )

    report = diagnose(path)

    codes = {finding.code for finding in report.findings}
    assert "chunk-group-out-of-time-order" not in codes


def test_no_group_map_skips_the_time_order_check(tmp_path: Path) -> None:
    path = tmp_path / "no_group_map.mcap"
    write_chunked_episode(
        path,
        [
            ("/alpha", 10 * 10**9, 200),
            ("/alpha", 11 * 10**9, 900),
            ("/beta", 0 * 10**9, 200),
            ("/beta", 1 * 10**9, 900),
        ],
        {},
    )

    report = diagnose(path)

    codes = {finding.code for finding in report.findings}
    assert "chunk-group-out-of-time-order" not in codes


def test_not_an_mcap_file(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.mcap"
    bogus.write_bytes(b"definitely not mcap")
    report = diagnose(bogus)
    assert not report.conforming
    assert report.findings[0].code == "unreadable"
    assert report.findings[0].level is DiagnosticLevel.ERROR


def test_cli_doctor_exit_codes(
    canonical_episode: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli_main(["doctor", str(canonical_episode)]) == 0
    assert "CONFORMING" in capsys.readouterr().out

    bogus = tmp_path / "bogus.mcap"
    bogus.write_bytes(b"nope")
    assert cli_main(["doctor", str(bogus)]) == 1
    assert "NOT CONFORMING" in capsys.readouterr().out

    assert cli_main(["doctor", str(canonical_episode), str(canonical_episode)]) == 0
    both_conforming = capsys.readouterr().out
    assert both_conforming.count(str(canonical_episode)) == 2
    assert "NOT CONFORMING" not in both_conforming

    assert cli_main(["doctor", str(canonical_episode), str(bogus)]) == 1
    mixed = capsys.readouterr().out
    assert str(canonical_episode) in mixed
    assert str(bogus) in mixed


def test_cli_logs_library_warning_to_stderr(
    canonical_episode: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = logging.getLogger("hflow.test")

    def diagnose_with_warning(path: Path) -> object:
        logger.warning("test library warning")
        return diagnose(path)

    monkeypatch.setattr("hflow.cli.diagnose", diagnose_with_warning)

    root_logger = logging.getLogger()
    handlers = root_logger.handlers.copy()
    root_logger.handlers.clear()

    try:
        assert cli_main(["doctor", str(canonical_episode)]) == 0
    finally:
        root_logger.handlers[:] = handlers

    captured = capsys.readouterr()
    assert "WARNING hflow.test: test library warning" in captured.err


def test_cli_doctor_missing_file_prints_one_line(capsys: pytest.CaptureFixture[str]) -> None:
    missing = "C:/definitely/not/here.mcap"
    assert cli_main(["doctor", missing]) == 2
    captured = capsys.readouterr()
    assert "doctor:" in captured.out
    assert "[error] unreadable:" in captured.out
    assert "No such file or directory" in captured.out
    assert "Traceback" not in captured.out
    assert "Traceback" not in captured.err


def test_cli_doctor_continues_past_unreadable_file(
    canonical_episode: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = "C:/definitely/not/here.mcap"
    assert cli_main(["doctor", str(canonical_episode), missing, str(canonical_episode)]) == 1
    out = capsys.readouterr().out
    first = out.index(str(canonical_episode))
    missing_at = out.index(missing)
    last = out.rindex(str(canonical_episode))
    assert first < missing_at < last
    assert out.count("[error] unreadable:") == 1
    assert "NOT CONFORMING" in out
    assert "Traceback" not in out


def test_cli_doctor_all_unreadable_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    missing = "C:/definitely/not/here.mcap"
    assert cli_main(["doctor", missing, missing]) == 2
    captured = capsys.readouterr()
    assert captured.out.count("[error] unreadable:") == 2
    assert "Traceback" not in captured.out


def test_cli_curate_bad_catalog_prints_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad_catalog = tmp_path / "not-a-catalog"
    bad_catalog.mkdir()
    assert (
        cli_main(
            [
                "curate",
                "SELECT 1",
                "--catalog",
                str(bad_catalog),
                "--output",
                str(tmp_path / "m.parquet"),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "curate:" in captured.err
    assert "Traceback" not in captured.err
