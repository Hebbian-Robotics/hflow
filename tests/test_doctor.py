"""The canonical episode format as an executable conformance check."""

import logging
from pathlib import Path

import pytest
from foxglove_schemas_protobuf.CompressedVideo_pb2 import CompressedVideo
from mcap.writer import Writer as StockWriter
from mcap_protobuf.schema import build_file_descriptor_set

from hflow.cli import main as cli_main
from hflow.doctor import DiagnosticLevel, diagnose
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode
from hflow.transform import write_canonical_episode


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
