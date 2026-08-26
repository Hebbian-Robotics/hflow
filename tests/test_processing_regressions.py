"""Processing, publication, and user-step boundary regressions."""

import functools
import json
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from foxglove_schemas_protobuf.CompressedVideo_pb2 import CompressedVideo
from mcap.exceptions import InvalidMagic
from mcap.reader import make_reader
from mcap.writer import Writer as StockWriter
from mcap_protobuf.schema import build_file_descriptor_set
from mcap_ros2.decoder import DecoderFactory as Ros2DecoderFactory
from mcap_ros2.writer import Writer as Ros2Writer

import hflow
from hflow._grouped_mcap_writer import NO_SCHEMA_ID, GroupedMcapWriter
from hflow.doctor import diagnose
from hflow.format import METADATA_RECORD_EPISODE
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode
from hflow.transform import write_canonical_episode
from hflow.video import estimate_fps_from_log_times

ANNEX_B_START_CODE = b"\x00\x00\x00\x01"
KEYFRAME_ACCESS_UNIT = b"".join(
    ANNEX_B_START_CODE + bytes([nal_type]) + (b"\x80payload" if nal_type == 0x65 else b"payload")
    for nal_type in (0x09, 0x67, 0x68, 0x65)
)
NON_KEYFRAME_ACCESS_UNIT = b"".join(
    ANNEX_B_START_CODE + bytes([nal_type]) + (b"\x80payload" if nal_type == 0x41 else b"payload")
    for nal_type in (0x09, 0x41)
)
KEYFRAME_WITHOUT_AUD = b"".join(
    ANNEX_B_START_CODE + bytes([nal_type]) + (b"\x80payload" if nal_type == 0x65 else b"payload")
    for nal_type in (0x67, 0x68, 0x65)
)
NON_KEYFRAME_WITHOUT_AUD = ANNEX_B_START_CODE + b"\x41\x80payload"
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


def test_estimate_fps_normal_stream() -> None:
    log_times = [i * 100_000_000 for i in range(10)]
    assert estimate_fps_from_log_times(log_times, topic="/cam") == pytest.approx(10.0)


def test_estimate_fps_rejects_duplicate_timestamps() -> None:
    with pytest.raises(ValueError, match="/stereo"):
        estimate_fps_from_log_times([1000, 1000], topic="/stereo")


def test_estimate_fps_rejects_single_timestamp() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        estimate_fps_from_log_times([1000], topic="/cam")


@pytest.fixture()
def state_only_source(tmp_path: Path) -> Path:
    path = tmp_path / "state_only.mcap"
    with path.open("wb") as stream:
        writer = StockWriter(stream)
        writer.start(profile="", library="test")
        channel_id = writer.register_channel(topic="/status", message_encoding="json", schema_id=0)
        for index in range(5):
            payload = json.dumps({"ok": True, "index": index}).encode()
            writer.add_message(
                channel_id, log_time=index * 10**9, data=payload, publish_time=index * 10**9
            )
        writer.add_metadata(METADATA_RECORD_EPISODE, {})
        writer.finish()
    return path


def test_transform_preserves_no_schema_sentinel_and_empty_episode_record(
    state_only_source: Path, tmp_path: Path
) -> None:
    output = tmp_path / "out.mcap"
    write_canonical_episode(state_only_source, output)
    with output.open("rb") as stream:
        reader = make_reader(stream)
        summary = reader.get_summary()
        assert summary is not None
        (channel,) = summary.channels.values()
        assert channel.schema_id == 0
        record_names = [record.name for record in reader.iter_metadata()]
    assert METADATA_RECORD_EPISODE in record_names


def test_transform_rejects_nonconforming_passthrough_video(tmp_path: Path) -> None:
    source = tmp_path / "h265.mcap"
    with source.open("wb") as stream:
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
        message.data = b"\x00\x00\x00\x01\x40junk"
        message.format = "h265"
        writer.add_message(
            channel_id, log_time=10**9, data=message.SerializeToString(), publish_time=10**9
        )
        writer.finish()
    with pytest.raises(ValueError, match="requires 'h264'"):
        write_canonical_episode(source, tmp_path / "out.mcap")


def _write_passthrough_video_source(
    path: Path, messages: list[tuple[str, int, bytes]]
) -> dict[str, list[bytes]]:
    payloads_by_topic: dict[str, list[bytes]] = {}
    with path.open("wb") as stream:
        writer = StockWriter(stream)
        writer.start(profile="", library="test")
        schema_id = writer.register_schema(
            name="foxglove.CompressedVideo",
            encoding="protobuf",
            data=build_file_descriptor_set(CompressedVideo).SerializeToString(),
        )
        channel_ids = {
            topic: writer.register_channel(
                topic=topic, message_encoding="protobuf", schema_id=schema_id
            )
            for topic in dict.fromkeys(topic for topic, _log_time, _data in messages)
        }
        for topic, log_time, access_unit_data in messages:
            message = CompressedVideo()
            message.timestamp.FromNanoseconds(log_time)
            message.frame_id = topic.strip("/")
            message.data = access_unit_data
            message.format = "h264"
            payload = message.SerializeToString()
            writer.add_message(
                channel_ids[topic], log_time=log_time, data=payload, publish_time=log_time
            )
            payloads_by_topic.setdefault(topic, []).append(payload)
        writer.finish()
    return payloads_by_topic


def test_transform_rejects_each_passthrough_video_channel_starting_mid_gop(
    tmp_path: Path,
) -> None:
    source = tmp_path / "interleaved-mid-gop.mcap"
    _write_passthrough_video_source(
        source,
        [
            ("/cam/left", 1, KEYFRAME_ACCESS_UNIT),
            ("/cam/right", 2, NON_KEYFRAME_ACCESS_UNIT),
            ("/cam/left", 3, NON_KEYFRAME_ACCESS_UNIT),
        ],
    )

    with pytest.raises(ValueError, match=r"/cam/right.*starts mid-GOP"):
        write_canonical_episode(source, tmp_path / "out.mcap")


def test_transform_accepts_independent_interleaved_passthrough_video_channels(
    tmp_path: Path,
) -> None:
    source = tmp_path / "interleaved-keyframes.mcap"
    expected_payloads = _write_passthrough_video_source(
        source,
        [
            ("/cam/left", 1, KEYFRAME_ACCESS_UNIT),
            ("/cam/right", 2, KEYFRAME_ACCESS_UNIT),
            ("/cam/left", 3, NON_KEYFRAME_ACCESS_UNIT),
            ("/cam/right", 4, NON_KEYFRAME_ACCESS_UNIT),
        ],
    )
    output = tmp_path / "out.mcap"

    write_canonical_episode(source, output)

    with output.open("rb") as stream:
        reader = make_reader(stream)
        actual_payloads: dict[str, list[bytes]] = {}
        for _schema, channel, message in reader.iter_messages(log_time_order=True):
            actual_payloads.setdefault(channel.topic, []).append(message.data)
    assert actual_payloads == expected_payloads
    report = diagnose(output)
    assert report.conforming, report.summary()


def test_transform_losslessly_inserts_missing_passthrough_video_auds(tmp_path: Path) -> None:
    source = tmp_path / "missing-auds.mcap"
    original_payloads = _write_passthrough_video_source(
        source,
        [
            ("/cam", 1, KEYFRAME_WITHOUT_AUD),
            ("/cam", 2, NON_KEYFRAME_WITHOUT_AUD),
        ],
    )["/cam"]
    output = tmp_path / "out.mcap"

    write_canonical_episode(source, output)

    with output.open("rb") as stream:
        output_payloads = [
            message.data for _schema, _channel, message in make_reader(stream).iter_messages()
        ]
    original_messages = [CompressedVideo.FromString(payload) for payload in original_payloads]
    output_messages = [CompressedVideo.FromString(payload) for payload in output_payloads]
    for original, repaired in zip(original_messages, output_messages, strict=True):
        assert bytes(repaired.data).endswith(bytes(original.data))
        assert len(repaired.data) == len(original.data) + 6
        assert repaired.timestamp == original.timestamp
        assert repaired.frame_id == original.frame_id
        assert repaired.format == original.format
    report = diagnose(output)
    assert report.conforming, report.summary()


def test_transform_still_rejects_undelimited_video_starting_mid_gop(tmp_path: Path) -> None:
    source = tmp_path / "undelimited-mid-gop.mcap"
    _write_passthrough_video_source(source, [("/cam", 1, NON_KEYFRAME_WITHOUT_AUD)])

    with pytest.raises(ValueError, match="starts mid-GOP"):
        write_canonical_episode(source, tmp_path / "out.mcap")


def test_transform_and_doctor_reject_multiple_pictures_without_auds(tmp_path: Path) -> None:
    source = tmp_path / "two-pictures-without-auds.mcap"
    _write_passthrough_video_source(
        source, [("/cam", 1, KEYFRAME_WITHOUT_AUD + NON_KEYFRAME_WITHOUT_AUD)]
    )

    with pytest.raises(ValueError, match="message contains 2 pictures"):
        write_canonical_episode(source, tmp_path / "out.mcap")

    report = diagnose(source)
    assert "video-multiple-access-units" in {finding.code for finding in report.findings}


def test_transform_inserts_missing_aud_into_ros2_video(tmp_path: Path) -> None:
    source = tmp_path / "ros2-missing-aud.mcap"
    writer = Ros2Writer(str(source))
    schema = writer.register_msgdef(
        "foxglove_msgs/msg/CompressedVideo", ROS2_COMPRESSED_VIDEO_SCHEMA
    )
    writer.write_message(
        "/cam",
        schema,
        SimpleNamespace(
            timestamp=SimpleNamespace(sec=0, nanosec=1),
            frame_id="cam",
            data=KEYFRAME_WITHOUT_AUD,
            format="h264",
        ),
        log_time=1,
        publish_time=1,
    )
    writer.finish()
    output = tmp_path / "out.mcap"

    write_canonical_episode(source, output)

    with output.open("rb") as stream:
        schema, channel, message = next(make_reader(stream).iter_messages())
        assert schema is not None
        decode = Ros2DecoderFactory().decoder_for(channel.message_encoding, schema)
        assert decode is not None
        repaired = decode(message.data)
    assert channel.message_encoding == "cdr"
    assert bytes(repaired.data).endswith(KEYFRAME_WITHOUT_AUD)
    assert len(repaired.data) == len(KEYFRAME_WITHOUT_AUD) + 6
    assert repaired.frame_id == "cam"
    assert repaired.format == "h264"


def test_aborted_writer_does_not_publish_partial_file(tmp_path: Path) -> None:
    path = tmp_path / "aborted.mcap"
    with pytest.raises(RuntimeError, match="boom"):  # noqa: SIM117
        with GroupedMcapWriter(path) as writer:
            schema_id = writer.register_schema("s", "jsonschema", b"{}")
            writer.register_channel("/t", "json", schema_id, group="state")
            raise RuntimeError("boom")
    assert not path.exists(), "an aborted write must not leave a partial file"


def test_writer_abort_preserves_previously_published_file(tmp_path: Path) -> None:
    path = tmp_path / "published.mcap"
    path.write_bytes(b"previous valid bytes")
    with pytest.raises(RuntimeError, match="boom"):  # noqa: SIM117
        with GroupedMcapWriter(path) as writer:
            writer.register_channel("/t", "json", NO_SCHEMA_ID, group="state")
            raise RuntimeError("boom")
    assert path.read_bytes() == b"previous valid bytes"


def test_sources_with_the_same_basename_get_distinct_artifact_paths(tmp_path: Path) -> None:
    first_source = synthesize_episode(
        tmp_path / "robot-a" / "run.mcap",
        SyntheticEpisodeSpec(duration_s=1.0, cameras=(), task="task-a"),
    )
    second_source = synthesize_episode(
        tmp_path / "robot-b" / "run.mcap",
        SyntheticEpisodeSpec(duration_s=1.0, cameras=(), task="task-b"),
    )
    app = hflow.App("artifact-identity", data_root=tmp_path / "data", default_checks=())

    first_report = app.process(first_source, record=False, stages={hflow.Stage.SYNC})
    first_canonical_bytes = first_report.canonical_path.read_bytes()
    second_report = app.process(second_source, record=False, stages={hflow.Stage.SYNC})

    assert first_report.canonical_path != second_report.canonical_path
    assert first_report.canonical_path.read_bytes() == first_canonical_bytes


def test_failed_sync_clears_completion_proof_and_blocks_later_stages(tmp_path: Path) -> None:
    source = synthesize_episode(
        tmp_path / "source.mcap",
        SyntheticEpisodeSpec(duration_s=1.0, cameras=()),
    )
    app = hflow.App("sync-proof", data_root=tmp_path / "data", default_checks=())
    successful_report = app.process(source, record=False, stages={hflow.Stage.SYNC})
    previous_canonical_bytes = successful_report.canonical_path.read_bytes()

    source.write_bytes(b"not an mcap file")
    with pytest.raises(InvalidMagic):
        app.process(source, record=False, stages={hflow.Stage.SYNC})

    # Atomic publication preserves the last valid artifact, but its missing
    # completion marker prevents a later stage from mistaking it for this run.
    assert successful_report.canonical_path.read_bytes() == previous_canonical_bytes
    with pytest.raises(FileNotFoundError, match="sync completion marker"):
        app.process(source, record=False, stages={hflow.Stage.META})


def test_check_returning_wrong_type_is_an_error_not_a_crash(tmp_path: Path) -> None:
    source = synthesize_episode(
        tmp_path / "episode.mcap", SyntheticEpisodeSpec(duration_s=2.0, cameras=())
    )
    app = hflow.App("boundary", data_root=tmp_path / "data", default_checks=())

    @app.check(version="1")
    def returns_a_dict(ep: hflow.Episode) -> hflow.CheckResult:
        # Deliberate misuse: the cast smuggles a dict past the type checker,
        # which is exactly what un-typechecked user code would do.
        return cast(hflow.CheckResult, {"black_pct": 1.0})

    @app.check(version="1")
    def well_behaved(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"ran": True})

    report = app.test(source, verbose=False)
    by_name = {run.check.name: run for run in report.checks}
    assert by_name["returns_a_dict"].status == hflow.CheckStatus.ERROR
    assert by_name["returns_a_dict"].error is not None
    assert "expected hflow.CheckResult" in by_name["returns_a_dict"].error
    assert by_name["well_behaved"].status == hflow.CheckStatus.MEASURED
    assert report.has_errors


def _state_only_episode(tmp_path: Path) -> Path:
    """A cheap episode with no cameras, so no ffmpeg runs."""
    return synthesize_episode(
        tmp_path / "episode.mcap",
        SyntheticEpisodeSpec(duration_s=2.0, cameras=()),
    )


def test_two_checks_recording_one_measurement_key_are_refused(tmp_path: Path) -> None:
    """Every step of one run shares its fingerprint and timestamp, so a shared
    key is a tie the catalog resolves arbitrarily -- one step's value silently
    disappears. Refuse it where it is still fixable.
    """
    app = hflow.App("key-collision", data_root=tmp_path / "data", default_checks=())

    @app.check(version="1")
    def first(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"shared_count": 1})

    @app.check(version="1")
    def second(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"shared_count": 2})

    with pytest.raises(ValueError, match="same measurement key") as failure:
        app.test(_state_only_episode(tmp_path), verbose=False)
    message = str(failure.value)
    assert "'shared_count'" in message
    assert "'first'" in message
    assert "'second'" in message
    # The suggested fix is pasteable, so it has to parse as written.
    suggestion = message.split("Namespacing looks like:", 1)[1]
    compile(textwrap.dedent(suggestion).strip(), "<suggestion>", "exec")


def test_a_check_and_an_enrichment_label_collision_is_refused(tmp_path: Path) -> None:
    """Checks and enrichment labels share one measurement-key namespace."""
    app = hflow.App("key-collision-enrich", data_root=tmp_path / "data", default_checks=())

    @app.check(version="1")
    def measures(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"overlap": 1})

    @app.enrich(version="1")
    def labels(ep: hflow.Episode) -> hflow.EnrichmentResult:
        return hflow.EnrichmentResult(labels={"overlap": 2})

    with pytest.raises(ValueError, match="same measurement key"):
        app.test(_state_only_episode(tmp_path), verbose=False)


def test_a_refused_collision_records_nothing(tmp_path: Path) -> None:
    """The guard runs before the append, so a refused run leaves no row behind.

    Recorded on purpose: ``record=True`` is what ``process`` defaults to and
    what every DAG batch uses, so testing the non-recording path would prove
    the ordering only by implication.
    """
    app = hflow.App("key-collision-record", data_root=tmp_path / "data", default_checks=())

    @app.check(version="1")
    def left(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"same": 1.0})

    @app.check(version="1")
    def right(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"same": 2.0})

    with pytest.raises(ValueError, match="same measurement key"):
        app.process(_state_only_episode(tmp_path), record=True)

    catalog_root = tmp_path / "data" / "catalog"
    assert list(catalog_root.rglob("*.parquet")) == []


def test_two_steps_may_share_a_tag(tmp_path: Path) -> None:
    """Tags carry check_name and have no per-key latest ranking, so sharing one
    loses nothing -- the guard must not overreach into them.
    """
    app = hflow.App("shared-tag", data_root=tmp_path / "data", default_checks=())

    @app.check(version="1")
    def first(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"first_count": 1}, tags=["reviewed"])

    @app.check(version="1")
    def second(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"second_count": 2}, tags=["reviewed"])

    report = app.test(_state_only_episode(tmp_path), verbose=False)
    assert not report.has_errors


def test_check_with_required_extra_parameter_fails_at_registration() -> None:
    app = hflow.App("signature-guard", data_root=Path("/tmp"), default_checks=())

    def requires_topics(ep: hflow.Episode, *, topics: list[str]) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"n": len(topics)})

    with pytest.raises(ValueError, match="topics"):
        app.check(version="1")(cast(hflow.steps.CheckFunction, requires_topics))

    assert app.checks == []


def test_check_with_optional_extra_parameter_registers() -> None:
    app = hflow.App("signature-optional", data_root=Path("/tmp"), default_checks=())

    @app.check(version="1")
    def optional_topic(
        ep: hflow.Episode, *, topics: tuple[str, ...] = ("/joint_states",)
    ) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"n": len(topics)})

    assert {check.name for check in app.checks} == {"optional_topic"}


def test_check_without_episode_parameter_fails_at_registration() -> None:
    app = hflow.App("signature-zeroarg", data_root=Path("/tmp"), default_checks=())

    def no_episode() -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"n": 1})

    with pytest.raises(ValueError, match="cannot accept the episode"):
        app.check(version="1")(cast(hflow.steps.CheckFunction, no_episode))

    assert app.checks == []


def test_check_with_only_kwargs_fails_at_registration() -> None:
    app = hflow.App("signature-kwargs-only", data_root=Path("/tmp"), default_checks=())

    def kwargs_only(**kwargs: object) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"n": len(kwargs)})

    with pytest.raises(ValueError, match="cannot accept the episode"):
        app.check(version="1")(cast(hflow.steps.CheckFunction, kwargs_only))

    assert app.checks == []


def test_check_with_varargs_registers() -> None:
    app = hflow.App("signature-varargs", data_root=Path("/tmp"), default_checks=())

    @app.check(version="1")
    def varargs_check(*args: object) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"n": len(args)})

    assert {check.name for check in app.checks} == {"varargs_check"}


def test_check_whose_episode_parameter_has_a_default_registers_and_runs() -> None:
    """A default on the episode parameter does not stop it receiving the episode.

    The check is called as ``function(canonical_episode)``, so the default is
    never used and the signature is satisfiable. An earlier form of the
    accepts-the-episode test skipped every defaulted positional parameter
    before claiming the episode's slot, which rejected this at registration
    even though it had always worked.
    """
    app = hflow.App("signature-defaulted-episode", data_root=Path("/tmp"), default_checks=())

    @app.check(version="1")
    def defaulted_episode(episode: hflow.Episode | None = None) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"episode_arrived": episode is not None})

    assert {check.name for check in app.checks} == {"defaulted_episode"}
    # Registration is the regression, but assert it is callable the way the
    # runtime calls it, so the test fails if the call convention ever changes.
    assert app.checks[0].function(cast(hflow.Episode, object())).measurements == {
        "episode_arrived": True
    }


@pytest.mark.parametrize("step_kind", ["enrichment", "derived channel"])
def test_enrichments_and_derives_reject_unsatisfiable_signatures(step_kind: str) -> None:
    """The same guard as checks: all three are called with one Episode.

    #86 and #93 fixed this for ``app.check(version="1")`` only, leaving the two
    structurally identical registration paths accepting a signature the
    runtime could never call.
    """
    app = hflow.App(f"guard-{step_kind.split()[0]}", data_root=Path("/tmp"), default_checks=())

    def needs_topics(episode: hflow.Episode, *, topics: list[str]) -> object:
        return None

    def takes_nothing() -> object:
        return None

    def register(function: object) -> None:
        if step_kind == "enrichment":
            app.enrich(version="1", name="guarded")(cast(hflow.steps.EnrichmentFunction, function))
        else:
            app.derive("/derived", version="1")(cast(hflow.steps.DerivedFunction, function))

    with pytest.raises(ValueError, match="topics"):
        register(needs_topics)
    with pytest.raises(ValueError, match="cannot accept the episode"):
        register(takes_nothing)
    assert app.enrichments == []
    assert app.derived == []


def test_wrapper_example_binds_every_missing_parameter() -> None:
    """The example is meant to be pasted, so it must bind all of them.

    Binding only the first left a snippet that still raised TypeError on the
    second parameter.
    """
    app = hflow.App("wrapper-example", data_root=Path("/tmp"), default_checks=())

    def two_missing(
        episode: hflow.Episode, *, topics: list[str], threshold: float
    ) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"n": len(topics) + threshold})

    with pytest.raises(ValueError) as raised:
        app.check(version="1")(cast(hflow.steps.CheckFunction, two_missing))
    message = str(raised.value)
    assert "threshold=..." in message
    assert "topics=..." in message


def test_registration_accepts_a_partial_with_an_explicit_version() -> None:
    app = hflow.App("partial-registration", data_root=Path("/tmp"), default_checks=())
    bound = functools.partial(hflow.checks.action_rate, topics=["/joint_states"])

    app.check(version="action-rate-v2", name="bound")(bound)

    assert {check.name for check in app.checks} == {"bound"}
    assert app.checks[0].version == "action-rate-v2"


def test_explicit_version_supports_opaque_callable_configuration() -> None:
    class OpaqueClient:
        pass

    def scored_by_client(episode: hflow.Episode, *, client: object) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"ok": client is not None})

    app = hflow.App("opaque-registration", data_root=Path("/tmp"), default_checks=())
    bound = functools.partial(scored_by_client, client=OpaqueClient())

    app.check(version="vendor-model-v2", name="opaque")(bound)

    assert app.checks[0].version == "vendor-model-v2"


def test_step_version_changes_only_when_the_author_bumps_it() -> None:
    def before_refactor(_episode: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"n": 0})

    def after_refactor(_episode: hflow.Episode) -> hflow.CheckResult:
        running_total = 0
        return hflow.CheckResult(measurements={"n": running_total})

    before = hflow.App("before", data_root=Path("/tmp"), default_checks=())
    before.check(version="3", name="owned")(before_refactor)
    after = hflow.App("after", data_root=Path("/tmp"), default_checks=())
    after.check(version="3", name="owned")(after_refactor)
    bumped = hflow.App("bumped", data_root=Path("/tmp"), default_checks=())
    bumped.check(version="4", name="owned")(after_refactor)

    assert before.checks[0].version == after.checks[0].version
    assert bumped.checks[0].version != before.checks[0].version


def test_all_step_registration_surfaces_reject_invalid_versions() -> None:
    app = hflow.App("versions", data_root=Path("/tmp"), default_checks=())

    with pytest.raises(ValueError, match="must not be empty"):
        app.check(version="")
    with pytest.raises(ValueError, match="leading or trailing whitespace"):
        app.enrich(version=" 1")
    with pytest.raises(ValueError, match="leading or trailing whitespace"):
        app.derive("/derived", version="1 ")


def test_non_sync_stages_never_fetch_the_raw_source(tmp_path: Path) -> None:
    """Relabel on a fresh worker must not download the (possibly huge) raw file.

    Proven behaviorally: after a full run, the remote source object is
    DELETED; a labels-only rerun still succeeds because it reads only the
    canonical episode and the sync-completion marker.
    """
    from hflow.storage import BucketStorageRoot
    from hflow.testing import SyntheticEpisodeSpec, synthesize_episode

    pytest.importorskip("obstore")
    remote_dir = tmp_path / "bucket"
    remote_dir.mkdir()
    data_root = BucketStorageRoot(f"file://{remote_dir}", mirror=tmp_path / "mirror-a")
    episode_file = synthesize_episode(
        tmp_path / "e.mcap", SyntheticEpisodeSpec(cameras=(), black_segment=None, duration_s=2.0)
    )
    data_root.publish(episode_file, "landing/e.mcap")

    app = hflow.App("fetchless", data_root=data_root, default_checks=())
    app.process("landing/e.mcap", record=True)

    data_root.delete("landing/e.mcap")
    # A different worker: same bucket, fresh mirror (no cached source).
    fresh_root = BucketStorageRoot(f"file://{remote_dir}", mirror=tmp_path / "mirror-b")
    relabel_app = hflow.App("fetchless", data_root=fresh_root, default_checks=())
    report = relabel_app.process("landing/e.mcap", record=True, stages={hflow.Stage.LABELS})
    assert report.stamps.pipeline_version


def test_missing_artifact_is_the_steps_error_not_the_runs(tmp_path: Path) -> None:
    from hflow.testing import SyntheticEpisodeSpec, synthesize_episode

    episode_file = synthesize_episode(
        tmp_path / "e.mcap", SyntheticEpisodeSpec(cameras=(), black_segment=None, duration_s=2.0)
    )
    app = hflow.App("artifact-crash", data_root=tmp_path / "data", default_checks=())

    @app.enrich(version="1")
    def declares_a_ghost(ep: hflow.Episode) -> hflow.EnrichmentResult:
        return hflow.EnrichmentResult(artifacts={"ghost": tmp_path / "never-written.png"})

    report = app.process(episode_file, record=True)
    (enrichment_run,) = report.enrichments
    assert enrichment_run.status is hflow.CheckStatus.ERROR
    assert "never-written.png" in (enrichment_run.error or "")
    assert report.catalog_entry is not None and report.catalog_entry.written


def test_source_identity_is_stable_across_vantage_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every spelling of one data-root episode shares one identity and run dir.

    This is the host-vs-container case: /opt/airflow/data/landing/e.mcap in
    the runtime, ./data/landing/e.mcap on the host, and the bare key accepted
    by ingest are the same episode.
    """
    from hflow.testing import SyntheticEpisodeSpec, synthesize_episode

    data_root = tmp_path / "data"
    episode_file = synthesize_episode(
        data_root / "landing" / "e.mcap",
        SyntheticEpisodeSpec(cameras=(), black_segment=None, duration_s=2.0),
    )
    app = hflow.App("vantage", data_root=data_root, default_checks=())
    monkeypatch.chdir(tmp_path)

    assert {
        app.source_identity("landing/e.mcap"),
        app.source_identity(Path("data") / "landing/e.mcap"),
        app.source_identity(episode_file.resolve()),
    } == {"landing/e.mcap"}

    full_report = app.process(episode_file.resolve(), record=True)

    relative_app = hflow.App("vantage", data_root=Path("data"), default_checks=())
    prefixed_report = relative_app.process(
        "data/landing/e.mcap", record=True, stages={hflow.Stage.LABELS}
    )
    bare_report = relative_app.process("landing/e.mcap", record=True, stages={hflow.Stage.LABELS})

    assert prefixed_report.canonical_path.resolve() == full_report.canonical_path.resolve()
    assert bare_report.canonical_path.resolve() == full_report.canonical_path.resolve()


def test_source_identity_refuses_two_different_relative_local_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root_source = tmp_path / "data" / "landing" / "e.mcap"
    cwd_source = tmp_path / "landing" / "e.mcap"
    data_root_source.parent.mkdir(parents=True)
    cwd_source.parent.mkdir(parents=True)
    data_root_source.write_bytes(b"data-root recording")
    cwd_source.write_bytes(b"working-directory recording")
    monkeypatch.chdir(tmp_path)

    app = hflow.App("ambiguous", data_root=tmp_path / "data", default_checks=())

    with pytest.raises(ValueError, match=r"relative source reference.*is ambiguous"):
        app.source_identity("landing/e.mcap")
