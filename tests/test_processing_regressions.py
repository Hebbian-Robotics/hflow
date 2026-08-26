"""Processing, publication, and user-step boundary regressions."""

import functools
import json
import logging
import re
import textwrap
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
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
from hflow.steps import (
    UNDESCRIBED_CONFIGURATION_KEY,
    compute_check_version,
    step_identity_payload,
)
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode
from hflow.transform import write_canonical_episode
from hflow.video import estimate_fps_from_log_times

ANNEX_B_START_CODE = b"\x00\x00\x00\x01"
KEYFRAME_ACCESS_UNIT = b"".join(
    ANNEX_B_START_CODE + bytes([nal_type]) + b"payload" for nal_type in (0x09, 0x67, 0x68, 0x65)
)
NON_KEYFRAME_ACCESS_UNIT = b"".join(
    ANNEX_B_START_CODE + bytes([nal_type]) + b"payload" for nal_type in (0x09, 0x41)
)
KEYFRAME_WITHOUT_AUD = b"".join(
    ANNEX_B_START_CODE + bytes([nal_type]) + b"payload" for nal_type in (0x67, 0x68, 0x65)
)
NON_KEYFRAME_WITHOUT_AUD = ANNEX_B_START_CODE + b"\x41payload"
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

    @app.check()
    def returns_a_dict(ep: hflow.Episode) -> hflow.CheckResult:
        # Deliberate misuse: the cast smuggles a dict past the type checker,
        # which is exactly what un-typechecked user code would do.
        return cast(hflow.CheckResult, {"black_pct": 1.0})

    @app.check()
    def well_behaved(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"ran": True})

    report = app.test(source, verbose=False)
    by_name = {run.check.name: run for run in report.checks}
    assert by_name["returns_a_dict"].status == hflow.CheckStatus.ERROR
    assert by_name["returns_a_dict"].error is not None
    assert "expected hflow.CheckResult" in by_name["returns_a_dict"].error
    assert by_name["well_behaved"].status == hflow.CheckStatus.MEASURED
    assert report.has_errors


def test_step_version_includes_captured_configuration() -> None:
    def make_threshold_check(threshold: float) -> Callable[[hflow.Episode], hflow.CheckResult]:
        def threshold_check(_episode: hflow.Episode) -> hflow.CheckResult:
            return hflow.CheckResult(verdict=threshold > 0.5)

        return threshold_check

    low_threshold_version = compute_check_version(
        "threshold", make_threshold_check(0.1), False, frozenset(), None
    )
    high_threshold_version = compute_check_version(
        "threshold", make_threshold_check(0.9), False, frozenset(), None
    )

    assert low_threshold_version != high_threshold_version


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

    @app.check()
    def first(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"shared_count": 1})

    @app.check()
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

    @app.check()
    def measures(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"overlap": 1})

    @app.enrich()
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

    @app.check()
    def left(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"same": 1.0})

    @app.check()
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

    @app.check()
    def first(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"first_count": 1}, tags=["reviewed"])

    @app.check()
    def second(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"second_count": 2}, tags=["reviewed"])

    report = app.test(_state_only_episode(tmp_path), verbose=False)
    assert not report.has_errors


def test_check_with_required_extra_parameter_fails_at_registration() -> None:
    app = hflow.App("signature-guard", data_root=Path("/tmp"), default_checks=())

    def requires_topics(ep: hflow.Episode, *, topics: list[str]) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"n": len(topics)})

    with pytest.raises(ValueError, match="topics"):
        app.check()(cast(hflow.steps.CheckFunction, requires_topics))

    assert app.checks == []


def test_check_with_optional_extra_parameter_registers() -> None:
    app = hflow.App("signature-optional", data_root=Path("/tmp"), default_checks=())

    @app.check()
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
        app.check()(cast(hflow.steps.CheckFunction, no_episode))

    assert app.checks == []


def test_check_with_only_kwargs_fails_at_registration() -> None:
    app = hflow.App("signature-kwargs-only", data_root=Path("/tmp"), default_checks=())

    def kwargs_only(**kwargs: object) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"n": len(kwargs)})

    with pytest.raises(ValueError, match="cannot accept the episode"):
        app.check()(cast(hflow.steps.CheckFunction, kwargs_only))

    assert app.checks == []


def test_check_with_varargs_registers() -> None:
    app = hflow.App("signature-varargs", data_root=Path("/tmp"), default_checks=())

    @app.check()
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

    @app.check()
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

    #86 and #93 fixed this for ``app.check()`` only, leaving the two
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
            app.enrich(name="guarded")(cast(hflow.steps.EnrichmentFunction, function))
        else:
            app.derive("/derived")(cast(hflow.steps.DerivedFunction, function))

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
        app.check()(cast(hflow.steps.CheckFunction, two_missing))
    message = str(raised.value)
    assert "threshold=..." in message
    assert "topics=..." in message


def test_step_version_accepts_functools_partial_with_bound_args() -> None:
    bound = functools.partial(hflow.checks.action_rate, topics=["/joint_states"])
    version = compute_check_version("bound", bound, False, frozenset(), None)

    assert version


def test_step_version_differs_when_partial_bindings_change() -> None:
    """Hold the name fixed, so only the binding can move the version.

    The step name is part of the identity, so naming these differently would
    pass whether or not the bound arguments were read at all.
    """
    topics_a = functools.partial(hflow.checks.action_rate, topics=["/joint_states"])
    topics_b = functools.partial(hflow.checks.action_rate, topics=["/other_stream"])
    version_a = compute_check_version("same_name", topics_a, False, frozenset(), None)
    version_b = compute_check_version("same_name", topics_b, False, frozenset(), None)

    assert version_a != version_b
    # Stable across construction, so the identity cannot be address-derived.
    rebuilt_a = functools.partial(hflow.checks.action_rate, topics=["/joint_states"])
    assert compute_check_version("same_name", rebuilt_a, False, frozenset(), None) == version_a


def test_step_version_still_refuses_a_partial_bound_to_an_opaque_value() -> None:
    """Transparency of the wrapper does not make its contents transparent.

    A partial is identifiable only because its bound values are. Bind
    something the identity machinery cannot describe and it must keep
    refusing, or the version silently stops tracking that value.
    """

    class OpaqueClient:
        pass

    def scored_by_client(episode: hflow.Episode, *, client: object) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"ok": client is not None})

    bound = functools.partial(scored_by_client, client=OpaqueClient())
    with pytest.raises(ValueError, match="cannot derive a stable version identity"):
        compute_check_version("opaque", bound, False, frozenset(), None)


def test_check_registration_accepts_functools_partial() -> None:
    app = hflow.App("partial-registration", data_root=Path("/tmp"), default_checks=())
    bound = functools.partial(hflow.checks.action_rate, topics=["/joint_states"])

    app.check(name="bound")(bound)

    assert {check.name for check in app.checks} == {"bound"}
    assert app.checks[0].version


_STEP_VERSION_GLOBAL_THRESHOLD = 0.1


def _global_threshold_check(_episode: hflow.Episode) -> hflow.CheckResult:
    return hflow.CheckResult(verdict=_STEP_VERSION_GLOBAL_THRESHOLD > 0.5)


def test_step_version_includes_referenced_global_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    low_threshold_version = compute_check_version(
        "global-threshold", _global_threshold_check, False, frozenset(), None
    )
    monkeypatch.setattr(
        f"{__name__}._STEP_VERSION_GLOBAL_THRESHOLD",
        0.9,
    )
    high_threshold_version = compute_check_version(
        "global-threshold", _global_threshold_check, False, frozenset(), None
    )

    assert low_threshold_version != high_threshold_version


# Composition in HFlow runs through shared library code, not through edges
# between checks: two built-ins share the video-measurements package's frame
# statistics, and the motion checks share its camera-motion definition. That only
# keeps its integrity if a step version follows the code the step calls, all
# the way down. Otherwise editing a parser or a constant one level below the
# step changes what it measures while its version stands still, and the new
# rows append under the old version.
#
# These helpers are module level on purpose: a step hash follows GLOBAL names,
# so a helper defined inside a test body would be invisible to the walk under
# test and every assertion below would pass vacuously.

_TRANSITIVE_TOLERANCE = 0.25
_MEMOISED_TOLERANCE = 0.5


def _two_levels_below_the_step(value: float) -> bool:
    return value > _TRANSITIVE_TOLERANCE


def _named_directly_by_the_step(value: float) -> bool:
    return _two_levels_below_the_step(value)


def _check_over_a_helper_chain(_episode: hflow.Episode) -> hflow.CheckResult:
    return hflow.CheckResult(verdict=_named_directly_by_the_step(1.0))


@functools.lru_cache(maxsize=1)
def _memoised_helper() -> bool:
    return _MEMOISED_TOLERANCE > 0.1


def _check_over_a_memoised_helper(_episode: hflow.Episode) -> hflow.CheckResult:
    return hflow.CheckResult(verdict=_memoised_helper())


class _OpaqueHelperState:
    """Something the identity machinery cannot describe, held by a helper."""


_OPAQUE_HELPER_STATE = _OpaqueHelperState()


def _helper_holding_opaque_state() -> bool:
    return _OPAQUE_HELPER_STATE is not None


def _check_over_an_opaque_helper(_episode: hflow.Episode) -> hflow.CheckResult:
    return hflow.CheckResult(verdict=_helper_holding_opaque_state())


def _mutually_recursive_even(remaining: int) -> bool:
    return True if remaining == 0 else _mutually_recursive_odd(remaining - 1)


def _mutually_recursive_odd(remaining: int) -> bool:
    return False if remaining == 0 else _mutually_recursive_even(remaining - 1)


def _check_over_mutual_recursion(_episode: hflow.Episode) -> hflow.CheckResult:
    return hflow.CheckResult(verdict=_mutually_recursive_even(2))


_NONE: frozenset[str] = frozenset()


def _version_of(function: Callable[..., object], name: str = "probe") -> str:
    return compute_check_version(name, function, False, _NONE, None)


def test_step_version_follows_a_helper_the_step_does_not_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The step names one helper; the behavior lives in the one below it."""
    before = _version_of(_check_over_a_helper_chain)
    monkeypatch.setattr(
        f"{__name__}._two_levels_below_the_step",
        lambda value: value > 99.0,
    )

    assert _version_of(_check_over_a_helper_chain) != before


def test_step_version_follows_a_constant_read_below_the_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tuned constant is a behavior change wherever it is defined.

    Distinct from the helper test above: swapping a function changes a source
    the walk already read, while a constant is only reachable by descending
    into that helper's own captured globals.
    """
    before = _version_of(_check_over_a_helper_chain)
    monkeypatch.setattr(f"{__name__}._TRANSITIVE_TOLERANCE", 0.75)

    assert _version_of(_check_over_a_helper_chain) != before


def test_step_version_reads_through_a_memoised_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cache is not behavior, so the walk unwraps it and keeps going.

    Registration used to refuse a step that reached a memoised helper at all
    (``hflow.ffmpeg._binary`` memoises the probes the instrument calls), which
    made "cannot version this" indistinguishable from "nothing to version".
    """
    before = _version_of(_check_over_a_memoised_helper)
    monkeypatch.setattr(f"{__name__}._MEMOISED_TOLERANCE", 0.9)

    assert _version_of(_check_over_a_memoised_helper) != before


def test_step_version_terminates_on_mutually_recursive_helpers() -> None:
    """Cycles are cut by the visited set, not by a depth limit.

    Completing at all is most of the assertion; the repeat pins that the cut
    is taken at the same place every time, since a cycle broken by traversal
    order rather than by identity would hash differently per call.
    """
    assert _version_of(_check_over_mutual_recursion) == _version_of(_check_over_mutual_recursion)


def test_step_version_follows_a_parsing_pattern_below_the_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A compiled regex is behavior when it is how output is read.

    ``_METADATA_LINE_PATTERN`` is how the ffmpeg instrument turns a pass over
    the pixels into the measurements every camera check reports, two calls
    below the check. Recompiling it differently changes those numbers, so it
    has to change their versions.
    """
    before = _version_of(hflow.checks.camera_signal_quality)
    monkeypatch.setattr(
        "hflow._video_measurements._frame_statistics._METADATA_LINE_PATTERN",
        re.compile(r"^(?P<key>never_matches)=(?P<value>.*)$"),
    )

    assert _version_of(hflow.checks.camera_signal_quality) != before


def test_step_version_ignores_a_logger_a_helper_happens_to_hold() -> None:
    """Runtime logging configuration is not part of what a step computes.

    A logger is reachable from real built-ins (``hflow.ffmpeg._binary`` warns
    about an unpinned binary), so it must be describable or the walk stops
    there. Describing its handlers or level would instead hash the same code
    differently under different logging setups.
    """
    logger = logging.getLogger("hflow.test.identity")

    def check_holding_a_logger(_episode: hflow.Episode) -> hflow.CheckResult:
        logger.debug("measuring")
        return hflow.CheckResult(measurements={"ok": True})

    before = _version_of(check_holding_a_logger)
    logger.setLevel(logging.CRITICAL)
    logger.addHandler(logging.NullHandler())

    assert _version_of(check_holding_a_logger) == before


def _builtin_check_names() -> list[str]:
    """Every check function hflow.checks defines, discovered rather than
    listed, so a newly added built-in is guarded without editing this file."""
    return sorted(
        name
        for name in dir(hflow.checks)
        if not name.startswith("_")
        and callable(getattr(hflow.checks, name))
        and getattr(getattr(hflow.checks, name), "__module__", "") == "hflow.checks"
    )


@pytest.mark.parametrize("builtin_name", _builtin_check_names())
def test_no_builtin_check_leaves_part_of_itself_undescribed(builtin_name: str) -> None:
    """Every built-in's version must cover ALL the code it reaches.

    The walk degrades rather than refusing when it meets state it cannot
    describe, which is right for a user's pipeline and wrong for hflow's own
    code: a marker here means someone can edit the helper below it and no
    version will move. That gap is invisible in a hash, which is why this
    asserts over the payload. It is the regression this whole change is
    about, one level further down.
    """
    builtin = getattr(hflow.checks, builtin_name)
    payload = step_identity_payload(builtin_name, builtin, False, frozenset(), None)

    assert UNDESCRIBED_CONFIGURATION_KEY not in json.dumps(payload)


def test_step_version_degrades_instead_of_refusing_over_a_helpers_private_state() -> None:
    """A user's helper may hold anything; registration must still work.

    The step's OWN captured state stays strict (see the opaque-partial test
    below). This is only about state one call further down, where refusing
    would make an unrelated helper's internals fatal to registering a step
    that is perfectly describable itself.
    """
    payload = step_identity_payload(
        "over-opaque-helper", _check_over_an_opaque_helper, False, frozenset(), None
    )

    assert UNDESCRIBED_CONFIGURATION_KEY in json.dumps(payload)
    assert _version_of(_check_over_an_opaque_helper)


def test_step_version_does_not_descend_into_a_dependency() -> None:
    """A step is not re-versioned by an unrelated release of what it imports.

    The walk stops at the first-party boundary. Following a dependency's
    internals would restore exactly the coupling ``hflow.behavior`` removed:
    a numpy or scipy upgrade that changed nothing a step observes would still
    invalidate every corpus. The dependency's own source is still read, so
    swapping which function is called is caught. Only its private internals
    are out of scope.
    """
    dependency = ModuleType("vendor_analytics")
    exec(
        "def _private_internal(value):\n"
        "    return value + 1\n"
        "def public_entry_point(value):\n"
        "    return _private_internal(value)\n",
        dependency.__dict__,
    )
    public_entry_point = dependency.public_entry_point

    def check_over_a_dependency(_episode: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(verdict=public_entry_point(1) > 0)

    assert public_entry_point.__module__ == "vendor_analytics"
    before = _version_of(check_over_a_dependency)
    exec(
        "def _private_internal(value):\n    return value * 1000\n",
        dependency.__dict__,
    )

    assert _version_of(check_over_a_dependency) == before


def _declared_check_before_refactor(_episode: hflow.Episode) -> hflow.CheckResult:
    total = 0
    return hflow.CheckResult(measurements={"n": total})


def _declared_check_after_refactor(_episode: hflow.Episode) -> hflow.CheckResult:
    # Renamed local, added comment, reads the shared constant the same way.
    running_total = 0
    return hflow.CheckResult(measurements={"n": running_total})


def test_a_declared_version_survives_a_refactor_of_the_step() -> None:
    """The point of declaring: the author judges two spellings equivalent.

    Without this the declaration was advisory: the source hash leaked back
    in, so refactoring a step that had opted out of derived versioning still
    split its rows, which is the one thing declaring exists to prevent.
    """
    before = compute_check_version(
        "owned", _declared_check_before_refactor, False, _NONE, None, "3"
    )
    after = compute_check_version("owned", _declared_check_after_refactor, False, _NONE, None, "3")

    assert before == after
    assert _version_of(_declared_check_before_refactor) != _version_of(
        _declared_check_after_refactor
    ), "the same refactor must still move a DERIVED version"


def test_a_declared_version_still_tracks_what_was_declared_beside_it() -> None:
    """Declaring owns the implementation, never the registration.

    ``critical`` and a gate are written at the decorator, not refactored into
    existence: changing one is a deliberate policy edit with a visible diff,
    and a gate especially must keep moving the version or two thresholds share
    one and curation can pin neither.
    """
    owned = ("owned", _declared_check_before_refactor, False, _NONE, None, "3")
    baseline = compute_check_version(*owned)
    gate = hflow.Gate(accept_when=(hflow.Threshold("n", hflow.Comparison.AT_MOST, 1.0),))
    looser = hflow.Gate(accept_when=(hflow.Threshold("n", hflow.Comparison.AT_MOST, 9.0),))

    assert compute_check_version(*owned, gate=gate) != baseline
    assert compute_check_version(*owned, gate=looser) != compute_check_version(*owned, gate=gate)
    critical_owned = ("owned", _declared_check_before_refactor, True, _NONE, None, "3")
    assert compute_check_version(*critical_owned) != baseline


def test_bumping_a_declared_version_is_what_moves_it() -> None:
    """The author's promise is the whole identity, so the promise must count."""
    arguments = ("owned", _declared_check_before_refactor, False, _NONE, None)

    assert compute_check_version(*arguments, "3") != compute_check_version(*arguments, "4")


def test_declared_step_version_supports_opaque_callable() -> None:
    opaque_callable = cast(Callable[..., object], len)
    version = compute_check_version(
        "opaque",
        opaque_callable,
        False,
        frozenset(),
        None,
        declared_version="vendor-model-v2",
    )
    assert len(version) == 12


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

    @app.enrich()
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
    """The same episode named absolutely and root-relatively shares one run dir.

    This is the host-vs-container case: /opt/airflow/data/landing/e.mcap in
    the runtime and ./data/landing/e.mcap on the host are the same episode.
    """
    from hflow.testing import SyntheticEpisodeSpec, synthesize_episode

    data_root = tmp_path / "data"
    episode_file = synthesize_episode(
        data_root / "landing" / "e.mcap",
        SyntheticEpisodeSpec(cameras=(), black_segment=None, duration_s=2.0),
    )
    app = hflow.App("vantage", data_root=data_root, default_checks=())
    full_report = app.process(episode_file.resolve(), record=True)

    monkeypatch.chdir(tmp_path)
    relative_app = hflow.App("vantage", data_root=Path("data"), default_checks=())
    relabel_report = relative_app.process(
        "data/landing/e.mcap", record=True, stages={hflow.Stage.LABELS}
    )
    assert relabel_report.canonical_path.resolve() == full_report.canonical_path.resolve()
