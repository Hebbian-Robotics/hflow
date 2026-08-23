"""Processing, publication, and user-step boundary regressions."""

import functools
import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from foxglove_schemas_protobuf.CompressedVideo_pb2 import CompressedVideo
from mcap.exceptions import InvalidMagic
from mcap.reader import make_reader
from mcap.writer import Writer as StockWriter
from mcap_protobuf.schema import build_file_descriptor_set

import hflow
from hflow.format import METADATA_RECORD_EPISODE
from hflow.mcap_writer import CanonicalMcapWriter
from hflow.steps import compute_check_version
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode
from hflow.transform import write_canonical_episode
from hflow.video import estimate_fps_from_log_times


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


def test_aborted_writer_does_not_publish_partial_file(tmp_path: Path) -> None:
    path = tmp_path / "aborted.mcap"
    with pytest.raises(RuntimeError, match="boom"):  # noqa: SIM117
        with CanonicalMcapWriter(path) as writer:
            schema_id = writer.register_schema("s", "jsonschema", b"{}")
            writer.register_channel("/t", "json", schema_id, group="state")
            raise RuntimeError("boom")
    assert not path.exists(), "an aborted write must not leave a partial file"


def test_writer_abort_preserves_previously_published_file(tmp_path: Path) -> None:
    path = tmp_path / "published.mcap"
    path.write_bytes(b"previous valid bytes")
    with pytest.raises(RuntimeError, match="boom"):  # noqa: SIM117
        with CanonicalMcapWriter(path) as writer:
            writer.register_channel("/t", "json", 0, group="state")
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
    app = hflow.App("artifact-identity", data_root=tmp_path / "data")

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
    app = hflow.App("sync-proof", data_root=tmp_path / "data")
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
    app = hflow.App("boundary", data_root=tmp_path / "data")

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
    app = hflow.App("key-collision", data_root=tmp_path / "data")

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


def test_a_check_and_an_enrichment_label_collision_is_refused(tmp_path: Path) -> None:
    """Checks and enrichment labels share one measurement-key namespace."""
    app = hflow.App("key-collision-enrich", data_root=tmp_path / "data")

    @app.check()
    def measures(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"overlap": 1})

    @app.enrich()
    def labels(ep: hflow.Episode) -> hflow.EnrichmentResult:
        return hflow.EnrichmentResult(labels={"overlap": 2})

    with pytest.raises(ValueError, match="same measurement key"):
        app.test(_state_only_episode(tmp_path), verbose=False)


def test_the_dev_loop_refuses_a_collision_without_recording(tmp_path: Path) -> None:
    """The guard runs outside ``if record:`` so it fires on episode one rather
    than at the first curation query.
    """
    app = hflow.App("key-collision-dev", data_root=tmp_path / "data")

    @app.check()
    def left(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"same": 1.0})

    @app.check()
    def right(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"same": 2.0})

    with pytest.raises(ValueError, match="same measurement key"):
        app.process(_state_only_episode(tmp_path), record=False)

    catalog_root = tmp_path / "data" / "catalog"
    assert list(catalog_root.rglob("*.parquet")) == []


def test_two_steps_may_share_a_tag(tmp_path: Path) -> None:
    """Tags carry check_name and have no per-key latest ranking, so sharing one
    loses nothing -- the guard must not overreach into them.
    """
    app = hflow.App("shared-tag", data_root=tmp_path / "data")

    @app.check()
    def first(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"first_count": 1}, tags=["reviewed"])

    @app.check()
    def second(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"second_count": 2}, tags=["reviewed"])

    report = app.test(_state_only_episode(tmp_path), verbose=False)
    assert not report.has_errors


def test_check_with_required_extra_parameter_fails_at_registration() -> None:
    app = hflow.App("signature-guard", data_root=Path("/tmp"))

    def requires_topics(ep: hflow.Episode, *, topics: list[str]) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"n": len(topics)})

    with pytest.raises(ValueError, match="topics"):
        app.check()(cast(hflow.steps.CheckFunction, requires_topics))

    assert app.checks == []


def test_check_with_optional_extra_parameter_registers() -> None:
    app = hflow.App("signature-optional", data_root=Path("/tmp"))

    @app.check()
    def optional_topic(
        ep: hflow.Episode, *, topics: tuple[str, ...] = ("/joint_states",)
    ) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"n": len(topics)})

    assert {check.name for check in app.checks} == {"optional_topic"}


def test_check_without_episode_parameter_fails_at_registration() -> None:
    app = hflow.App("signature-zeroarg", data_root=Path("/tmp"))

    def no_episode() -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"n": 1})

    with pytest.raises(ValueError, match="cannot accept the episode"):
        app.check()(cast(hflow.steps.CheckFunction, no_episode))

    assert app.checks == []


def test_check_with_only_kwargs_fails_at_registration() -> None:
    app = hflow.App("signature-kwargs-only", data_root=Path("/tmp"))

    def kwargs_only(**kwargs: object) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"n": len(kwargs)})

    with pytest.raises(ValueError, match="cannot accept the episode"):
        app.check()(cast(hflow.steps.CheckFunction, kwargs_only))

    assert app.checks == []


def test_check_with_varargs_registers() -> None:
    app = hflow.App("signature-varargs", data_root=Path("/tmp"))

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
    app = hflow.App("signature-defaulted-episode", data_root=Path("/tmp"))

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
    app = hflow.App(f"guard-{step_kind.split()[0]}", data_root=Path("/tmp"))

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
    app = hflow.App("wrapper-example", data_root=Path("/tmp"))

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
    app = hflow.App("partial-registration", data_root=Path("/tmp"))
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

    app = hflow.App("fetchless", data_root=data_root)
    app.process("landing/e.mcap", record=True)

    data_root.delete("landing/e.mcap")
    # A different worker: same bucket, fresh mirror (no cached source).
    fresh_root = BucketStorageRoot(f"file://{remote_dir}", mirror=tmp_path / "mirror-b")
    relabel_app = hflow.App("fetchless", data_root=fresh_root)
    report = relabel_app.process("landing/e.mcap", record=True, stages={hflow.Stage.LABELS})
    assert report.stamps.pipeline_version


def test_missing_artifact_is_the_steps_error_not_the_runs(tmp_path: Path) -> None:
    from hflow.testing import SyntheticEpisodeSpec, synthesize_episode

    episode_file = synthesize_episode(
        tmp_path / "e.mcap", SyntheticEpisodeSpec(cameras=(), black_segment=None, duration_s=2.0)
    )
    app = hflow.App("artifact-crash", data_root=tmp_path / "data")

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
    app = hflow.App("vantage", data_root=data_root)
    full_report = app.process(episode_file.resolve(), record=True)

    monkeypatch.chdir(tmp_path)
    relative_app = hflow.App("vantage", data_root=Path("data"))
    relabel_report = relative_app.process(
        "data/landing/e.mcap", record=True, stages={hflow.Stage.LABELS}
    )
    assert relabel_report.canonical_path.resolve() == full_report.canonical_path.resolve()
