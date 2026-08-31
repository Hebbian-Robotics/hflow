"""Stage-run semantics (hflow.stage_execution): the library-owned contract
the generated DAGs are thin callers of -- lane planning, pipeline loading,
per-episode accounting, and the error/quarantine budgets."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, cast

import pytest

import hflow
from hflow import stage_execution
from hflow.runtime import parse_data_root_relative_uri
from hflow.stage_execution import (
    StageBatchCounts,
    load_pipeline_application,
    parse_conf_flag,
    plan_stage_batches,
    process_stage_batch,
    require_application_data_root,
    resolve_episode_reference,
    resolve_user_pipeline_path,
    run_failure_budget,
    summarize_error_budget,
    summarize_quarantine_budget,
)


class TestBudgets:
    def test_budget_floor_and_fraction(self) -> None:
        assert run_failure_budget(0) == 8
        assert run_failure_budget(100) == 8
        assert run_failure_budget(10_000) == 100

    def test_errors_within_budget_pass_and_report_totals(self) -> None:
        summary = summarize_error_budget(
            [
                {"processed": 95, "quarantined": 2, "errors": 2},
                {"processed": 1, "quarantined": 0, "errors": 0},
            ]
        )
        assert summary == {"total": 100, "errors": 2, "budget": 8}

    def test_errors_over_budget_fail_loudly(self) -> None:
        with pytest.raises(RuntimeError, match="9 of 100 episodes had processing errors"):
            summarize_error_budget([{"processed": 91, "quarantined": 0, "errors": 9}])

    def test_all_errors_always_fails_even_under_budget(self) -> None:
        with pytest.raises(RuntimeError, match="2 of 2 episodes"):
            summarize_error_budget([{"processed": 0, "quarantined": 0, "errors": 2}])

    def test_quarantine_budget_applies_only_in_the_quarantine_summary(self) -> None:
        over_quarantine: list[StageBatchCounts] = [
            {"processed": 80, "quarantined": 20, "errors": 0}
        ]
        # The error gate treats quarantined episodes as part of the total.
        assert summarize_error_budget(over_quarantine)["errors"] == 0
        with pytest.raises(RuntimeError, match="quarantine is a tag, never a deletion"):
            summarize_quarantine_budget(over_quarantine)

    def test_quarantine_summary_still_budgets_errors(self) -> None:
        with pytest.raises(RuntimeError, match="processing errors"):
            summarize_quarantine_budget([{"processed": 0, "quarantined": 0, "errors": 9}])


class TestLanePlanning:
    def test_online_is_one_immediate_batch(self, tmp_path: Path) -> None:
        batches = plan_stage_batches(
            ["a.mcap", "b.mcap"], mode="online", batch_count=3, data_root=str(tmp_path)
        )
        assert batches == [{"items": ["a.mcap", "b.mcap"], "start_delay_s": 0.0}]

    def test_batch_lane_bin_packs_and_staggers_by_file_size(self, tmp_path: Path) -> None:
        for name, size in (("big.mcap", 4000), ("small-1.mcap", 100), ("small-2.mcap", 100)):
            (tmp_path / name).write_bytes(b"x" * size)
        batches = plan_stage_batches(
            ["big.mcap", "small-1.mcap", "small-2.mcap"],
            mode="batch",
            batch_count=2,
            data_root=str(tmp_path),
        )
        assert len(batches) == 2
        # First-fit-decreasing: the big file alone, the small ones together.
        items_by_batch = sorted(batches, key=lambda batch: len(batch["items"]))
        assert items_by_batch[0]["items"] == ["big.mcap"]
        assert sorted(items_by_batch[1]["items"]) == ["small-1.mcap", "small-2.mcap"]
        assert {batch["start_delay_s"] for batch in batches} == {0.0, 2.0}

    def test_empty_uris_plan_nothing(self, tmp_path: Path) -> None:
        assert plan_stage_batches([], mode="batch", batch_count=None, data_root=str(tmp_path)) == []

    def test_unknown_mode_is_refused_loudly(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown mode"):
            plan_stage_batches(["a.mcap"], mode="turbo", batch_count=None, data_root=str(tmp_path))


class TestConfFlags:
    """The conf vocabulary's booleans, parsed where mode is parsed."""

    @pytest.mark.parametrize("spelling", ["1", "true", "TRUE", " yes ", "on"])
    def test_the_spellings_that_mean_yes(self, spelling: str) -> None:
        assert parse_conf_flag(spelling) is True

    @pytest.mark.parametrize("spelling", ["0", "false", "FALSE", "no", "off", "", "   "])
    def test_every_other_string_means_no(self, spelling: str) -> None:
        """The direction where a regression is silent. ``bool()`` on the raw
        string reads all of these as true, and a flag that gates a filter then
        turns the filter off without anything failing."""
        assert parse_conf_flag(spelling) is False

    def test_a_native_value_is_taken_as_it_stands(self) -> None:
        """render_template_as_native_obj hands a real bool straight through,
        and an unset param arrives as None."""
        assert parse_conf_flag(True) is True
        assert parse_conf_flag(False) is False
        assert parse_conf_flag(None) is False


class TestEpisodeReferences:
    def test_bucket_root_joins_as_url_and_local_as_path(self) -> None:
        uri = parse_data_root_relative_uri("episodes-in/a.mcap")
        assert (
            resolve_episode_reference("gs://bucket/prefix", uri)
            == "gs://bucket/prefix/episodes-in/a.mcap"
        )
        assert resolve_episode_reference("/opt/airflow/data", uri) == Path(
            "/opt/airflow/data/episodes-in/a.mcap"
        )

    def test_bucket_reference_does_not_repair_an_invalid_leading_slash(self) -> None:
        with pytest.raises(ValueError, match="relative to the data root"):
            parse_data_root_relative_uri("/episodes-in/a.mcap")


class TestPipelineLoading:
    def test_user_dir_environment_variable_relocates_the_pipeline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("HFLOW_USER_DIR", raising=False)
        assert resolve_user_pipeline_path("pipeline.py") == "/opt/user/pipeline.py"
        monkeypatch.setenv("HFLOW_USER_DIR", "/mnt/synced/user/")
        assert resolve_user_pipeline_path("pipeline.py") == "/mnt/synced/user/pipeline.py"

    def test_loads_the_app_and_enforces_the_data_root_contract(self, tmp_path: Path) -> None:
        pipeline_file = tmp_path / "pipeline.py"
        pipeline_file.write_text(
            "import hflow\n\nmy_app = hflow.App('demo', data_root='/opt/airflow/data')\n"
        )
        app = load_pipeline_application(str(pipeline_file), "my_app")
        assert app.name == "demo"
        require_application_data_root(app, "/opt/airflow/data")
        with pytest.raises(RuntimeError, match="data_root must be /elsewhere"):
            require_application_data_root(app, "/elsewhere")

    def test_missing_app_variable_is_a_loud_contract_failure(self, tmp_path: Path) -> None:
        pipeline_file = tmp_path / "pipeline.py"
        pipeline_file.write_text("value = 42\n")
        with pytest.raises(RuntimeError, match=r"no hflow\.App named 'app'"):
            load_pipeline_application(str(pipeline_file), "app")

    def test_a_pipeline_can_import_a_sibling_module(self, tmp_path: Path) -> None:
        # A pipeline is an ordinary multi-file project: `python pipeline.py`
        # puts its directory on sys.path, and loading the same file BY PATH
        # (which is all the runtime ever does) has to agree.
        (tmp_path / "rig_constants.py").write_text("FLEET_NAME = 'kitchen'\n")
        pipeline_file = tmp_path / "pipeline.py"
        pipeline_file.write_text(
            "import hflow\n"
            "from rig_constants import FLEET_NAME\n\n"
            "app = hflow.App(FLEET_NAME, data_root='/opt/airflow/data')\n"
        )
        assert load_pipeline_application(str(pipeline_file), "app").name == "kitchen"

    def test_a_pipeline_exiting_at_import_fails_the_task_diagnosably(self, tmp_path: Path) -> None:
        # SystemExit is a BaseException: a module-scope sys.exit() (a config
        # guard, a stray argparse) must not walk past the loader and take the
        # worker down with the pipeline's own exit status.
        pipeline_file = tmp_path / "pipeline.py"
        pipeline_file.write_text("import sys\n\nsys.exit('set ROBOT_FLEET')\n")
        with pytest.raises(RuntimeError, match="set ROBOT_FLEET"):
            load_pipeline_application(str(pipeline_file), "app")


@dataclass
class _StubReport:
    has_errors: bool = False
    quarantined: bool = False


@dataclass
class _StubWorkspace:
    catalog_root: Path


@dataclass
class _StubApp:
    """A processing double at the App boundary: records calls, scripts outcomes."""

    data_root: str
    workspace: _StubWorkspace
    expected_stage: str = "meta"
    outcomes: dict[str, object] = field(default_factory=dict)
    processed_references: list[object] = field(default_factory=list)
    received_histories: list[object] = field(default_factory=list)
    received_run_ids: list[str | None] = field(default_factory=list)

    def process(
        self,
        episode_reference: object,
        *,
        record: bool,
        stages: set[str],
        quarantine_history: object = None,
        orchestrator_run_id: str | None = None,
    ) -> _StubReport:
        assert record is True
        assert stages == {self.expected_stage}
        self.processed_references.append(episode_reference)
        self.received_histories.append(quarantine_history)
        self.received_run_ids.append(orchestrator_run_id)
        outcome = self.outcomes.get(Path(str(episode_reference)).name, "ok")
        if outcome == "crash":
            raise RuntimeError("episode exploded")
        return _StubReport(has_errors=outcome == "step-error", quarantined=outcome == "quarantined")


def test_process_stage_batch_counts_every_outcome_kind(tmp_path: Path) -> None:
    stub_app = _StubApp(
        data_root=str(tmp_path),
        workspace=_StubWorkspace(catalog_root=tmp_path / "catalog"),
        outcomes={
            "good.mcap": "ok",
            "quarantined.mcap": "quarantined",
            "steperror.mcap": "step-error",
            "crash.mcap": "crash",
        },
    )
    counts = process_stage_batch(
        cast("hflow.App", stub_app),  # a double at the App boundary
        ["good.mcap", "quarantined.mcap", "steperror.mcap", "crash.mcap"],
        "meta",
    )
    # Per-episode crashes are counted, never batch-fatal: all four were tried.
    assert len(stub_app.processed_references) == 4
    assert counts == {"processed": 1, "quarantined": 1, "errors": 2}
    # No orchestrator, so nothing is claimed about one.
    assert stub_app.received_run_ids == [None, None, None, None]


def test_the_orchestrator_run_id_reaches_every_episode_in_the_batch(tmp_path: Path) -> None:
    """One run id for the whole batch, recorded per episode.

    The generated DAGs bind it once with ``partial()`` across the whole
    fan-out, so every episode a mapped instance touches has to carry it or the
    catalog could not answer which run produced a given row.
    """
    stub_app = _StubApp(
        data_root=str(tmp_path),
        workspace=_StubWorkspace(catalog_root=tmp_path / "catalog"),
        outcomes={"crash.mcap": "crash"},
    )

    process_stage_batch(
        cast("hflow.App", stub_app),  # a double at the App boundary
        ["good.mcap", "crash.mcap", "also_good.mcap"],
        "meta",
        "scheduled__2026-08-23T00:00:00+00:00",
    )

    # Including the episode that crashed: it was still attempted under this
    # run, and the id is handed over before the outcome is known.
    assert stub_app.received_run_ids == ["scheduled__2026-08-23T00:00:00+00:00"] * 3


class _CountingQuarantineHistory:
    """Stands in for the real reader so opens can be counted, not timed."""

    opened: ClassVar[list[object]] = []

    def __init__(self, catalog_root: object) -> None:
        type(self).opened.append(catalog_root)

    def quarantine_tags(self, episode_id: str) -> None:
        return None

    def __enter__(self) -> "_CountingQuarantineHistory":
        return self

    def __exit__(self, *_exception: object) -> None:
        return None


@pytest.fixture
def counting_quarantine_history(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    _CountingQuarantineHistory.opened = []
    monkeypatch.setattr(stage_execution, "QuarantineHistory", _CountingQuarantineHistory)
    return _CountingQuarantineHistory.opened


@pytest.mark.parametrize("stage_name", ["sync", "labels", "media"])
def test_a_gated_stage_opens_one_quarantine_reader_for_the_whole_batch(
    tmp_path: Path, counting_quarantine_history: list[object], stage_name: str
) -> None:
    """The quarantine gate reads the catalog once per batch, not per episode.

    Every stage but ``meta`` asks the catalog for the episode's recorded
    quarantine state. Opening that reader per episode makes a bucket catalog
    re-sync its mirror once per episode for one boolean, which is the whole
    cost this batching exists to remove -- so count the opens.
    """
    catalog_root = tmp_path / "catalog"
    stub_app = _StubApp(
        data_root=str(tmp_path),
        workspace=_StubWorkspace(catalog_root=catalog_root),
        expected_stage=stage_name,
    )
    counts = process_stage_batch(
        cast("hflow.App", stub_app),
        [f"episode_{index}.mcap" for index in range(5)],
        stage_name,
    )

    assert counts == {"processed": 5, "quarantined": 0, "errors": 0}
    assert counting_quarantine_history == [catalog_root]
    # ...and every episode was handed that one reader, so none of them falls
    # back to opening its own inside App.process.
    assert stub_app.received_histories == [stub_app.received_histories[0]] * 5
    assert isinstance(stub_app.received_histories[0], _CountingQuarantineHistory)


def test_the_meta_stage_never_opens_the_quarantine_reader(
    tmp_path: Path, counting_quarantine_history: list[object]
) -> None:
    """Meta decides quarantine in memory, so reading it back would be a lie."""
    stub_app = _StubApp(
        data_root=str(tmp_path),
        workspace=_StubWorkspace(catalog_root=tmp_path / "catalog"),
        expected_stage="meta",
    )
    process_stage_batch(cast("hflow.App", stub_app), ["a.mcap", "b.mcap"], "meta")

    assert counting_quarantine_history == []
    assert stub_app.received_histories == [None, None]


def test_an_unknown_stage_is_refused_before_any_episode_is_touched(tmp_path: Path) -> None:
    stub_app = _StubApp(
        data_root=str(tmp_path), workspace=_StubWorkspace(catalog_root=tmp_path / "catalog")
    )
    with pytest.raises(ValueError, match="not a valid Stage"):
        process_stage_batch(cast("hflow.App", stub_app), ["a.mcap"], "nonsense")
    assert stub_app.processed_references == []
