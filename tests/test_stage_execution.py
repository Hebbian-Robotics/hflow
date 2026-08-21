"""Stage-run semantics (hflow.stage_execution): the library-owned contract
the generated DAGs are thin callers of -- lane planning, pipeline loading,
per-episode accounting, and the error/quarantine budgets."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pytest

import hflow
from hflow.stage_execution import (
    StageBatchCounts,
    load_pipeline_application,
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


class TestEpisodeReferences:
    def test_bucket_root_joins_as_url_and_local_as_path(self) -> None:
        assert (
            resolve_episode_reference("gs://bucket/prefix", "episodes-in/a.mcap")
            == "gs://bucket/prefix/episodes-in/a.mcap"
        )
        assert resolve_episode_reference("/opt/airflow/data", "episodes-in/a.mcap") == Path(
            "/opt/airflow/data/episodes-in/a.mcap"
        )


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


@dataclass
class _StubReport:
    has_errors: bool = False
    quarantined: bool = False


@dataclass
class _StubApp:
    """A processing double at the App boundary: records calls, scripts outcomes."""

    data_root: str
    outcomes: dict[str, object] = field(default_factory=dict)
    processed_references: list[object] = field(default_factory=list)

    def process(self, episode_reference: object, *, record: bool, stages: set[str]) -> _StubReport:
        assert record is True
        assert stages == {"meta"}
        self.processed_references.append(episode_reference)
        outcome = self.outcomes.get(Path(str(episode_reference)).name, "ok")
        if outcome == "crash":
            raise RuntimeError("episode exploded")
        return _StubReport(has_errors=outcome == "step-error", quarantined=outcome == "quarantined")


def test_process_stage_batch_counts_every_outcome_kind(tmp_path: Path) -> None:
    stub_app = _StubApp(
        data_root=str(tmp_path),
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
