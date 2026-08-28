"""Per-episode stage planning: re-ingest costs only what is not already current.

Every test here asserts what a stage DID -- episodes processed, catalog rows
written -- rather than what the planner returned, so the plan's shape can be
reworked without rewriting them. The one exception is the source-changed test,
which is about ordering: it exists because a planner that ran before `sync`
instead of after it would pass every other test in this file.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import hflow
from hflow.cli import main as cli_main
from hflow.curation import open_catalog_connection
from hflow.stage_execution import StageOutcome, run_stages_directly
from hflow.stage_planning import StageSelection
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode

EPISODE_URI = "episodes-in/episode_0001.mcap"

PIPELINE_SOURCE = """
import hflow
from hflow.checks import episode_duration

app = hflow.App("planning-demo")
app.check(version="1")(episode_duration)
"""

# The configuration docs/how-to/enable-built-in-checks.md teaches: wrap a
# built-in under a name of your own to configure it, and the auto-registered
# copy stands down and records `skipped` on every episode, forever.
PIPELINE_THAT_SUPERSEDES_A_DEFAULT = """
import hflow
from hflow.checks import episode_duration

app = hflow.App("planning-demo")


@app.check(version="1")
def my_duration(ep: hflow.Episode) -> hflow.CheckResult:
    return episode_duration(ep)
"""

PIPELINE_WITHOUT_DEFAULT_CHECKS = """
import hflow

app = hflow.App("planning-demo", default_checks=())
"""


@pytest.fixture(scope="module")
def source_episode(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return synthesize_episode(
        tmp_path_factory.mktemp("planning-source") / "episode_0001.mcap",
        SyntheticEpisodeSpec(duration_s=1.0, cameras=()),
    )


@pytest.fixture
def project(tmp_path: Path, source_episode: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data_root = tmp_path / "data"
    episodes_in = data_root / "episodes-in"
    episodes_in.mkdir(parents=True)
    (episodes_in / "episode_0001.mcap").write_bytes(source_episode.read_bytes())
    (tmp_path / "pipeline.py").write_text(PIPELINE_SOURCE)
    (tmp_path / "hflow.toml").write_text('data_root = "./data"\n')
    monkeypatch.delenv("HFLOW_DATA_ROOT", raising=False)
    monkeypatch.delenv("HFLOW_AIRFLOW_URL", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _ingest(
    project: Path, *uris: str, selection: StageSelection | None = None
) -> list[StageOutcome]:
    """One in-process ingest of the project's own pipeline."""
    app = hflow.import_pipeline_application(str(project / "pipeline.py"))
    return run_stages_directly(
        app,
        list(uris) or [EPISODE_URI],
        hflow.RUN_PROFILES["full"],
        **({"selection": selection} if selection is not None else {}),
    )


def _stage(outcomes: list[StageOutcome], stage: hflow.Stage) -> StageOutcome:
    return next(outcome for outcome in outcomes if outcome.stage is stage)


class TestReingestingAnUnchangedCorpus:
    def test_the_checks_do_not_run_a_second_time(self, project: Path) -> None:
        """The whole point: an ffmpeg decode pass per camera, not spent again
        to re-measure bytes that are already measured."""
        _ingest(project)

        second = _ingest(project)

        assert _stage(second, hflow.Stage.META).counts["processed"] == 0
        assert _stage(second, hflow.Stage.META).skipped_as_current == 1

    def test_sync_still_runs_and_still_reuses(self, project: Path) -> None:
        """Sync is deliberately never planned away: it is the stage that proves
        the canonical file is there, and it is already its own cache. A planner
        that skipped it would turn a cleaned run directory into a
        FileNotFoundError three stages later."""
        _ingest(project)

        second = _ingest(project)

        assert _stage(second, hflow.Stage.SYNC).counts["processed"] == 1
        assert _stage(second, hflow.Stage.SYNC).skipped_as_current == 0

    def test_the_corpus_is_unchanged_by_the_second_pass(self, project: Path) -> None:
        _ingest(project)
        _ingest(project)

        connection = open_catalog_connection(project / "data" / "catalog")
        try:
            (episodes,) = connection.execute("SELECT count(*) FROM episodes_latest").fetchone() or (
                0,
            )
            measured = connection.execute(
                "SELECT count(*) FROM measurements_latest WHERE key = 'duration_s'"
            ).fetchone() or (0,)
        finally:
            connection.close()
        assert episodes == 1
        assert measured == (1,)


class TestWhatSchedulesWorkAgain:
    def test_a_check_added_afterwards(self, project: Path) -> None:
        _ingest(project)
        (project / "pipeline.py").write_text(
            PIPELINE_SOURCE
            + """

@app.check(version="1")
def added_later(ep: hflow.Episode) -> hflow.CheckResult:
    return hflow.CheckResult(measurements={"added": 1.0})
"""
        )

        second = _ingest(project)

        assert _stage(second, hflow.Stage.META).counts["processed"] == 1
        assert _stage(second, hflow.Stage.META).skipped_as_current == 0

    def test_a_source_whose_bytes_changed_underneath_the_catalog(
        self, project: Path, tmp_path: Path
    ) -> None:
        """The reason the plan is built AFTER sync rather than before it.

        Nothing in the catalog hashes the source, so a plan read before sync
        would find this recording's previous episode fully checked and skip
        meta -- leaving a canonical episode in the corpus with another
        episode's measurements attached to it. Built after sync, the catalog
        already names the episode this run just minted, which has none.
        """
        _ingest(project)
        replacement = synthesize_episode(
            tmp_path / "replacement.mcap",
            SyntheticEpisodeSpec(duration_s=2.0, cameras=()),
        )
        (project / "data" / EPISODE_URI).write_bytes(replacement.read_bytes())

        second = _ingest(project)

        assert _stage(second, hflow.Stage.META).counts["processed"] == 1
        connection = open_catalog_connection(project / "data" / "catalog")
        try:
            rows = connection.execute(
                'SELECT "duration_s" FROM episodes WHERE source_uri = ?', [EPISODE_URI]
            ).fetchall()
        finally:
            connection.close()
        assert rows == [(pytest.approx(2.0, abs=0.2),)]

    def test_a_source_the_catalog_has_never_seen(self, project: Path, source_episode: Path) -> None:
        _ingest(project)
        second_uri = "episodes-in/episode_0002.mcap"
        (project / "data" / second_uri).write_bytes(source_episode.read_bytes())

        both = _ingest(project, EPISODE_URI, second_uri)

        assert _stage(both, hflow.Stage.META).counts["processed"] == 1
        assert _stage(both, hflow.Stage.META).skipped_as_current == 1


class TestTheEscapeHatches:
    def test_all_stages_re_runs_everything(self, project: Path) -> None:
        """For the one thing the catalog cannot see: an artifact deleted out
        from under a step whose rows still say it ran."""
        _ingest(project)

        second = _ingest(project, selection=StageSelection.EVERY_STAGE)

        assert _stage(second, hflow.Stage.META).counts["processed"] == 1
        assert _stage(second, hflow.Stage.META).skipped_as_current == 0

    def test_the_cli_exposes_it(self, project: Path, capsys: pytest.CaptureFixture[str]) -> None:
        assert cli_main(["ingest", EPISODE_URI]) == 0
        capsys.readouterr()

        assert cli_main(["ingest", "--all-stages", EPISODE_URI]) == 0

        assert "meta: 1 processed" in capsys.readouterr().out

    def test_the_cli_reports_what_it_skipped(
        self, project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`meta: 0 processed` on its own reads as nothing having happened;
        beside a count of what was already current it reads as nothing left to
        do, which is a different claim."""
        assert cli_main(["ingest", EPISODE_URI]) == 0
        capsys.readouterr()

        assert cli_main(["ingest", EPISODE_URI]) == 0

        assert "meta: 0 processed, 0 quarantined, 0 errors, 1 already current" in (
            capsys.readouterr().out
        )

    def test_a_profile_without_sync_runs_exactly_as_asked(self, project: Path) -> None:
        """Without sync nothing has re-established which episode each recording
        currently is, so there is no fresh ground truth to plan against and the
        stages run as the caller asked."""
        _ingest(project)
        app = hflow.import_pipeline_application(str(project / "pipeline.py"))

        backfill = run_stages_directly(app, [EPISODE_URI], hflow.RUN_PROFILES["metadata_backfill"])

        assert _stage(backfill, hflow.Stage.META).counts["processed"] == 1
        assert _stage(backfill, hflow.Stage.META).skipped_as_current == 0


def test_a_superseded_default_does_not_schedule_meta_forever(project: Path) -> None:
    """A default check the pipeline supersedes records `skipped` on every
    episode, always. Read as unfinished work it would schedule the meta stage
    on every pass, and the corpus would never be up to date."""
    (project / "pipeline.py").write_text(PIPELINE_THAT_SUPERSEDES_A_DEFAULT)
    _ingest(project)

    second = _ingest(project)

    assert _stage(second, hflow.Stage.META).counts["processed"] == 0
    assert _stage(second, hflow.Stage.META).skipped_as_current == 1


def _pipeline_with_a_gate_at(minimum_duration_s: float, *, version: str) -> str:
    """A critical gate whose author bumps its explicit version when retuned."""
    return f"""
import hflow
from hflow.checks import episode_duration

app = hflow.App("planning-demo", default_checks=())


@app.check(version={version!r}, critical=True)
def long_enough(ep: hflow.Episode) -> hflow.CheckResult:
    seconds = float(episode_duration(ep).measurements["duration_s"])
    return hflow.CheckResult(
        measurements={{"seconds": seconds}}, verdict=seconds >= {minimum_duration_s}
    )


@app.enrich(version="1")
def labeller(ep: hflow.Episode) -> hflow.EnrichmentResult:
    return hflow.EnrichmentResult(labels={{"reviewed": 1.0}})
"""


class TestUnQuarantiningAnEpisode:
    """Retuning a critical check is the ordinary way to let an episode back in.

    Every step downstream of the failing gate recorded `skipped` on the
    quarantined pass. Reading that as settled work meant the episode came back
    with no labels, on that pass and on every later one, while the dataset
    policy -- whose quarantine rule now passed too -- shipped it as complete.
    """

    def test_the_enrichment_that_stood_aside_runs_afterwards(self, project: Path) -> None:
        (project / "pipeline.py").write_text(_pipeline_with_a_gate_at(99.0, version="1"))
        quarantined = _ingest(project)
        assert _stage(quarantined, hflow.Stage.META).counts["quarantined"] == 1

        (project / "pipeline.py").write_text(_pipeline_with_a_gate_at(0.1, version="2"))
        released = _ingest(project)

        assert _stage(released, hflow.Stage.META).counts["processed"] == 1
        assert _stage(released, hflow.Stage.LABELS).counts["processed"] == 1
        connection = open_catalog_connection(project / "data" / "catalog")
        try:
            labels = connection.execute(
                "SELECT value_double FROM measurements_latest WHERE key = 'reviewed'"
            ).fetchall()
        finally:
            connection.close()
        assert labels == [(1.0,)]

    def test_the_dataset_does_not_ship_it_until_the_labels_are_there(self, project: Path) -> None:
        """Both dataset rules turn green the moment the gate is retuned, so
        without the distinction the un-labelled episode is selected."""
        from hflow.dataset import create_dataset

        (project / "pipeline.py").write_text(_pipeline_with_a_gate_at(99.0, version="1"))
        _ingest(project)
        (project / "pipeline.py").write_text(_pipeline_with_a_gate_at(0.1, version="2"))
        app = hflow.import_pipeline_application(str(project / "pipeline.py"))
        run_stages_directly(app, [EPISODE_URI], {hflow.Stage.SYNC, hflow.Stage.META})

        # meta cleared the quarantine; labels has not run yet.
        assert create_dataset(app, "too-early").row_count == 0

        _ingest(project)
        assert create_dataset(app, "complete").row_count == 1


class TestARecordingSyncCouldNotCanonicalize:
    """Sync appends a row for everything it canonicalizes, so after it has run,
    "no episode in the catalog" means it failed on that recording."""

    def test_the_later_stages_do_not_pile_a_second_failure_on_it(self, project: Path) -> None:
        """Handing it on to meta earns a FileNotFoundError classified
        `infrastructure`, next to the `source-unreadable` row that already told
        the truth about the same file. One failure, one row."""
        (project / "data" / "episodes-in" / "corrupt.mcap").write_bytes(b"not an mcap file")

        outcomes = _ingest(project, EPISODE_URI, "episodes-in/corrupt.mcap")

        assert _stage(outcomes, hflow.Stage.SYNC).counts["errors"] == 1
        assert _stage(outcomes, hflow.Stage.META).counts["errors"] == 0
        connection = open_catalog_connection(project / "data" / "catalog")
        try:
            rows = connection.execute(
                "SELECT stage, failure_kind FROM ingest_failures "
                "WHERE source_uri = 'episodes-in/corrupt.mcap'"
            ).fetchall()
        finally:
            connection.close()
        assert rows == [("sync", "source-unreadable")]

    def test_it_is_not_counted_as_already_current(self, project: Path) -> None:
        """It ran nothing and it is not up to date. Folding it into the
        already-current count would report a corrupt recording as done."""
        (project / "data" / "episodes-in" / "corrupt.mcap").write_bytes(b"not an mcap file")

        outcomes = _ingest(project, EPISODE_URI, "episodes-in/corrupt.mcap")

        assert _stage(outcomes, hflow.Stage.META).skipped_as_current == 0

    def test_the_command_exits_one_even_under_budget(
        self, project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The budget decides whether the RUN keeps going, never whether a
        failure is reported: `hflow ingest ... && next-step` has to see it."""
        (project / "data" / "episodes-in" / "corrupt.mcap").write_bytes(b"not an mcap file")

        assert cli_main(["ingest", EPISODE_URI, "episodes-in/corrupt.mcap"]) == 1

        printed = capsys.readouterr()
        assert "sync: 1 processed, 0 quarantined, 1 errors" in printed.out
        assert "ingest_failures" in printed.err

    def test_a_crashed_check_is_reported_where_it_actually_is(
        self, project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A check that crashes leaves an ordinary catalogued episode with an
        `error` step on it -- nothing in the failure ledger. Pointing only at
        the ledger sends that user looking in an empty table."""
        (project / "pipeline.py").write_text(
            PIPELINE_SOURCE
            + """

@app.check(version="1")
def explodes(ep: hflow.Episode) -> hflow.CheckResult:
    raise RuntimeError("boom")
"""
        )

        assert cli_main(["ingest", EPISODE_URI]) == 1

        assert "check_runs" in capsys.readouterr().err
        connection = open_catalog_connection(project / "data" / "catalog")
        try:
            assert connection.execute("SELECT count(*) FROM ingest_failures").fetchone() == (0,)
            assert connection.execute(
                "SELECT check_name FROM check_runs WHERE status = 'error'"
            ).fetchall() == [("explodes",)]
        finally:
            connection.close()


class TestMediaPlanning:
    """Contact sheets: the other stage that costs a decode pass."""

    @pytest.fixture(scope="class")
    @staticmethod
    def camera_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
        return synthesize_episode(
            tmp_path_factory.mktemp("planning-camera") / "episode_0001.mcap",
            SyntheticEpisodeSpec(duration_s=1.0, cameras=("wrist_cam",), black_segment=None),
        )

    def test_a_rendered_contact_sheet_is_not_rendered_again(
        self, project: Path, camera_source: Path
    ) -> None:
        (project / "data" / EPISODE_URI).write_bytes(camera_source.read_bytes())
        first = _ingest(project)
        assert _stage(first, hflow.Stage.MEDIA).counts["processed"] == 1

        second = _ingest(project)

        assert _stage(second, hflow.Stage.MEDIA).counts["processed"] == 0
        assert _stage(second, hflow.Stage.MEDIA).skipped_as_current == 1

    def test_a_camera_bearing_episode_without_a_contact_sheet_is_planned(
        self, project: Path, camera_source: Path
    ) -> None:
        (project / "data" / EPISODE_URI).write_bytes(camera_source.read_bytes())
        application = hflow.import_pipeline_application(str(project / "pipeline.py"))
        run_stages_directly(
            application,
            [EPISODE_URI],
            {hflow.Stage.SYNC, hflow.Stage.META},
        )

        second = _ingest(project)

        assert _stage(second, hflow.Stage.MEDIA).counts["processed"] == 1
        assert _stage(second, hflow.Stage.MEDIA).skipped_as_current == 0

    def test_a_camera_less_episode_is_not_planned_again(self, project: Path) -> None:
        _ingest(project)

        second = _ingest(project)

        assert _stage(second, hflow.Stage.MEDIA).counts["processed"] == 0
        assert _stage(second, hflow.Stage.MEDIA).skipped_as_current == 1

    def test_disabling_default_checks_keeps_planning_camera_less_media(self, project: Path) -> None:
        (project / "pipeline.py").write_text(PIPELINE_WITHOUT_DEFAULT_CHECKS)
        _ingest(project)

        second = _ingest(project)

        assert _stage(second, hflow.Stage.MEDIA).counts["processed"] == 1
        assert _stage(second, hflow.Stage.MEDIA).skipped_as_current == 0

    def test_an_episode_without_camera_frame_stats_is_still_planned(self, project: Path) -> None:
        (project / "pipeline.py").write_text(PIPELINE_WITHOUT_DEFAULT_CHECKS)
        _ingest(project)
        (project / "pipeline.py").write_text(PIPELINE_SOURCE)

        second = _ingest(project)

        assert _stage(second, hflow.Stage.MEDIA).counts["processed"] == 1
        assert _stage(second, hflow.Stage.MEDIA).skipped_as_current == 0


class TestFilteringAScheduledStagesUris:
    """The scheduled lane asks the same question, in conf vocabulary.

    ``outstanding_stage_uris`` is what a rendered stage sub-DAG calls at plan
    time. It takes the data-root-relative strings a trigger conf carries and
    answers with the subset of them that stage still owes work on, so these
    tests pin the translation as much as the planning: a conf uri that does not
    reduce to the identity the catalog filed under reads as "everything is
    outstanding" and quietly costs a full re-run.
    """

    @staticmethod
    def _application(project: Path) -> hflow.App:
        return hflow.import_pipeline_application(str(project / "pipeline.py"))

    @staticmethod
    def _filter(project: Path, stage: hflow.Stage, *uris: str) -> list[str]:
        from hflow.stage_planning import outstanding_stage_uris

        application = TestFilteringAScheduledStagesUris._application(project)
        return outstanding_stage_uris(
            application,
            list(uris) or [EPISODE_URI],
            stage,
            data_root=str(project / "data"),
        )

    def test_an_unchanged_corpus_leaves_the_stage_nothing_to_do(self, project: Path) -> None:
        """The whole point of the issue: the scheduled lane stops paying for a
        decode pass the catalog already records."""
        _ingest(project)

        assert self._filter(project, hflow.Stage.META) == []

    def test_a_synced_but_unchecked_recording_is_still_owed(self, project: Path) -> None:
        """Sync alone leaves meta outstanding, which is the case the filter has
        to keep: dropping it would silently never run the checks."""
        application = self._application(project)
        run_stages_directly(application, [EPISODE_URI], frozenset({hflow.Stage.SYNC}))

        assert self._filter(project, hflow.Stage.META) == [EPISODE_URI]

    def test_a_check_added_afterwards_makes_it_outstanding_again(self, project: Path) -> None:
        _ingest(project)
        (project / "pipeline.py").write_text(
            PIPELINE_SOURCE
            + """

@app.check(version="1")
def added_later(ep: hflow.Episode) -> hflow.CheckResult:
    return hflow.CheckResult(evidence={})
"""
        )

        assert self._filter(project, hflow.Stage.META) == [EPISODE_URI]

    def test_a_recording_the_catalog_cannot_vouch_for_is_handed_on(self, project: Path) -> None:
        """The one place this differs from the direct planner, which drops such
        a recording because it ran sync itself and so knows the row is missing
        because sync failed. A stage sub-DAG cannot tell that apart from a
        corpus nobody has synced yet, and dropping it would turn the loud
        FileNotFoundError that a meta-only run raises today into a silent skip.
        Filtering is here to stop paying for finished work, not to swallow
        failures."""
        _ingest(project)
        (project / "data" / "episodes-in" / "unreadable.mcap").write_bytes(b"not an mcap")

        assert self._filter(
            project, hflow.Stage.META, EPISODE_URI, "episodes-in/unreadable.mcap"
        ) == ["episodes-in/unreadable.mcap"]

    def test_the_conf_spelling_reaches_the_rows_the_catalog_filed(self, project: Path) -> None:
        """A conf uri is relative to the DATA ROOT, not to the working
        directory, and the two differ in every deployment: a task process runs
        wherever the operator put it. Resolving the wrong one queries for rows
        filed under another name, which reads as "outstanding" for a corpus
        that is entirely up to date."""
        _ingest(project)
        elsewhere = project / "not-the-data-root"
        elsewhere.mkdir()

        with pytest.MonkeyPatch.context() as patched:
            patched.chdir(elsewhere)

            assert self._filter(project, hflow.Stage.META) == []

    def test_sync_is_refused_rather_than_answered(self, project: Path) -> None:
        """Sync records no steps and is its own cache, so an empty answer would
        filter away the stage that produces the file every later stage reads."""
        with pytest.raises(ValueError, match="never planned away"):
            self._filter(project, hflow.Stage.SYNC)


# What a rendered `plan` task is once its Airflow decorator is stripped: conf
# values in, JSON-able batches out.
RenderedPlan = Callable[[list[str], str, int | None, bool | str], list[dict[str, Any]]]


class TestTheRenderedPlanTask:
    """The filter as a stage sub-DAG actually runs it.

    Everything above calls ``outstanding_stage_uris`` directly. What that
    cannot catch is the injected snippet being wrong: it reaches the bundle as
    a string, so a bad name or a missed substitution compiles and only fails
    when a task runs, which needs Airflow and Docker. Extracting the rendered
    ``plan`` body and calling it closes that gap without either, because the
    body's whole contract is that it runs under plain CPython in the user venv.
    """

    @staticmethod
    def _rendered_plan(project: Path, stage: hflow.Stage) -> RenderedPlan:
        import ast

        from hflow.runtime._bundle import render_dag_sources

        sources = render_dag_sources(
            master_dag_id="rendered",
            pipeline_filename="pipeline.py",
            app_variable="app",
            data_root=str(project / "data"),
            venv_python="/unused/python",
        )
        module = ast.parse(sources[f"rendered_{stage.value}"])
        dag_function = next(
            node
            for node in ast.walk(module)
            if isinstance(node, ast.FunctionDef) and node.name == "ingest_stage"
        )
        plan_function = next(
            node
            for node in dag_function.body
            if isinstance(node, ast.FunctionDef) and node.name == "plan"
        )
        # The decorator is Airflow's; the body underneath it is plain Python
        # and is the only part this test is about.
        plan_function.decorator_list = []
        extracted = ast.Module(body=[plan_function], type_ignores=[])
        ast.fix_missing_locations(extracted)
        namespace: dict[str, RenderedPlan] = {}
        exec(compile(extracted, "<rendered plan>", "exec"), namespace)
        return namespace["plan"]

    @pytest.fixture
    def plan(self, project: Path, monkeypatch: pytest.MonkeyPatch) -> RenderedPlan:
        monkeypatch.setenv("HFLOW_USER_DIR", str(project))
        return self._rendered_plan(project, hflow.Stage.META)

    def test_an_unchanged_corpus_skips_the_whole_stage(
        self, project: Path, plan: RenderedPlan
    ) -> None:
        """Exit 99 is what skip_on_exit_code turns into a SKIPPED task, so the
        UI shows a stage that had nothing to do rather than a hollow success."""
        _ingest(project)

        with pytest.raises(SystemExit) as excinfo:
            plan([EPISODE_URI], "batch", None, False)

        assert excinfo.value.code == 99

    def test_all_stages_hands_the_whole_batch_over(self, project: Path, plan: RenderedPlan) -> None:
        _ingest(project)

        batches = plan([EPISODE_URI], "batch", None, True)

        assert [item for batch in batches for item in batch["items"]] == [EPISODE_URI]

    def test_a_conf_string_reads_as_the_flag_it_spells(
        self, project: Path, plan: RenderedPlan
    ) -> None:
        """Airflow renders params into the call, and a hand-typed conf value
        arrives as text. Reading "true" as a true value is what keeps the
        escape hatch usable from the trigger form."""
        _ingest(project)

        batches = plan([EPISODE_URI], "batch", None, "true")

        assert [item for batch in batches for item in batch["items"]] == [EPISODE_URI]

    @pytest.mark.parametrize("spelling", ["false", "no", "off", "0", "", "   "])
    def test_a_conf_string_that_spells_no_leaves_the_filter_on(
        self, project: Path, plan: RenderedPlan, spelling: str
    ) -> None:
        """The direction where a regression is silent.

        Reading a false spelling as true does not fail anything: the stage
        simply processes everything, which is what it did before this filter
        existed. So the corpus stays correct and the run just costs what #172
        was about, forever, with nothing saying so. Truthiness on the raw
        string (``bool(all_stages)``) is the obvious wrong implementation and
        every spelling here survives it.
        """
        _ingest(project)

        with pytest.raises(SystemExit) as excinfo:
            plan([EPISODE_URI], "batch", None, spelling)

        assert excinfo.value.code == 99
