"""Per-episode stage planning: re-ingest costs only what is not already current.

Every test here asserts what a stage DID -- episodes processed, catalog rows
written -- rather than what the planner returned, so the plan's shape can be
reworked without rewriting them. The one exception is the source-changed test,
which is about ordering: it exists because a planner that ran before `sync`
instead of after it would pass every other test in this file.
"""

from pathlib import Path

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
app.check()(episode_duration)
"""

# The configuration docs/how-to/enable-built-in-checks.md teaches: wrap a
# built-in under a name of your own to configure it, and the auto-registered
# copy stands down and records `skipped` on every episode, forever.
PIPELINE_THAT_SUPERSEDES_A_DEFAULT = """
import hflow
from hflow.checks import episode_duration

app = hflow.App("planning-demo")


@app.check()
def my_duration(ep: hflow.Episode) -> hflow.CheckResult:
    return episode_duration(ep)
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

@app.check()
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


def test_the_budget_applies_to_what_the_stage_actually_ran(project: Path) -> None:
    """A stage handed four outstanding episodes out of four hundred must fail
    when all four error; a budget over the whole corpus would swallow it."""
    _ingest(project)
    (project / "data" / "episodes-in" / "corrupt.mcap").write_bytes(b"not an mcap file")

    with pytest.raises(RuntimeError, match="processing errors"):
        _ingest(project, EPISODE_URI, "episodes-in/corrupt.mcap")


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

    def test_a_camera_less_episode_is_planned_every_time(self, project: Path) -> None:
        """The stated limitation. A camera-less episode records no
        `media/contact_sheet` row at all, so "no row" cannot be told from
        "nothing to render" -- and the alternative reading would silently never
        render a sheet for an episode that wanted one. The stage is a no-op for
        it, so the cost is one canonical open."""
        _ingest(project)

        second = _ingest(project)

        assert _stage(second, hflow.Stage.MEDIA).skipped_as_current == 0
