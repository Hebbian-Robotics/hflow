"""Which stages a recording still needs, asked of the catalog rather than redone.

Re-ingesting a corpus used to redo every stage on every episode: the
transcode, the ffmpeg decode the camera checks share, the contact sheets.
``sync`` already stopped redoing the transcode when the source bytes, the
pipeline version and the encoder all match what it last recorded (see
:meth:`hflow.App.process`). This module does the same for the rest of the
graph.

It is one question, asked per episode and per step: **has this step already
recorded an outcome against this exact episode content, at the version the
step has now?** Both halves are content hashes and neither is an ordered
comparison. ``episode_id`` hashes the canonical bytes, so an episode the
transform would now produce differently is a different id and nothing filed
under the old one counts. ``check_version`` hashes the step and the
first-party code it reaches, so a retuned threshold is a different version and
its old rows stop counting. A step is current or it is not.

Two design decisions carry the correctness of the whole module.

**The plan is computed after ``sync`` has run, never before.** Only then does
``episodes_latest`` name the episode this run actually produced, so the
outstanding-steps question is asked against the right content id. Planning
first would read the PREVIOUS generation's id, and would skip ``meta`` for a
recording whose source bytes had changed underneath it -- the one thing the
catalog genuinely cannot see, since nothing in it hashes the source. That also
removes any need to compare ``pipeline_version`` here: after sync, the episode
the catalog names for a source IS this run's, by construction.

**``sync`` itself is never planned away.** It is both the producer of the
canonical file every later stage reads and its own cache, so leaving it in
means the planner reasons only about whether WORK is current, never about
whether a FILE still exists -- which the catalog cannot answer either. A run
directory someone cleaned out costs one transcode, not a ``FileNotFoundError``
in the middle of the meta stage.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from hflow.steps import SETTLED_STATUSES, Stage

if TYPE_CHECKING:
    import duckdb

    from hflow.app import App


class StageSelection(StrEnum):
    """Whether a direct run re-does work the catalog already records as current.

    A run's own vocabulary rather than a flag, because the two modes are
    genuinely different requests and a caller should have to name which one it
    means. It is deliberately NOT conf vocabulary like :class:`hflow.steps.Stage`
    -- the planner is direct-executor only (see :func:`plan_outstanding_stages`),
    so nothing crosses into a rendered DAG.
    """

    # Per episode, run only the stages whose steps the catalog does not already
    # record as settled at their current versions.
    OUTSTANDING = "outstanding"
    # Every stage of the profile on every episode, whatever the catalog says.
    # The escape hatch for the one thing the catalog cannot see: an artifact
    # deleted out from under a recorded step (a cleaned `media/` directory
    # still has its `media/contact_sheet` rows).
    EVERY_STAGE = "every-stage"


@dataclass(frozen=True)
class EpisodeStagePlan:
    """The stages one source recording still needs, and the steps that ask for them."""

    source_identity: str
    stages: frozenset[Stage]
    # Named so a report can say WHY a stage is scheduled. Empty exactly when
    # `stages` is: one fact, one owner, and the caller never has to reconcile
    # a stage set against a step list that disagrees with it.
    outstanding_steps: tuple[str, ...]


def _required_steps_by_stage(
    application: "App", candidate_stages: Iterable[Stage]
) -> dict[Stage, tuple[tuple[str, str], ...]]:
    """The ``(name, version)`` pairs each stage is responsible for recording.

    ``sync`` records no step rows -- it produces the canonical episode, and
    that is what its own reuse gate covers -- so it never appears here and is
    never planned away.
    """
    from hflow.app import MEDIA_CONTACT_SHEET_STEP_NAME, media_contact_sheet_step_version

    by_stage: dict[Stage, tuple[tuple[str, str], ...]] = {
        Stage.META: tuple(
            sorted((registered.name, registered.version) for registered in application.checks)
        ),
        Stage.LABELS: tuple(
            sorted((registered.name, registered.version) for registered in application.enrichments)
        ),
        Stage.MEDIA: ((MEDIA_CONTACT_SHEET_STEP_NAME, media_contact_sheet_step_version()),),
    }
    candidates = frozenset(candidate_stages)
    # Iterated in stage-graph order rather than the caller's, so a plan reads
    # the same way twice regardless of what kind of collection it was handed.
    return {stage: by_stage[stage] for stage in Stage if stage in candidates and stage in by_stage}


def _quote_sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _settled_step_names(
    connection: "duckdb.DuckDBPyConnection",
    episode_ids: Sequence[str],
    identity_pairs: Sequence[tuple[str, str]],
) -> dict[str, set[str]]:
    """Per episode id, the step names settled at exactly the versions given.

    Filtering on ``check_version`` inside the query is what lets the result be
    keyed by name alone downstream: a row that came back is a row at the
    current version, so a name present is a step that needs no rerun.

    Deliberately NOT joined on ``run_fingerprint``. A stage-profile run appends
    check_runs rows only for the steps that ran, under a fresh fingerprint, so
    matching this run's fingerprint would report every other stage's work as
    missing after any partial pass -- and the planner would schedule the whole
    graph forever. The pair ``(episode_id, check_version)`` is the identity
    that matters; which run recorded it is provenance.
    """
    if not episode_ids or not identity_pairs:
        return {}
    episode_placeholders = ", ".join("?" for _ in episode_ids)
    identity_clauses = " OR ".join("(check_name = ? AND check_version = ?)" for _ in identity_pairs)
    statuses = ", ".join(_quote_sql_string(status.value) for status in SETTLED_STATUSES)
    parameters: list[str] = list(episode_ids)
    for name, version in identity_pairs:
        parameters.extend((name, version))
    rows = connection.execute(
        f"""
        SELECT episode_id, check_name FROM check_runs
        WHERE status IN ({statuses})
          AND episode_id IN ({episode_placeholders})
          AND ({identity_clauses})
        GROUP BY episode_id, check_name
        """,
        parameters,
    ).fetchall()
    settled: dict[str, set[str]] = {}
    for episode_id, check_name in rows:
        settled.setdefault(str(episode_id), set()).add(str(check_name))
    return settled


def plan_outstanding_stages(
    application: "App",
    source_identities: Sequence[str],
    candidate_stages: Iterable[Stage],
) -> dict[str, EpisodeStagePlan]:
    """Per source recording, which of ``candidate_stages`` still has work to do.

    ``source_identities`` are catalog ``source_uri`` values -- ask
    :meth:`hflow.App.source_identity` for one rather than spelling it, or the
    query looks for rows filed under another name and every episode reads as
    outstanding.

    A source the catalog has never seen gets every candidate stage: that is the
    ordinary first-ingest case and also the honest answer for a corpus whose
    catalog was rebuilt. So does a source whose latest episode is one this run
    just minted, since nothing has been recorded against those bytes yet.

    Known limitation, stated because it costs work rather than correctness: the
    media stage records nothing at all on a camera-less episode (there is no
    contact sheet to render), so "no row" cannot be told from "nothing to
    render" and such an episode is planned for ``media`` on every pass. The
    stage is a no-op for it beyond opening the canonical, and the alternative
    -- treating a missing row as done -- would silently never render a sheet
    for an episode that wanted one.
    """
    from hflow.curation import open_catalog_connection

    required_by_stage = _required_steps_by_stage(application, candidate_stages)
    every_candidate = frozenset(required_by_stage)
    all_outstanding = {
        source: EpisodeStagePlan(
            source_identity=source,
            stages=every_candidate,
            outstanding_steps=tuple(
                sorted(name for pairs in required_by_stage.values() for name, _ in pairs)
            ),
        )
        for source in source_identities
    }
    if not source_identities or not every_candidate:
        return all_outstanding

    try:
        connection = open_catalog_connection(application.workspace.catalog_root)
    except (FileNotFoundError, ValueError):
        # No catalog to plan against -- every episode of this run failed before
        # anything was appended, or the workspace has none yet. Plan everything
        # and let the stages report their own failures.
        return all_outstanding
    try:
        source_placeholders = ", ".join("?" for _ in source_identities)
        episode_by_source = {
            str(source): str(episode_id)
            for source, episode_id in connection.execute(
                f"SELECT source_uri, episode_id FROM episodes_latest "
                f"WHERE source_uri IN ({source_placeholders})",
                list(source_identities),
            ).fetchall()
        }
        identity_pairs = sorted({pair for pairs in required_by_stage.values() for pair in pairs})
        settled_by_episode = _settled_step_names(
            connection, sorted(set(episode_by_source.values())), identity_pairs
        )
    finally:
        connection.close()

    plans: dict[str, EpisodeStagePlan] = {}
    for source in source_identities:
        episode_id = episode_by_source.get(source)
        if episode_id is None:
            plans[source] = all_outstanding[source]
            continue
        settled = settled_by_episode.get(episode_id, set())
        outstanding_stages: set[Stage] = set()
        outstanding_steps: list[str] = []
        for stage, pairs in required_by_stage.items():
            missing = [name for name, _version in pairs if name not in settled]
            if missing:
                outstanding_stages.add(stage)
                outstanding_steps.extend(missing)
        plans[source] = EpisodeStagePlan(
            source_identity=source,
            stages=frozenset(outstanding_stages),
            outstanding_steps=tuple(sorted(outstanding_steps)),
        )
    return plans
